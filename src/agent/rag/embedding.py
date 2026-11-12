"""Local code-semantic embeddings with an explicit, isolated offline hash fallback."""

from __future__ import annotations

import hashlib
import math
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
MODEL_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
MODEL_FILES = {
    "tokenizer.json": "b01c78a902aa4facb2f47f95449f48e2f7bbfea5d2472ee2f6ce92323c6f86e5",
    "model_quantized.onnx": "ed45870251c9f0cf656e78aab0d37a23489066df8a222bb1c8caf8a45f2cb16d",
}
DEFAULT_MODEL_DIR = PROJECT_ROOT / ".agent_state" / "models" / "jina-code-v2"


class Embedder(Protocol):
    dimensions: int

    @property
    def identity(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def describe(self) -> dict: ...


def _identity(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "auto"
    model_dir: Path = DEFAULT_MODEL_DIR
    max_tokens: int = 512
    threads: int = 4

    def __post_init__(self):
        if self.backend not in {"auto", "hash", "semantic"}:
            raise ValueError("RAG embedding backend must be auto, hash or semantic")
        if not 8 <= self.max_tokens <= 8192 or not 1 <= self.threads <= 64:
            raise ValueError("Embedding requires 8..8192 max tokens and 1..64 threads")

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            backend=os.getenv("RAG_EMBEDDING", "auto"),
            model_dir=Path(os.getenv("RAG_EMBEDDING_MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve(),
            max_tokens=int(os.getenv("RAG_EMBEDDING_MAX_TOKENS", "512")),
            threads=int(os.getenv("RAG_EMBEDDING_THREADS", "4")),
        )

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+|\d+(?:\.\d+)?")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Split prose and Python identifiers into normalized retrieval terms."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for camel_part in _CAMEL_RE.sub(" ", raw).split():
            parts = [part for part in camel_part.lower().split("_") if part]
            tokens.extend(parts or [camel_part.lower()])
    return tokens


class HashingEmbedder:
    """Small hashing-vector embedder with no model download or API key."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = dimensions
        self.fallback_reason: str | None = None

    @property
    def identity(self) -> str:
        return _identity({"backend": "hash", "version": 1, "dimensions": self.dimensions})

    def describe(self) -> dict:
        return {"backend": "hash", "version": 1, "dimensions": self.dimensions,
                "identity": self.identity, "fallback_reason": self.fallback_reason}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        features = tokens + [f"{left}::{right}" for left, right in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SemanticCodeEmbedder:
    """Pinned Jina code model, quantized ONNX CPU inference; never downloads or runs remote code."""

    dimensions = 768

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.texts_encoded = 0
        self.truncated_texts = 0
        for name, expected in MODEL_FILES.items():
            path = config.model_dir / name
            if not path.is_file():
                raise RuntimeError("Code embedding files missing; run scripts/download_code_embedding.py")
            if file_sha256(path) != expected:
                raise RuntimeError(f"Code embedding checksum mismatch: {name}")
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Semantic embedding dependencies missing; run uv sync --extra rag") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(config.model_dir / "model_quantized.onnx"), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.tokenizer = Tokenizer.from_file(str(config.model_dir / "tokenizer.json"))
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize code embedding runtime: {type(exc).__name__}") from exc
        self.tokenizer.enable_truncation(max_length=config.max_tokens)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        self.runtime_version = ort.__version__

    @property
    def identity(self) -> str:
        return _identity({"backend": "semantic", "model": MODEL_ID, "revision": MODEL_REVISION,
                          "files": MODEL_FILES, "max_tokens": self.config.max_tokens,
                          "pooling": "masked-mean-l2-v1", "dimensions": self.dimensions})

    def describe(self) -> dict:
        return {"backend": "semantic", "model": MODEL_ID, "revision": MODEL_REVISION,
                "files": MODEL_FILES, "identity": self.identity, "dimensions": self.dimensions,
                "max_tokens": self.config.max_tokens, "pooling": "masked-mean-l2-v1",
                "runtime": f"onnxruntime:{self.runtime_version}:CPUExecutionProvider",
                "threads": self.config.threads, "batch_size": 4,
                "texts_encoded": self.texts_encoded, "truncated_texts": self.truncated_texts,
                "fallback_reason": None}

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        results: list[list[float]] = []
        for offset in range(0, len(texts), 4):
            encoded = self.tokenizer.encode_batch(texts[offset:offset + 4])
            ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
            try:
                hidden = self.session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
            except Exception as exc:
                # Never change vector spaces halfway through an index update or query.
                raise RuntimeError(f"Code embedding inference failed: {type(exc).__name__}") from exc
            if hidden.shape != (*ids.shape, self.dimensions):
                raise RuntimeError("Unexpected code embedding output shape")
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if not np.isfinite(pooled).all() or (norms <= 0).any():
                raise RuntimeError("Invalid code embedding values")
            results.extend((pooled / norms).tolist())
            self.texts_encoded += len(encoded)
            self.truncated_texts += sum(bool(item.overflowing) for item in encoded)
        return results


def create_embedder(config: EmbeddingConfig | None = None) -> Embedder:
    config = config or EmbeddingConfig.from_env()
    if config.backend == "hash":
        return HashingEmbedder()
    try:
        return SemanticCodeEmbedder(config)
    except (RuntimeError, OSError) as exc:
        if config.backend == "semantic":
            raise
        fallback = HashingEmbedder()
        fallback.fallback_reason = str(exc)
        warnings.warn(f"RAG auto embedding fell back to hash: {exc}", RuntimeWarning, stacklevel=2)
        return fallback
