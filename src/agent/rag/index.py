"""Incremental Chroma index for structure-aware source chunks."""

from __future__ import annotations

import math
import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from agent.rag.chunking import chunk_codebase
from agent.rag.cache import RetrievalCache, RetrievalCacheStats
from agent.rag.embedding import HashingEmbedder, tokenize
from agent.rag.models import CodeChunk, IndexStats, SearchResult
from agent.tools._paths import PROJECT_ROOT, resolve_within_root

DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION = "agentic_robo_swe_code"
SearchStrategy = Literal["vector", "lexical", "hybrid"]


@dataclass(frozen=True)
class _Candidate:
    chunk_id: str
    result: SearchResult


class CodeIndex:
    """Backend-neutral index logic operating on a Chroma-compatible collection."""

    def __init__(
        self,
        collection: Any,
        embedder: HashingEmbedder | None = None,
        cache: RetrievalCache | None = None,
    ) -> None:
        self.collection = collection
        self.embedder = embedder or HashingEmbedder()
        self.cache = cache or RetrievalCache()
        self._revision: str | None = None

    def sync(
        self,
        roots: Iterable[str] = ("src",),
        *,
        rebuild: bool = False,
    ) -> IndexStats:
        files, chunks = chunk_codebase(roots)
        existing_payload = self.collection.get(include=["metadatas"])
        existing_ids = [str(item) for item in existing_payload.get("ids") or []]
        existing_metadatas = existing_payload.get("metadatas") or []
        existing_hashes = {
            chunk_id: str(metadata.get("content_hash", ""))
            for chunk_id, metadata in zip(existing_ids, existing_metadatas)
            if isinstance(metadata, dict)
        }
        if rebuild and existing_ids:
            self.collection.delete(ids=existing_ids)
            existing_hashes = {}

        current_ids = {chunk.id for chunk in chunks}
        stale_ids = sorted(set(existing_hashes) - current_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        changed = [
            chunk for chunk in chunks
            if existing_hashes.get(chunk.id) != chunk.content_hash
        ]
        for batch in _batches(changed, 128):
            documents = [chunk.content for chunk in batch]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=documents,
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=self.embedder.embed(documents),
            )

        self._set_revision(_revision_for_chunks(chunks))

        return IndexStats(
            files=len(files),
            chunks=len(chunks),
            added_or_updated=len(changed),
            deleted=len(stale_ids) + (len(existing_ids) if rebuild else 0),
            unchanged=len(chunks) - len(changed),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        path_prefix: str | None = None,
        strategy: SearchStrategy = "hybrid",
        use_cache: bool = True,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if strategy not in {"vector", "lexical", "hybrid"}:
            raise ValueError(f"Unsupported search strategy: {strategy}")
        top_k = min(max(1, top_k), 20)
        revision = self._ensure_revision()
        key = self.cache.make_key(
            revision=revision,
            query=query,
            top_k=top_k,
            path_prefix=path_prefix,
            strategy=strategy,
        )
        if use_cache:
            found, cached = self.cache.get(key)
            if found:
                return cached
        else:
            self.cache.record_bypass()

        started = time.perf_counter()
        results = self._search_uncached(
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            strategy=strategy,
        )
        if use_cache:
            self.cache.put(
                key,
                results,
                compute_ms=(time.perf_counter() - started) * 1000,
            )
        return results

    def _search_uncached(
        self,
        query: str,
        *,
        top_k: int,
        path_prefix: str | None,
        strategy: SearchStrategy,
    ) -> list[SearchResult]:
        count = int(self.collection.count())
        if count == 0:
            return []

        vector_ranked = self._vector_candidates(query, top_k, path_prefix)
        if strategy == "vector":
            return [candidate.result for candidate in vector_ranked[:top_k]]

        lexical_ranked = self._lexical_candidates(query, path_prefix)
        if strategy == "lexical":
            return [candidate.result for candidate in lexical_ranked[:top_k]]

        return self._fuse_rankings(vector_ranked, lexical_ranked, top_k)

    def cache_stats(self) -> RetrievalCacheStats:
        return self.cache.stats()

    def _ensure_revision(self) -> str:
        # Re-read lightweight metadata on every lookup so a long-lived Agent also notices
        # an index refreshed by another process. This is cheaper than stale source results.
        payload = self.collection.get(include=["metadatas"])
        ids = [str(item) for item in payload.get("ids") or []]
        metadatas = payload.get("metadatas") or []
        pairs = [
            (chunk_id, str(metadata.get("content_hash", "")))
            for chunk_id, metadata in zip(ids, metadatas)
            if isinstance(metadata, dict)
        ]
        revision = _revision_for_pairs(pairs)
        if revision != self._revision:
            self._set_revision(revision)
        return revision

    def _set_revision(self, revision: str) -> None:
        self._revision = revision
        self.cache.prepare_revision(revision)

    def _vector_candidates(
        self,
        query: str,
        top_k: int,
        path_prefix: str | None,
    ) -> list[_Candidate]:
        count = int(self.collection.count())
        candidates = count if path_prefix else min(count, max(top_k * 8, 32))
        payload = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=candidates,
            include=["documents", "metadatas", "distances"],
        )
        ids = _first_result_list(payload.get("ids"))
        documents = _first_result_list(payload.get("documents"))
        metadatas = _first_result_list(payload.get("metadatas"))
        distances = _first_result_list(payload.get("distances"))
        results: list[_Candidate] = []
        normalized_prefix = path_prefix.replace("\\", "/") if path_prefix else None
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            if not isinstance(metadata, dict):
                continue
            path = str(metadata.get("path", ""))
            if normalized_prefix and not path.startswith(normalized_prefix):
                continue
            results.append(_Candidate(
                chunk_id=str(chunk_id),
                result=_make_result(document, metadata, 1.0 - float(distance)),
            ))
        return results

    def _lexical_candidates(
        self,
        query: str,
        path_prefix: str | None,
    ) -> list[_Candidate]:
        payload = self.collection.get(include=["documents", "metadatas"])
        ids = [str(item) for item in payload.get("ids") or []]
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        normalized_prefix = path_prefix.replace("\\", "/") if path_prefix else None
        rows: list[tuple[str, str, dict[str, Any]]] = []
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            if not isinstance(metadata, dict):
                continue
            path = str(metadata.get("path", ""))
            if normalized_prefix and not path.startswith(normalized_prefix):
                continue
            rows.append((chunk_id, str(document), metadata))

        query_tokens = list(dict.fromkeys(tokenize(query)))
        if not query_tokens or not rows:
            return []
        document_tokens = [tokenize(document) for _, document, _ in rows]
        document_frequency = {
            token: sum(token in set(tokens) for tokens in document_tokens)
            for token in query_tokens
        }
        average_length = sum(len(tokens) for tokens in document_tokens) / len(document_tokens)
        ranked: list[tuple[float, _Candidate]] = []
        for (chunk_id, document, metadata), tokens in zip(rows, document_tokens):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                frequency_in_docs = document_frequency[token]
                inverse_document_frequency = math.log(
                    1.0 + (len(rows) - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)
                )
                length_ratio = len(tokens) / average_length if average_length else 1.0
                score += inverse_document_frequency * (
                    frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length_ratio))
                )

            query_set = set(query_tokens)
            symbol_tokens = set(tokenize(str(metadata.get("symbol", ""))))
            path_tokens = set(tokenize(str(metadata.get("path", ""))))
            score += 2.5 * len(query_set & symbol_tokens)
            score += 0.6 * len(query_set & path_tokens)
            if score <= 0:
                continue
            ranked.append((score, _Candidate(
                chunk_id=chunk_id,
                result=_make_result(document, metadata, score),
            )))
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))
        maximum = ranked[0][0] if ranked else 1.0
        return [
            _Candidate(candidate.chunk_id, _with_score(candidate.result, score / maximum))
            for score, candidate in ranked
        ]

    @staticmethod
    def _fuse_rankings(
        vector_ranked: list[_Candidate],
        lexical_ranked: list[_Candidate],
        top_k: int,
    ) -> list[SearchResult]:
        """Weighted reciprocal-rank fusion; scores are normalized to [0, 1]."""
        weights = tuple(
            (ranking, weight)
            for ranking, weight in ((vector_ranked, 1.0), (lexical_ranked, 1.25))
            if ranking
        )
        fused: dict[str, float] = {}
        candidates: dict[str, _Candidate] = {}
        for ranking, weight in weights:
            for rank, candidate in enumerate(ranking, start=1):
                candidates[candidate.chunk_id] = candidate
                fused[candidate.chunk_id] = fused.get(candidate.chunk_id, 0.0) + weight / (60 + rank)
        maximum = sum(weight / 61 for _, weight in weights)
        ordered_ids = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]
        return [
            _with_score(candidates[chunk_id].result, fused[chunk_id] / maximum)
            for chunk_id in ordered_ids
        ]


def _batches(items: list[CodeChunk], size: int) -> Iterable[list[CodeChunk]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _revision_for_chunks(chunks: Iterable[CodeChunk]) -> str:
    return _revision_for_pairs((chunk.id, chunk.content_hash) for chunk in chunks)


def _revision_for_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for chunk_id, content_hash in sorted(pairs):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _first_result_list(value: Any) -> list[Any]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return []


def _make_result(document: Any, metadata: dict[str, Any], score: float) -> SearchResult:
    return SearchResult(
        path=str(metadata.get("path", "")),
        symbol=str(metadata.get("symbol", "<unknown>")),
        kind=str(metadata.get("kind", "code")),
        start_line=int(metadata.get("start_line", 1)),
        end_line=int(metadata.get("end_line", 1)),
        content=str(document),
        score=round(score, 6),
    )


def _with_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(
        path=result.path,
        symbol=result.symbol,
        kind=result.kind,
        start_line=result.start_line,
        end_line=result.end_line,
        content=result.content,
        score=round(score, 6),
    )


def open_code_index(
    persist_directory: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> CodeIndex:
    """Open the persistent Chroma collection with a clear optional-dependency error."""
    try:
        import chromadb  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed. Run: uv sync --extra rag"
        ) from exc

    path = Path(persist_directory)
    if not path.is_absolute():
        path = resolve_within_root(str(path))
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return CodeIndex(collection)
