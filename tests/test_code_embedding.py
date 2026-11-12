"""Semantic encoder contracts, offline fallback and vector-space isolation."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.rag import index as index_module
from agent.rag.embedding import (
    DEFAULT_MODEL_DIR, MODEL_FILES, EmbeddingConfig, HashingEmbedder,
    SemanticCodeEmbedder, create_embedder,
)
from agent.rag.index import CodeIndex
from tests.test_rag import MemoryCollection


def test_explicit_hash_never_initializes_semantic_runtime(monkeypatch):
    def unexpected(config):
        pytest.fail("explicit hash must not load a model or access the network")
    monkeypatch.setattr("agent.rag.embedding.SemanticCodeEmbedder", unexpected)
    model = create_embedder(EmbeddingConfig(backend="hash"))
    assert model.describe()["backend"] == "hash"
    assert len(model.embed(["code"])[0]) == 384


def test_missing_weights_fallback_is_reported_but_semantic_mode_is_strict(tmp_path):
    config = EmbeddingConfig(model_dir=tmp_path)
    with pytest.warns(RuntimeWarning, match="fell back to hash"):
        fallback = create_embedder(config)
    assert "missing" in fallback.describe()["fallback_reason"]
    with pytest.raises(RuntimeError, match="missing"):
        create_embedder(replace(config, backend="semantic"))


def test_corrupt_model_is_rejected_before_runtime_load(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        create_embedder(EmbeddingConfig(backend="semantic", model_dir=tmp_path))


def test_encoder_identity_includes_token_budget_but_not_model_location(tmp_path):
    first = SemanticCodeEmbedder.__new__(SemanticCodeEmbedder)
    second = SemanticCodeEmbedder.__new__(SemanticCodeEmbedder)
    first.config = EmbeddingConfig(max_tokens=512)
    second.config = EmbeddingConfig(model_dir=tmp_path, max_tokens=512, threads=2)
    assert first.identity == second.identity
    second.config = replace(second.config, max_tokens=1024)
    assert first.identity != second.identity


def test_masked_pooling_normalization_and_truncation_accounting():
    np = pytest.importorskip("numpy")
    model = SemanticCodeEmbedder.__new__(SemanticCodeEmbedder)
    model.dimensions = 3
    model.texts_encoded = model.truncated_texts = 0
    encoded = [SimpleNamespace(ids=[1, 0], attention_mask=[1, 0], overflowing=[]),
               SimpleNamespace(ids=[1, 2], attention_mask=[1, 1], overflowing=["overflow"])]
    model.tokenizer = SimpleNamespace(encode_batch=lambda texts: encoded)
    model.session = SimpleNamespace(run=lambda outputs, inputs: [np.array(
        [[[3., 4., 0.], [99., 99., 99.]], [[1., 0., 0.], [1., 0., 0.]]]
    )])
    result = model.embed(["one", "two"])
    assert result[0] == pytest.approx([0.6, 0.8, 0.0])
    assert result[1] == pytest.approx([1.0, 0.0, 0.0])
    assert model.truncated_texts == 1 and model.texts_encoded == 2
    assert model.embed([]) == []
    model.session = SimpleNamespace(run=lambda outputs, inputs: [np.full((2, 2, 3), np.nan)])
    with pytest.raises(RuntimeError, match="Invalid"):
        model.embed(["one", "two"])


def test_lexical_search_does_not_encode_query(monkeypatch):
    collection = MemoryCollection()
    index = CodeIndex(collection)
    index.sync(["src/agent/rag/embedding.py"])
    index.search("HashingEmbedder", strategy="lexical")  # Prepare the source snapshot.

    def unexpected(texts):
        pytest.fail("lexical retrieval must not perform vector inference")

    monkeypatch.setattr(index.embedder, "embed", unexpected)
    assert index.search("HashingEmbedder", strategy="lexical", use_cache=False)
    assert collection.query_calls == 0


def test_embedding_namespace_preserves_legacy_hash_collection(tmp_path, monkeypatch):
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    pytest.importorskip("chromadb")

    class OtherEncoder(HashingEmbedder):
        @property
        def identity(self):
            return "other-semantic-space"

    monkeypatch.setattr(index_module, "create_embedder", lambda config: HashingEmbedder())
    hashed = index_module.open_code_index(tmp_path / "db")
    hashed.sync(["src/agent/rag/embedding.py"])
    count = hashed.collection.count()
    # Emulate a legacy collection created before embedding identities were recorded.
    hashed.collection.modify(metadata={"legacy_format": True})
    assert index_module.open_code_index(tmp_path / "db").collection.count() == count
    monkeypatch.setattr(index_module, "create_embedder", lambda config: OtherEncoder())
    other = index_module.open_code_index(tmp_path / "db")
    assert other.collection.name != hashed.collection.name and other.collection.count() == 0
    assert hashed.collection.count() == count
    other.collection.modify(metadata={"embedding_identity": "wrong"})
    with pytest.raises(ValueError, match="embedding identity"):
        index_module.open_code_index(tmp_path / "db")


def test_real_code_model_runs_locally_when_assets_are_available():
    if not all((DEFAULT_MODEL_DIR / name).is_file() for name in MODEL_FILES):
        pytest.skip("Run download_code_embedding.py for the opt-in real-model smoke")
    pytest.importorskip("onnxruntime")
    model = create_embedder(EmbeddingConfig(backend="semantic"))
    vectors = model.embed(["How to sort an array?", "std::sort(items.begin(), items.end());",
                          "Disconnect the network socket."])
    assert all(len(vector) == 768 for vector in vectors)
    assert all(math.sqrt(sum(x*x for x in vector)) == pytest.approx(1, abs=1e-5) for vector in vectors)
    relevant = sum(x*y for x, y in zip(vectors[0], vectors[1]))
    unrelated = sum(x*y for x, y in zip(vectors[0], vectors[2]))
    assert relevant > unrelated
    assert model.describe()["fallback_reason"] is None


def test_comparison_requires_real_semantic_and_records_common_snapshot(tmp_path, monkeypatch):
    from scripts import run_rag_eval as cli
    from agent.rag.evaluation import RagEvalCase, RetrievalTarget
    from agent.tools import _paths

    monkeypatch.setattr(_paths, "PROJECT_ROOT", tmp_path)
    backends = []
    class ExperimentIndex:
        def sync(self, roots, rebuild=False):
            return SimpleNamespace(chunks=1, added_or_updated=1)

        def retrieval_manifest(self):
            return {"source_revision": "fixed", "embedding": {"backend": backends[-1]}}

    def opened(directory, *, workspace, embedding):
        backends.append(embedding.backend)
        return ExperimentIndex()

    monkeypatch.setattr(cli, "open_code_index", opened)
    def evaluated(index, cases, **kwargs):
        return {"metrics": {}, "average_query_latency_ms": 1, "retrieval": index.retrieval_manifest(),
                "results": [{"case_id": "one", "first_relevant_rank": 1}]}, None
    monkeypatch.setattr(cli, "run_rag_evaluation", evaluated)
    args = SimpleNamespace(output=str(tmp_path / "results"), persist_dir=None, roots=None,
                           rebuild=False, top_ks=None)
    cases = [RagEvalCase("one", "query", (RetrievalTarget("file.py"),))]
    assert cli.compare(args, cases, EmbeddingConfig(), None) == 0
    summary = json.loads((tmp_path / "results/comparison.json").read_text())
    assert backends == ["hash", "semantic"]
    assert len(summary["arms"]) == 5 and summary["source_revision"] == "fixed"
    assert summary["status"] == "complete"
    monkeypatch.setattr(ExperimentIndex, "retrieval_manifest", lambda self: {
        "source_revision": backends[-1], "embedding": {"backend": backends[-1]}
    })
    with pytest.raises(RuntimeError, match="Source changed"):
        cli.compare(args, cases, EmbeddingConfig(), None)
    assert json.loads((tmp_path / "results/comparison.json").read_text())["status"] == "incomplete"
