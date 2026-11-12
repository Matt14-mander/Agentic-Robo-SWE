"""Phase 3 source chunking, incremental indexing and retrieval tests."""

from __future__ import annotations

import math
import os

import pytest

from agent.rag.chunking import chunk_python_file
from agent.rag.cache import RetrievalCache
from agent.rag.index import CodeIndex
from agent.rag.models import SearchResult
from agent.tools._paths import PROJECT_ROOT


class MemoryCollection:
    def __init__(self) -> None:
        self.data = {}
        self.get_calls = 0
        self.query_calls = 0

    def get(self, include):
        self.get_calls += 1
        ids = list(self.data)
        return {
            "ids": ids,
            "documents": [self.data[item][0] for item in ids],
            "metadatas": [self.data[item][1] for item in ids],
        }

    def delete(self, ids):
        for item in ids:
            self.data.pop(item, None)

    def upsert(self, ids, documents, metadatas, embeddings):
        for item, document, metadata, embedding in zip(
            ids, documents, metadatas, embeddings
        ):
            self.data[item] = (document, metadata, embedding)

    def count(self):
        return len(self.data)

    def query(self, query_embeddings, n_results, include):
        self.query_calls += 1
        query = query_embeddings[0]
        ranked = sorted(
            self.data.items(),
            key=lambda item: _cosine_distance(query, item[1][2]),
        )[:n_results]
        return {
            "ids": [[item[0] for item in ranked]],
            "documents": [[item[1][0] for item in ranked]],
            "metadatas": [[item[1][1] for item in ranked]],
            "distances": [[_cosine_distance(query, item[1][2]) for item in ranked]],
        }


def _cosine_distance(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return 1.0 - dot / (left_norm * right_norm) if left_norm and right_norm else 1.0


def test_python_chunker_preserves_symbols_and_line_metadata():
    path = PROJECT_ROOT / "benchmarks/workspaces/_test_rag_chunks.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '"""module docs"""\nimport math\n\n'
        "def normalize_quaternion(q):\n"
        "    return q\n\n"
        "class PIDController:\n"
        "    limit = 1.0\n\n"
        "    def update(self, error):\n"
        "        return error\n",
        encoding="utf-8",
    )
    try:
        chunks = chunk_python_file(path)
    finally:
        path.unlink(missing_ok=True)

    by_symbol = {chunk.symbol: chunk for chunk in chunks}
    assert {"<module>", "normalize_quaternion", "PIDController", "PIDController.update"} <= set(
        by_symbol
    )
    assert by_symbol["normalize_quaternion"].start_line == 4
    assert by_symbol["PIDController.update"].kind == "method"
    assert all(chunk.path == "benchmarks/workspaces/_test_rag_chunks.py" for chunk in chunks)


def test_incremental_index_skips_unchanged_chunks_and_updates_changed_content():
    path = PROJECT_ROOT / "benchmarks/workspaces/_test_rag_incremental.py"
    relative = "benchmarks/workspaces/_test_rag_incremental.py"
    collection = MemoryCollection()
    index = CodeIndex(collection)
    path.write_text("def controller(error):\n    return error\n", encoding="utf-8")
    try:
        first = index.sync([relative])
        second = index.sync([relative])
        path.write_text("def controller(error):\n    return max(-1.0, min(1.0, error))\n", encoding="utf-8")
        third = index.sync([relative])
    finally:
        path.unlink(missing_ok=True)

    assert first.added_or_updated == 1
    assert second.added_or_updated == 0
    assert second.unchanged == 1
    assert third.added_or_updated == 1
    assert collection.count() == 1


def test_hash_retrieval_finds_relevant_symbol_and_filters_path_prefix():
    path = PROJECT_ROOT / "benchmarks/workspaces/_test_rag_search.py"
    relative = "benchmarks/workspaces/_test_rag_search.py"
    collection = MemoryCollection()
    index = CodeIndex(collection)
    path.write_text(
        "def normalize_quaternion(quaternion):\n"
        "    # reject zero norm before unit quaternion normalization\n"
        "    return quaternion\n\n"
        "def update_pid(error):\n"
        "    # clamp actuator saturation and prevent integral windup\n"
        "    return error\n",
        encoding="utf-8",
    )
    try:
        index.sync([relative])
        results = index.search("zero norm quaternion normalization", top_k=1)
        filtered = index.search(
            "quaternion",
            top_k=2,
            path_prefix="src/agent",
        )
    finally:
        path.unlink(missing_ok=True)

    assert results[0].symbol == "normalize_quaternion"
    assert results[0].score > 0
    assert filtered == []


def test_hybrid_retrieval_uses_exact_symbol_and_lexical_evidence():
    path = PROJECT_ROOT / "benchmarks/workspaces/_test_rag_hybrid.py"
    relative = "benchmarks/workspaces/_test_rag_hybrid.py"
    collection = MemoryCollection()
    index = CodeIndex(collection)
    path.write_text(
        "def generic_search(query):\n"
        "    # general vector similarity retrieval\n"
        "    return query\n\n"
        "def search_code_knowledge(query, path_prefix=None):\n"
        "    # ranked hybrid source knowledge results filtered by path prefix\n"
        "    return query\n",
        encoding="utf-8",
    )
    try:
        index.sync([relative])
        results = index.search(
            "search_code_knowledge path prefix ranked results",
            top_k=1,
            strategy="hybrid",
        )
    finally:
        path.unlink(missing_ok=True)

    assert results[0].symbol == "search_code_knowledge"
    assert 0 < results[0].score <= 1


def test_search_rejects_unknown_strategy():
    index = CodeIndex(MemoryCollection())
    with pytest.raises(ValueError, match="Unsupported search strategy"):
        index.search("query", strategy="unknown")


def test_retrieval_cache_normalizes_query_and_tracks_bypass():
    path = PROJECT_ROOT / "benchmarks/workspaces/_test_rag_cache.py"
    relative = "benchmarks/workspaces/_test_rag_cache.py"
    collection = MemoryCollection()
    index = CodeIndex(collection)
    path.write_text(
        "def normalize_quaternion(value):\n"
        "    # reject zero norm before normalization\n"
        "    return value\n",
        encoding="utf-8",
    )
    try:
        index.sync([relative])
        first = index.search("zero norm normalization", top_k=1, strategy="vector")
        second = index.search("  ZERO   NORM normalization  ", top_k=1, strategy="vector")
        index.search("zero norm normalization", top_k=2, strategy="vector")
        index.search(
            "zero norm normalization",
            top_k=1,
            strategy="vector",
            use_cache=False,
        )
    finally:
        path.unlink(missing_ok=True)

    stats = index.cache_stats()
    assert first == second
    assert collection.query_calls == 3
    assert stats.hits == 1
    assert stats.misses == 2
    assert stats.bypasses == 1
    assert stats.hit_rate == pytest.approx(1 / 3, abs=0.0001)
    assert stats.entries == 2


def test_retrieval_cache_invalidates_when_index_revision_changes():
    path = PROJECT_ROOT / "benchmarks/workspaces/_test_rag_cache_revision.py"
    relative = "benchmarks/workspaces/_test_rag_cache_revision.py"
    collection = MemoryCollection()
    index = CodeIndex(collection)
    path.write_text("def controller(error):\n    return error\n", encoding="utf-8")
    try:
        index.sync([relative])
        index.search("controller error", strategy="hybrid")
        old_revision = index.cache_stats().revision
        path.write_text(
            "def controller(error):\n    return max(-1.0, min(1.0, error))\n",
            encoding="utf-8",
        )
        index.sync([relative])
        new_stats = index.cache_stats()
    finally:
        path.unlink(missing_ok=True)

    assert new_stats.revision != old_revision
    assert new_stats.invalidations == 1
    assert new_stats.entries == 0


def test_retrieval_cache_is_lru_bounded():
    path = PROJECT_ROOT / "benchmarks/workspaces/_test_rag_cache_lru.py"
    relative = "benchmarks/workspaces/_test_rag_cache_lru.py"
    collection = MemoryCollection()
    index = CodeIndex(collection, cache=RetrievalCache(max_entries=1))
    path.write_text("def controller(error):\n    return error\n", encoding="utf-8")
    try:
        index.sync([relative])
        index.search("controller", strategy="vector")
        index.search("error", strategy="vector")
    finally:
        path.unlink(missing_ok=True)

    stats = index.cache_stats()
    assert stats.entries == 1
    assert stats.evictions == 1


def test_rag_tool_formats_ranked_source_results(monkeypatch):
    from agent.tools import rag_ops

    class FakeIndex:
        def search(self, query, top_k, path_prefix):
            assert query == "anti windup"
            return [SearchResult(
                path="src/control/pid.py",
                symbol="PIDController.update",
                kind="method",
                start_line=20,
                end_line=31,
                content="def update(self):\n    pass\n",
                score=0.875,
            )]

    monkeypatch.setattr(rag_ops, "get_code_index", lambda: FakeIndex())
    output = rag_ops.search_code_knowledge.invoke({"query": "anti windup"})

    assert "src/control/pid.py:20-31" in output
    assert "PIDController.update" in output
    assert "score=0.875" in output


@pytest.fixture
def live_rag(tmp_path, monkeypatch):
    from agent.rag import chunking
    from agent.rag import index as index_module
    from agent.tools import _paths, rag_ops

    for module in (chunking, index_module, _paths):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "controller.py"
    source.write_text('def controller():\n    return "before"\n', encoding="utf-8")
    collection = MemoryCollection()
    index = CodeIndex(collection)
    index.sync(["controller.py"])
    monkeypatch.setattr(rag_ops, "get_code_index", lambda: index)
    return source, collection, index


@pytest.mark.parametrize("strategy", ["vector", "lexical", "hybrid"])
@pytest.mark.parametrize("use_cache", [True, False])
def test_search_refreshes_after_write_patch(live_rag, strategy, use_cache):
    from agent.tools.write_ops import write_patch

    source, _, index = live_rag
    assert "before" in index.search("controller", strategy=strategy)[0].content
    stat = source.stat()
    status = write_patch.invoke({
        "path": "controller.py", "old_string": '"before"', "new_string": '"after!"',
    })
    assert status.startswith("OK:")
    # Same length and mtime: source freshness must depend on content, not stat alone.
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    results = index.search("controller", strategy=strategy, use_cache=use_cache)
    assert '"after!"' in results[0].content
    assert "before" not in results[0].content
    assert index.cache_stats().invalidations == 1


def test_reopened_index_and_tool_refresh_external_edits(live_rag):
    from agent.tools.rag_ops import search_code_knowledge

    source, collection, index = live_rag
    index.search("controller")
    source.write_text('\n\ndef repaired_controller():\n    return "repaired"\n', encoding="utf-8")
    # Simulate a new process opening an existing index without first calling sync.
    reopened = CodeIndex(collection)
    result = reopened.search("repaired_controller")[0]
    assert result.symbol == "repaired_controller"
    assert result.start_line == 3
    assert collection.count() == 1
    output = search_code_knowledge.invoke({"query": "controller"})
    assert "repaired_controller" in output
    assert "before" not in output


@pytest.mark.parametrize("change", ["delete", "empty"])
def test_refresh_removes_obsolete_chunks_without_losing_other_files(live_rag, change):
    source, collection, index = live_rag
    other = source.with_name("other.py")
    other.write_text("def untouched():\n    return 7\n", encoding="utf-8")
    index.sync(["controller.py", "other.py"])
    index.search("controller")
    if change == "delete":
        source.unlink()
    else:
        source.write_text("", encoding="utf-8")
    results = index.search("controller", top_k=20)
    assert {result.path for result in results} == {"other.py"}
    assert collection.count() == 1


def test_refresh_failure_does_not_serve_cached_source(live_rag, monkeypatch):
    from pathlib import Path
    from agent.tools.rag_ops import search_code_knowledge

    source, _, index = live_rag
    index.search("controller")
    read_text = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == source:
            raise PermissionError("source unavailable")
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", unreadable)
        output = search_code_knowledge.invoke({"query": "controller"})
    assert output.startswith("ERROR:")
    assert "before" not in output
    assert "before" in index.search("controller")[0].content


def test_unchanged_search_reuses_chunks_and_embeddings(live_rag, monkeypatch):
    from agent.rag import chunking

    _, collection, index = live_rag
    index.search("controller", strategy="vector")

    def unexpected(*args, **kwargs):
        pytest.fail("unchanged cached search must not parse or embed source again")

    monkeypatch.setattr(chunking.ast, "parse", unexpected)
    monkeypatch.setattr(index.embedder, "embed", unexpected)
    assert index.search("controller", strategy="vector")
    assert collection.query_calls == 1


def test_failed_index_update_can_retry_without_serving_old_cache(live_rag, monkeypatch):
    from agent.tools.rag_ops import search_code_knowledge

    source, collection, index = live_rag
    index.search("controller")
    source.write_text("def renamed_controller():\n    return 9\n", encoding="utf-8")

    def unavailable(**kwargs):
        raise RuntimeError("index unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(collection, "upsert", unavailable)
        output = search_code_knowledge.invoke({"query": "controller"})
    assert output.startswith("ERROR:")
    assert "before" not in output
    assert index.search("controller")[0].symbol == "renamed_controller"
    assert collection.count() == 1


def test_refresh_rejects_indexed_path_outside_repository(live_rag, monkeypatch):
    from pathlib import Path

    _, collection, index = live_rag
    next(iter(collection.data.values()))[1]["path"] = "../outside.py"

    def unexpected(*args, **kwargs):
        pytest.fail("must reject an escaping path before reading it")

    monkeypatch.setattr(Path, "read_text", unexpected)
    with pytest.raises(ValueError):
        index.search("controller")
