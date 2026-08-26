"""Phase 3 source chunking, incremental indexing and retrieval tests."""

from __future__ import annotations

import math

import pytest

from agent.rag.chunking import chunk_python_file
from agent.rag.index import CodeIndex
from agent.rag.models import SearchResult
from agent.tools._paths import PROJECT_ROOT


class MemoryCollection:
    def __init__(self) -> None:
        self.data = {}

    def get(self, include):
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
