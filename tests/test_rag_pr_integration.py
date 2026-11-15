"""Contracts spanning recoverable sync, incremental readers and token parents."""

import pytest

from agent.rag.embedding import EmbeddingConfig
from agent.rag.index import open_code_index
from tests.test_token_chunking import token_chunker as token_chunker, token_index as token_index
from tests.test_workspace_rag import make_repo


@pytest.mark.parametrize("strategy", ["lexical", "vector", "hybrid"])
def test_warm_peer_recovers_intent_even_without_generation_change(tmp_path, monkeypatch, strategy):
    pytest.importorskip("chromadb")
    workspace = make_repo(tmp_path)
    writer = open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))
    writer.sync(["src"])
    reader = open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))
    reader.search("torque", strategy=strategy)
    generation = reader.collection.revision_token()
    reads = reader.performance_stats()["collection_full_reads"]
    (workspace.root / "include/controller.hpp").write_text("// pending_scope_marker\n", encoding="utf-8")

    def fail(texts):
        raise RuntimeError("encoder unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(writer.embedder, "embed", fail)
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            writer.sync(["include"])
    assert reader.collection.revision_token() == generation  # No collection mutation yet.
    assert reader._sync_journal.load() is not None
    results = reader.search("pending_scope_marker", strategy=strategy)
    assert results and {item.path for item in results} == {"include/controller.hpp"}
    assert all("pending_scope_marker" in item.content for item in results)
    assert reader.performance_stats()["collection_full_reads"] == reads + 1
    assert reader._sync_journal.load() is None
    assert {parent.path for parent in reader._parents.values()} == {"include/controller.hpp"}
    assert reader.search("torque", path_prefix="src", strategy=strategy) == []
    assert writer.search("pending_scope_marker", strategy=strategy) == results


@pytest.mark.parametrize("rebuild", [False, True])
def test_recovered_token_children_keep_new_parent_context(token_index, monkeypatch, rebuild):
    source, index = token_index
    index.search("TAIL", strategy="lexical")
    old_ids = set(index.collection.data)
    source.write_text(source.read_text(encoding="utf-8").replace("TAIL", "TAIL new_parent"), encoding="utf-8")

    def fail(texts):
        raise RuntimeError("encoder unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(index.embedder, "embed", fail)
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            index.sync([source.name], rebuild=rebuild)
    assert set(index.collection.data) == old_ids
    result = index.search("TAIL", strategy="lexical")[0]
    assert result.parent_context.content == source.read_text(encoding="utf-8")
    assert "new_parent" in result.parent_context.content
    assert index._sync_journal.load() is None
