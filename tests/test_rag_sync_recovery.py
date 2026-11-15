"""Fault injection for retryable full syncs, including persistent reopen."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent.rag import chunking, index as index_module
from agent.rag.index import CodeIndex, open_code_index
from agent.tools import _paths
from tests.test_rag import MemoryCollection
from tests.test_workspace_rag import make_repo


@pytest.mark.parametrize("rebuild", [False, True])
def test_in_memory_sync_failure_preserves_source_and_automatically_recovers(
    tmp_path, monkeypatch, rebuild,
):
    for module in (chunking, index_module, _paths):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "controller.py"
    source.write_text("def old_symbol():\n    return 1\n", encoding="utf-8")
    index = CodeIndex(MemoryCollection())
    index.sync([source.name])
    before = deepcopy(index.collection.data)
    source.write_text("def new_symbol():\n    return 2\n", encoding="utf-8")

    def unavailable(texts):
        raise RuntimeError("inference failed")

    with monkeypatch.context() as patch:
        patch.setattr(index.embedder, "embed", unavailable)
        with pytest.raises(RuntimeError, match="inference failed"):
            index.sync([source.name], rebuild=rebuild)
    assert index.collection.data == before
    assert index.search("new_symbol")[0].symbol == "new_symbol"
    assert index.collection.count() == 1


@pytest.fixture
def sync_index(tmp_path, monkeypatch):
    for module in (chunking, index_module, _paths):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "controller.py"
    source.write_text("def old_symbol():\n    return 1\n", encoding="utf-8")
    journal = tmp_path / "state" / "sync.json"
    index = CodeIndex(MemoryCollection(), sync_state_path=journal)
    index.sync(["controller.py"])
    index.search("old_symbol")
    return source, journal, index


def change_sources(source):
    source.write_text("def new_symbol():\n    return 2\n", encoding="utf-8")
    source.with_name("new.py").write_text(
        "\n".join(f"def added_{i}():\n    return {i}\n" for i in range(129)),
        encoding="utf-8",
    )
    return ["controller.py", "new.py"]


def assert_recovered(index, journal):
    assert index.search("added_128", path_prefix="new.py")[0].symbol == "added_128"
    assert index.search("new_symbol", path_prefix="controller.py")[0].symbol == "new_symbol"
    assert index.collection.count() == 130
    assert not journal.exists()


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("failed_batch", [1, 2])
@pytest.mark.parametrize("reopen", [False, True])
def test_inference_failure_preserves_all_rows_and_retries_scope(
    sync_index, monkeypatch, rebuild, failed_batch, reopen,
):
    source, journal, index = sync_index
    before = deepcopy(index.collection.data)
    revision = index.cache_stats().revision
    roots = change_sources(source)
    embed = index.embedder.embed
    calls = 0

    def failing_embed(texts):
        nonlocal calls
        calls += 1
        if calls == failed_batch:
            raise RuntimeError("inference failed")
        return embed(texts)

    with monkeypatch.context() as patch:
        patch.setattr(index.embedder, "embed", failing_embed)
        with pytest.raises(RuntimeError, match="inference failed"):
            index.sync(roots, rebuild=rebuild)
    assert index.collection.data == before
    assert index.cache_stats().revision == revision
    assert journal.exists()
    if reopen:
        index = CodeIndex(index.collection, sync_state_path=journal)
    assert_recovered(index, journal)


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("failure", ["before_first", "after_first", "partial_second"])
def test_partial_upsert_keeps_old_paths_and_retries_after_reopen(
    sync_index, monkeypatch, rebuild, failure,
):
    source, journal, index = sync_index
    old_ids = set(index.collection.data)
    roots = change_sources(source)
    upsert = index.collection.upsert
    calls = 0

    def failing_upsert(**kwargs):
        nonlocal calls
        calls += 1
        if failure == "before_first":
            raise RuntimeError("write failed")
        if failure == "partial_second" and calls == 2:
            upsert(**{key: value[:1] for key, value in kwargs.items()})
            raise RuntimeError("write failed")
        upsert(**kwargs)
        if failure == "after_first":
            raise RuntimeError("write failed")

    with monkeypatch.context() as patch:
        patch.setattr(index.collection, "upsert", failing_upsert)
        with pytest.raises(RuntimeError, match="write failed"):
            index.sync(roots, rebuild=rebuild)
    assert old_ids <= set(index.collection.data)
    assert journal.exists()
    assert_recovered(CodeIndex(index.collection, sync_state_path=journal), journal)


@pytest.mark.parametrize("failure", ["before_delete", "after_delete", "clear_journal"])
def test_cleanup_failure_is_retried_without_losing_new_rows(sync_index, monkeypatch, failure):
    source, journal, index = sync_index
    roots = change_sources(source)
    delete = index.collection.delete

    def failing_delete(ids):
        if failure == "after_delete":
            delete(ids)
        raise RuntimeError("cleanup failed")

    def failing_clear():
        raise RuntimeError("cleanup failed")

    with monkeypatch.context() as patch:
        if failure == "clear_journal":
            patch.setattr(index._sync_journal, "clear", failing_clear)
        else:
            patch.setattr(index.collection, "delete", failing_delete)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            index.sync(roots)
    assert index.collection.count() >= 130
    assert journal.exists()
    assert_recovered(CodeIndex(index.collection, sync_state_path=journal), journal)


@pytest.mark.parametrize("use_cache", [False, True])
def test_persistent_failure_blocks_cache_and_partial_results(sync_index, monkeypatch, use_cache):
    source, journal, index = sync_index
    roots = change_sources(source)
    query_calls = index.collection.query_calls

    def unavailable(texts):
        raise RuntimeError("encoder unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(index.embedder, "embed", unavailable)
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            index.sync(roots)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="encoder unavailable"):
                index.search("old_symbol", use_cache=use_cache)
        assert index.collection.query_calls == query_calls
        assert index.cache_stats().hits == 0
    assert_recovered(index, journal)


def test_failed_initial_sync_recovers_new_file_without_any_chunk_metadata(sync_index, monkeypatch):
    source, journal, _ = sync_index
    index = CodeIndex(MemoryCollection(), sync_state_path=journal)

    def unavailable(texts):
        raise RuntimeError("encoder unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(index.embedder, "embed", unavailable)
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            index.sync([source.name])
    assert index.collection.count() == 0
    reopened = CodeIndex(index.collection, sync_state_path=journal)
    assert reopened.search("old_symbol")[0].symbol == "old_symbol"
    assert not journal.exists()


def test_rebuild_reencodes_without_deleting_unchanged_rows(sync_index, monkeypatch):
    _, journal, index = sync_index
    before = deepcopy(index.collection.data)
    embed = index.embedder.embed
    encoded = []

    def record(texts):
        encoded.extend(texts)
        return embed(texts)

    def unexpected(ids):
        pytest.fail("rebuild must not delete unchanged IDs")

    monkeypatch.setattr(index.embedder, "embed", record)
    monkeypatch.setattr(index.collection, "delete", unexpected)
    stats = index.sync(["controller.py"], rebuild=True)
    assert stats.added_or_updated == 1 and stats.deleted == 0 and stats.unchanged == 0
    assert len(encoded) == 1 and index.collection.data == before
    assert not journal.exists()


@pytest.mark.parametrize("content", ["not json", '{"version": 99}'])
def test_invalid_journal_blocks_queries_but_explicit_sync_can_repair(sync_index, content):
    _, journal, index = sync_index
    journal.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid pending RAG sync journal"):
        index.search("old_symbol")
    index.sync(["controller.py"])
    assert index.search("old_symbol")
    assert not journal.exists()


def test_journal_write_failure_preserves_existing_rows_and_previous_intent(sync_index, monkeypatch):
    from agent.rag import sync_state

    source, journal, index = sync_index
    before = deepcopy(index.collection.data)
    index._sync_journal.begin((source.name,), False)
    previous = journal.read_bytes()
    roots = change_sources(source)

    def unavailable(*args):
        raise OSError("journal disk unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(sync_state.os, "replace", unavailable)
        with pytest.raises(OSError, match="journal disk unavailable"):
            index.sync(roots)
    assert index.collection.data == before
    assert journal.read_bytes() == previous
    assert list(journal.parent.iterdir()) == [journal]
    # An explicit retry can replace the old intent after the disk recovers.
    index.sync(roots)
    assert_recovered(index, journal)


def test_real_chroma_reopen_recovers_scope_and_preserves_workspace_isolation(tmp_path, monkeypatch):
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    pytest.importorskip("chromadb")
    first, second = make_repo(tmp_path, "first"), make_repo(tmp_path, "second")
    index = open_code_index(workspace=first)
    other = open_code_index(workspace=second)
    index.sync(["include"])
    other.sync(["include"])
    before = index.collection.get(include=["documents", "metadatas"])
    (first.root / "src/new.cpp").write_text("// recovered_unique_marker\n", encoding="utf-8")
    upsert = index.collection.upsert

    def write_then_fail(**kwargs):
        upsert(**kwargs)
        raise RuntimeError("connection lost after write")

    with monkeypatch.context() as patch:
        patch.setattr(type(index.collection), "upsert", lambda self, **kw: write_then_fail(**kw))
        with pytest.raises(RuntimeError, match="connection lost"):
            index.sync(["src"], rebuild=True)
    assert set(before["ids"]) <= set(index.collection.get()["ids"])
    assert other.search("recovered_unique_marker", strategy="lexical") == []
    reopened = open_code_index(workspace=first)
    assert reopened.search("recovered_unique_marker", strategy="lexical")[0].path == "src/new.cpp"
    assert {row["path"] for row in reopened.collection.get()["metadatas"]} == {
        "src/new.cpp", "src/controller.cpp",
    }
    assert reopened._sync_journal.load() is None
    assert open_code_index(workspace=second).collection.count() == 1
