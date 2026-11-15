"""Freshness and work-count regressions; no timing thresholds or model downloads."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections import Counter

import pytest

from agent.rag.embedding import EmbeddingConfig, tokenize
from agent.rag.freshness import SourceDetector
from agent.rag.index import CodeIndex, open_code_index
from agent.rag.inverted import InvertedIndex
from agent.rag.storage import TrackedCollection
from agent.tools import write_patch
from agent.workspace import use_workspace
from tests.test_rag import MemoryCollection
from tests.test_workspace_rag import make_repo
from tests.test_token_chunking import BoundedEmbedder, token_chunker as token_chunker


@pytest.fixture
def tracked(tmp_path):
    workspace = make_repo(tmp_path)
    raw = MemoryCollection()
    raw.id = "test-index"
    index = CodeIndex(TrackedCollection(raw, tmp_path), workspace=workspace)
    index.sync()
    index.search("torque", strategy="lexical")
    return workspace, index


def scan_reference(rows, query, prefix):
    """Independent exhaustive implementation of the original BM25 + boosts."""
    rows = {key: row for key, row in rows.items()
            if not prefix or row[1]["path"].startswith(prefix.replace("\\", "/"))}
    if not rows:
        return []
    terms = set(tokenize(query))
    counts = {key: Counter(tokenize(row[0])) for key, row in rows.items()}
    average = sum(sum(count.values()) for count in counts.values()) / len(rows)
    scores = {}
    for key, (_, metadata) in rows.items():
        score = 0.0
        for term in terms:
            frequency = counts[key][term]
            if frequency:
                df = sum(term in count for count in counts.values())
                idf = math.log(1 + (len(rows) - df + .5) / (df + .5))
                length = sum(counts[key].values()) / average if average else 1
                score += idf * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * length))
        score += 2.5 * len(terms & set(tokenize(metadata["symbol"])))
        score += .6 * len(terms & set(tokenize(metadata["path"])))
        if score > 0:
            scores[key] = score
    return [(key, score / max(scores.values()))
            for key, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]


@pytest.mark.parametrize("prefix", [None, "src/", "include\\", "absent/"])
@pytest.mark.parametrize("query", ["torque", "saturation ctrl", "hpp", "missing", "!!!"])
def test_postings_match_scan_after_insert_update_delete(prefix, query):
    index = InvertedIndex()
    for key, doc, path, symbol in [
        ("a", "torque torque saturation", "src/a.cpp", "ctrl"),
        ("b", "torque", "include/b.hpp", "ctrl"),
        ("c", "", "include/empty.hpp", "saturation"),
        ("d", "unrelated text", "src/d.cpp", "other"),
    ]:
        index.upsert(key, doc, {"path": path, "symbol": symbol})
    for mutation in (lambda: None,
                     lambda: index.upsert("a", "new torque", {"path": "include/a.hpp", "symbol": "ctrl"}),
                     lambda: index.remove("b")):
        mutation()
        actual, expected = index.search(query, prefix), scan_reference(index.rows, query, prefix)
        assert [key for key, _ in actual] == [key for key, _ in expected]
        assert [score for _, score in actual] == pytest.approx([score for _, score in expected])


def test_warm_queries_do_not_reload_or_retokenize_and_only_read_hits(tracked):
    _, index = tracked
    before = index.performance_stats()
    for cached in (False, True, True):
        assert len(index.search("calibration", strategy="lexical", use_cache=cached)) == 1
    after = index.performance_stats()
    assert after["collection_full_reads"] == before["collection_full_reads"]
    assert after["lexical_documents_tokenized"] == before["lexical_documents_tokenized"]
    assert after["source_content_reads"] - before["source_content_reads"] == 3
    assert after["last_lexical_candidates"] == 1


@pytest.mark.parametrize("strategy", ["lexical", "vector", "hybrid"])
def test_external_same_size_restored_mtime_refreshes_all_strategies(tracked, strategy):
    workspace, index = tracked
    path = workspace.root / "src/controller.cpp"
    index.search("0.1234", strategy=strategy, path_prefix="src")
    before = index.performance_stats()
    stat = path.stat()
    path.write_text(path.read_text().replace("return torque;", "return 0.1234;"), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    result = index.search("0.1234", strategy=strategy, path_prefix="src")[0]
    assert "return 0.1234;" in result.content
    after = index.performance_stats()
    assert after["collection_full_reads"] == before["collection_full_reads"]
    assert after["lexical_documents_tokenized"] - before["lexical_documents_tokenized"] == 1


def test_atomic_replace_delete_and_scope_sync_remove_postings(tracked):
    workspace, index = tracked
    path = workspace.root / "include/controller.hpp"
    temporary = path.with_suffix(".new")
    temporary.write_text("// replacement_marker\n", encoding="utf-8")
    os.replace(temporary, path)
    assert index.search("replacement_marker", strategy="lexical")[0].path == "include/controller.hpp"
    path.unlink()
    assert not index.search("replacement_marker", strategy="lexical")
    index.sync(["src"])
    assert not index.search("calibration", strategy="lexical")
    assert {parent.path for parent in index._parents.values()} == {"src/controller.cpp"}


def test_notification_and_final_hash_guard_survive_constant_stamp(tracked, monkeypatch):
    workspace, index = tracked
    monkeypatch.setattr(index._detector, "fingerprint", lambda relative: ("constant",))
    index.search("torque", strategy="lexical")
    with use_workspace(workspace):
        assert write_patch.invoke({"path": "include/controller.hpp", "old_string": "#pragma once",
                                   "new_string": "// notification_marker"}).startswith("OK:")
    # A previously nonmatching file must become discoverable via the notification.
    assert index.search("notification_marker", strategy="lexical")
    path = workspace.root / "src/controller.cpp"
    path.write_text("// torque fresh_external_marker\n", encoding="utf-8")
    # No notification and no stamp change: the cached hit must still be revalidated.
    assert "fresh_external_marker" in index.search("torque", path_prefix="src", strategy="lexical")[0].content


def test_change_between_ranking_and_return_retries(tracked, monkeypatch):
    workspace, index = tracked
    search = index._search_uncached
    changed = False

    def racing(*args, **kwargs):
        nonlocal changed
        results = search(*args, **kwargs)
        if not changed:
            (workspace.root / "src/controller.cpp").write_text("// torque race_marker\n", encoding="utf-8")
            changed = True
        return results

    monkeypatch.setattr(index, "_search_uncached", racing)
    result = index.search("torque", path_prefix="src", strategy="lexical", use_cache=False)[0]
    assert "race_marker" in result.content


def test_continuous_changes_fail_closed(tracked, monkeypatch):
    workspace, index = tracked
    search = index._search_uncached
    serial = 0

    def racing(*args, **kwargs):
        nonlocal serial
        results = search(*args, **kwargs)
        serial += 1
        (workspace.root / "src/controller.cpp").write_text(f"// torque version {serial}\n", encoding="utf-8")
        return results

    monkeypatch.setattr(index, "_search_uncached", racing)
    with pytest.raises(RuntimeError, match="changed repeatedly"):
        index.search("torque", path_prefix="src", strategy="lexical", use_cache=False)


def test_missing_change_token_and_strict_mode_read_nonhits(tracked, monkeypatch):
    workspace, index = tracked
    monkeypatch.setattr(index._detector, "fingerprint", lambda relative: None)
    before = index.performance_stats()["source_content_reads"]
    index.search("calibration", strategy="lexical")
    assert index.performance_stats()["source_content_reads"] - before == 5
    monkeypatch.setattr(index._detector, "fingerprint", lambda relative: ("constant",))
    index._detector.strict = True
    index.search("new_marker", strategy="lexical")
    (workspace.root / "README.md").write_text("new_marker", encoding="utf-8")
    assert index.search("new_marker", strategy="lexical")[0].content.strip() == "new_marker"


def test_unreadable_cached_hit_raises(tracked, monkeypatch):
    _, index = tracked
    index.search("calibration", strategy="lexical")
    read = index._detector.read_source

    def fail(relative):
        if relative == "README.md":
            raise PermissionError("permission revoked")
        return read(relative)

    monkeypatch.setattr(index._detector, "read_source", fail)
    with pytest.raises(PermissionError):
        index.search("calibration", strategy="lexical")


def test_partial_backend_failure_invalidates_and_recovers(tracked, monkeypatch):
    workspace, index = tracked
    raw = index.collection.raw
    upsert = raw.upsert
    token = index.collection.revision_token()
    (workspace.root / "src/controller.cpp").write_text("// torque recovery_marker\n", encoding="utf-8")

    def fail(**kwargs):
        upsert(**kwargs)
        raise RuntimeError("partial write")

    monkeypatch.setattr(raw, "upsert", fail)
    with pytest.raises(RuntimeError, match="partial write"):
        index.search("torque", strategy="lexical")
    assert index.collection.revision_token() != token
    monkeypatch.setattr(raw, "upsert", upsert)
    assert "recovery_marker" in index.search("torque", path_prefix="src", strategy="lexical")[0].content


@pytest.mark.parametrize("strategy", ["lexical", "vector", "hybrid"])
def test_persistent_generation_not_count_invalidates_across_processes(tmp_path, strategy):
    pytest.importorskip("chromadb")
    workspace = make_repo(tmp_path)
    index = open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))
    index.sync(["src"])
    index.search("torque", strategy=strategy)
    peer = open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))
    peer.search("torque", strategy=strategy)
    before = index.performance_stats()["collection_full_reads"]
    # A separate process changes scope from one file to another: count stays one.
    code = """
import sys
from agent.workspace import Workspace
from agent.rag.index import open_code_index
from agent.rag.embedding import EmbeddingConfig
workspace = Workspace.create(sys.argv[1], sys.argv[2])
index = open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend='hash'))
assert index.sync(['include']).chunks == 1
"""
    subprocess.run([sys.executable, "-c", code, str(workspace.root), str(workspace.artifacts)],
                   check=True, timeout=60, capture_output=True, text=True)
    assert {result.path for result in index.search("torque", strategy=strategy)} == {"include/controller.hpp"}
    assert index.performance_stats()["collection_full_reads"] == before + 1
    assert peer.search("torque", strategy=strategy) == index.search("torque", strategy=strategy)
    reopened = open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))
    assert reopened.search("torque", strategy=strategy) == index.search("torque", strategy=strategy)


def test_stable_reader_rejects_changes_during_every_read(tmp_path):
    path = tmp_path / "source.cpp"
    serial = 0

    def read(relative):
        return path, "torque"

    detector = SourceDetector(tmp_path, lambda relative: path, read)

    def fingerprint(relative):
        nonlocal serial
        serial += 1
        return (serial,)

    detector.fingerprint = fingerprint
    with pytest.raises(RuntimeError, match="changed repeatedly"):
        detector.read_stable("source.cpp")


def test_changed_parent_refreshes_children_and_cached_context(tracked, token_chunker):
    workspace, original = tracked
    index = CodeIndex(original.collection, workspace=workspace, embedder=BoundedEmbedder(token_chunker))
    path = workspace.root / "src/controller.cpp"
    path.write_text("// " + "x" * 300 + " TAIL old_value\n", encoding="utf-8")
    index.sync(["src"])
    first = index.search("TAIL", strategy="lexical")[0]
    assert "old_value" in first.parent_context.content
    path.write_text(path.read_text().replace("old_value", "new_value"), encoding="utf-8")
    fresh = index.search("TAIL", strategy="lexical")[0]
    assert "new_value" in fresh.parent_context.content
    assert "old_value" not in fresh.parent_context.content
    assert all("old_value" not in document for document, _ in index._lexical.rows.values())


def test_persistent_freshness_configuration(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    workspace = make_repo(tmp_path)
    monkeypatch.setenv("RAG_FRESHNESS", "strict")
    assert open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))._detector.strict
    monkeypatch.setenv("RAG_FRESHNESS", "unknown")
    with pytest.raises(ValueError, match="RAG_FRESHNESS"):
        open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))


def test_deletion_before_first_query_removes_sync_parent_manifest(tmp_path):
    workspace = make_repo(tmp_path)
    index = CodeIndex(MemoryCollection(), workspace=workspace)
    index.sync(["src"])
    (workspace.root / "src/controller.cpp").unlink()
    assert index.search("torque", strategy="lexical") == []
    assert index.retrieval_manifest()["parent_count"] == 0
