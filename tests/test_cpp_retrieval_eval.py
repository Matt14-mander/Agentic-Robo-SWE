"""Dataset isolation and executable offline C++ retrieval protocol."""

import json
import shutil

import pytest

from scripts import run_cpp_retrieval_eval as cli


def test_frozen_dataset_has_disjoint_targets_and_labels_outside_corpus():
    dataset = cli.validate_dataset()
    assert dataset["case_counts"] == {"dev": 16, "heldout": 16}
    assert len(dataset["corpus_files"]) == 20
    assert dataset["target_files_disjoint"] and dataset["public_heldout"]


@pytest.mark.parametrize("problem", ["shared_target", "duplicate_query", "missing_target", "label_in_corpus"])
def test_dataset_rejects_leakage_and_missing_evidence(tmp_path, problem):
    root = tmp_path / "dataset"
    shutil.copytree(cli.DATASET, root)
    heldout = json.loads((root / "heldout.json").read_text())
    dev = json.loads((root / "dev.json").read_text())
    if problem == "shared_target":
        heldout["cases"][0]["targets"] = dev["cases"][0]["targets"]
    elif problem == "duplicate_query":
        heldout["cases"][0]["query"] = dev["cases"][0]["query"]
    elif problem == "missing_target":
        heldout["cases"][0]["targets"] = [{"path": "missing.hpp"}]
    else:
        (root / "corpus/answers.json").write_text("{}")
    (root / "heldout.json").write_text(json.dumps(heldout))
    with pytest.raises(ValueError):
        cli.validate_dataset(root)


def test_hash_only_runner_is_offline_and_records_dataset(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from agent.rag import embedding
    from agent.tools import _paths
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    monkeypatch.setattr(_paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(embedding, "SemanticCodeEmbedder", lambda config: pytest.fail("offline smoke loaded model"))
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["run_cpp_retrieval_eval", "--hash-only", "--output", str(output)])
    assert cli.main() == 0
    report = json.loads((output / "comparison.json").read_text())
    assert report["status"] == "complete" and report["case_count"] == 16
    assert set(report["arms"]) == {"lexical", "hash_vector", "hash_hybrid"}
    assert report["retrieval_cache_enabled"] is False
    assert report["dataset"]["purpose"] == "offline-smoke"
    assert report["dataset"]["split"] == "dev"


def test_changed_dataset_cannot_leave_a_complete_report(tmp_path, monkeypatch):
    from agent.tools import _paths
    monkeypatch.setattr(_paths, "PROJECT_ROOT", tmp_path)
    fingerprints = iter([{"corpus_sha256": "before"}, {"corpus_sha256": "after"}])
    monkeypatch.setattr(cli, "validate_dataset", lambda: next(fingerprints))
    output = tmp_path / "run"

    def compared(*args, **kwargs):
        output.mkdir(exist_ok=True)
        (output / "comparison.json").write_text('{"status": "complete"}')

    monkeypatch.setattr(cli, "compare", compared)
    monkeypatch.setattr("sys.argv", ["run_cpp_retrieval_eval", "--hash-only", "--output", str(output)])
    assert cli.main() == 2
    assert json.loads((output / "comparison.json").read_text())["status"] == "incomplete"
