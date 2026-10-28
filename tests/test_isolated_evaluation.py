from __future__ import annotations

import json
import sys

import pytest

from agent.containers import ProcessResult, container_command, run_bounded
from agent.evaluation import evaluate_repository, receipt_matches, strict_json

RUNNER = "sha256:" + "1" * 64
ORACLE = "sha256:" + "2" * 64


def make_case(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "source.cpp").write_text("candidate")
    spec = tmp_path / "private.json"
    spec.write_text(json.dumps({"schema_version": 1, "task_id": "test", "runner_image": RUNNER,
                                "oracle_image": ORACLE, "runner_command": ["runner"],
                                "oracle_command": ["oracle"]}))
    return root, spec, tmp_path / "private-output"


def test_container_boundary_has_no_network_secrets_or_implicit_pull(tmp_path):
    args = container_command(RUNNER, ["python3", "-c", "print(1)"],
                             [(tmp_path, "/candidate", False)], name="test")
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--pull=never", "--user=65534:65534"):
        assert flag in args
    assert "readonly" in args[args.index("--mount") + 1]
    assert not any("docker.sock" in value for value in args)
    with pytest.raises(ValueError, match="pinned"):
        container_command("ubuntu:latest", ["true"], [], name="test")


def test_oracle_never_sees_candidate_and_candidate_never_sees_oracle(tmp_path, monkeypatch):
    root, spec, output = make_case(tmp_path)
    calls = []

    def fake_run(image, command, mounts, **kwargs):
        calls.append((image, command, mounts))
        if command[-1] == "prepare":
            assert image == ORACLE and mounts == []
            value = {"cases": [[1, 2]]}
        elif image == RUNNER:
            assert len(mounts) == 1 and mounts[0][1:] == ("/candidate", False)
            assert not (mounts[0][0] / "private.json").exists()
            value = {"observations": [3]}
        else:
            assert mounts == []
            assert json.loads(kwargs["stdin"])["observations"] == [3]
            value = {"passed": True, "checks": 1}
        return ProcessResult(0, json.dumps(value), "")

    monkeypatch.setattr("agent.evaluation.run_container", fake_run)
    report = evaluate_repository(root, spec, output)
    assert report["passed"] and len(calls) == 3
    assert receipt_matches(report, root, spec)
    (root / "source.cpp").write_text("new candidate")
    assert not receipt_matches(report, root, spec)


@pytest.mark.parametrize("payload", [
    '{"passed":true}', '{"observations":[NaN]}', '{"observations":[]}',
    '{"observations":[3],"observations":[4]}',
])
def test_forged_or_invalid_candidate_output_fails_closed(tmp_path, monkeypatch, payload):
    root, spec, output = make_case(tmp_path)

    def fake_run(image, command, mounts, **kwargs):
        return ProcessResult(0, '{"cases":[[1,2]]}' if image == ORACLE else payload, "")

    monkeypatch.setattr("agent.evaluation.run_container", fake_run)
    report = evaluate_repository(root, spec, output)
    assert not report["passed"] and report["stage"] == "candidate"
    assert payload not in report["error"]


def test_missing_docker_does_not_fallback_to_local(tmp_path, monkeypatch):
    root, spec, output = make_case(tmp_path)

    def missing(*args, **kwargs):
        raise OSError("Docker unavailable")

    monkeypatch.setattr("agent.evaluation.run_container", missing)
    report = evaluate_repository(root, spec, output)
    assert not report["passed"] and report["error"] and report["stage"] == "prepare"


def test_private_paths_cannot_be_inside_candidate(tmp_path):
    root, spec, output = make_case(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        evaluate_repository(root, spec, root / "results")
    inside = root / "spec.json"
    inside.write_text(spec.read_text())
    with pytest.raises(ValueError, match="outside"):
        evaluate_repository(root, inside, output)


def test_source_change_during_evaluation_invalidates_pass(tmp_path, monkeypatch):
    root, spec, output = make_case(tmp_path)

    def fake_run(image, command, mounts, **kwargs):
        if command[-1] == "prepare":
            value = {"cases": [1]}
        elif image == RUNNER:
            value = {"observations": [1]}
        else:
            (root / "source.cpp").write_text("changed during judge")
            value = {"passed": True, "checks": 1}
        return ProcessResult(0, json.dumps(value), "")

    monkeypatch.setattr("agent.evaluation.run_container", fake_run)
    report = evaluate_repository(root, spec, output)
    assert not report["passed"] and report["error"]


def test_process_timeout_and_output_budget():
    timed = run_bounded([sys.executable, "-c", "import time;time.sleep(20)"], timeout=0.1)
    assert timed.returncode == 124
    flooded = run_bounded([sys.executable, "-c", "print('x'*50000)"], output_limit=1000)
    assert flooded.returncode == 124 and len(flooded.stdout) <= 1000


def test_json_rejects_duplicate_verdict():
    with pytest.raises(ValueError, match="Duplicate"):
        strict_json('{"passed":false,"passed":true}')
