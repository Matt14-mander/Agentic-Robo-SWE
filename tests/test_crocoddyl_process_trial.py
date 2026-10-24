"""Independent-process gates without treating within-process repeats as independent."""
import copy
import json
from pathlib import Path

import pytest

from agent.domain_packs.crocoddyl_mpc import process_trial as module
from tests.test_crocoddyl_codegen_trial import trial_observations


def observations(repeats=10):
    raws = [trial_observations(repeats) for _ in range(3)]
    for i, raw in enumerate(raws):
        raw["process_id"] = 100+i
    return raws


def test_all_fresh_processes_must_show_benefit():
    result = module.evaluate_process_trial(observations(), processes=3)
    assert result["correctness"]["passed"]
    p = result["profile"]
    assert p["recommendation"] == "candidate_for_opt_in_codegen_pilot"
    assert p["beneficial_processes"] == 3
    assert p["speedup_median"] == pytest.approx(4/3)
    assert p["production_adoption"] is False


@pytest.mark.parametrize("fault", ["count", "missing_pid", "duplicate_pid", "bool_pid"])
def test_missing_or_reused_process_evidence_rejected(fault):
    raws = observations()
    if fault == "count":
        raws.pop()
    elif fault == "missing_pid":
        raws[1].pop("process_id")
    elif fault == "duplicate_pid":
        raws[1]["process_id"] = raws[0]["process_id"]
    else:
        raws[1]["process_id"] = True
    with pytest.raises(ValueError):
        module.evaluate_process_trial(raws, processes=3)


def test_one_bad_process_blocks_aggregate_correctness():
    raws = observations()
    raws[2]["runs"]["codegen"][9]["all_solved"] = False
    gate = module.evaluate_process_trial(raws, processes=3)
    assert not gate["correctness"]["passed"]
    assert gate["profile"]["recommendation"] == "blocked_by_correctness"


def test_one_process_without_benefit_cannot_hide_in_pooled_median():
    raws = observations()
    raws[2]["mpc_timings_ns"]["codegen"] = [21000.] * 10
    gate = module.evaluate_process_trial(raws, processes=3)
    assert gate["correctness"]["passed"]
    assert gate["profile"]["recommendation"] != "candidate_for_opt_in_codegen_pilot"
    assert gate["profile"]["beneficial_processes"] == 2


@pytest.mark.parametrize("within", [True, False])
def test_within_or_between_process_noise_is_inconclusive(within):
    raws = observations()
    raws[2]["mpc_timings_ns"]["codegen"] = ([100000.] + [15000.] * 9 if within else [7000.] * 10)
    gate = module.evaluate_process_trial(raws, processes=3)
    assert gate["profile"]["recommendation"] == "inconclusive_noisy_profile"


@pytest.mark.parametrize("count", [1, 2, 21, True, 3.5])
def test_process_bounds(count):
    with pytest.raises(ValueError):
        module.check_process_count(count)


@pytest.mark.parametrize("fault", [None, "child", "identity", "raw"])
@pytest.mark.parametrize("repeats", [10, 50])
def test_orchestration_checkpoints_identity_and_rejudging(monkeypatch, tmp_path, fault, repeats):
    monkeypatch.setattr(module.validators, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module.validators, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    raws = observations(repeats)
    if fault == "raw":
        raws[2]["runs"]["codegen"][0]["all_solved"] = False
    calls = []
    def child(**options):
        assert options["codegen"] is True
        assert options["repeats"] == repeats
        index = len(calls)
        calls.append(index)
        directory = tmp_path / f"native-{index}"
        directory.mkdir()
        evidence = {"dependencies": {"fingerprint": "deps"},
                    "environment": {"toolchain": {"fingerprint": "tools"}},
                    "build": {"fingerprint": "build"},
                    "execution": {"executable_sha256": "changed" if index == 1 and fault == "identity" else "exe"}}
        (directory / "validation.json").write_text(json.dumps(evidence))
        (directory / "raw.json").write_text(json.dumps(copy.deepcopy(raws[index])))
        # Child summaries are deliberately untrusted: the raw fault must still be caught.
        return {"passed": not (fault == "child" and index == 1), "stage": "runtime" if fault == "child" else "complete",
                "artifacts": {"directory": str(directory), "validation": str(directory / "validation.json")}}
    monkeypatch.setattr(module.validators, "validate_crocoddyl_mpc", child)
    messages = []
    result = module.validate_process_trial(processes=3, repeats=repeats, progress=messages.append)
    if fault:
        assert not result["passed"]
        assert result["stage"] == {"child": "process_failure", "identity": "identity", "raw": "correctness"}[fault]
    else:
        assert result["passed"]
        assert result["profile"]["independent_processes"] == 3
        assert result["profile"]["repetitions_per_process"] == repeats
    saved = json.loads(Path(result["artifacts"]["validation"]).read_text())
    checkpoint = json.loads((Path(result["artifacts"]["directory"]) / "checkpoint.json").read_text())
    assert len(saved["processes"]) == len(calls)
    assert len(checkpoint["processes"]) == len(calls)
    assert "checkpoint:" in messages[0]


def test_aggregate_identifies_process_and_exact_rejected_gate():
    raws = observations(50)
    raws[1]["mpc_timings_ns"]["codegen"] = [15000.] * 47 + [21000.] * 3
    profile = module.evaluate_process_trial(raws, processes=3, repeats=50)["profile"]
    assert profile["rejection_reasons"] == [{"process_id": 101, "gates": ["p95_non_regression"]}]
    assert profile["beneficial_processes"] == 2
    assert profile["recommendation"] == "keep_native_inconsistent_process_benefit"
    assert profile["between_process_cv_gate"]["passed"]


def test_mixed_repeat_contracts_cannot_be_pooled():
    raws = observations(50)
    raws[1] = trial_observations()
    raws[1]["process_id"] = 101
    with pytest.raises(ValueError, match="contract"):
        module.evaluate_process_trial(raws, processes=3, repeats=50)
