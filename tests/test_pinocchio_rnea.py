"""Independent report gates plus explicitly optional native Pinocchio integration."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from agent.benchmark import domain_case_metadata, load_cases, prepare_workspace
from agent.domain import get_domain_registry
from agent.domain.toolchain import BuildResult
from agent.domain_packs.pinocchio_rnea.dependencies import inspect_dependencies
from agent.domain_packs.pinocchio_rnea.report import (
    aggregate_control_processes, evaluate_control_loop, evaluate_correctness,
    evaluate_performance,
)
from agent.domain_packs.pinocchio_rnea.spec import (
    BATCH_SIZE, CONTROL_DEADLINE_NS, CONTROL_DT, CONTROL_REPEATS, CONTROL_STEPS,
    CONTROL_WARMUP, MODEL_PATH, PACK_ROOT, REPEATS, WARMUP, model_contract, samples,
)
from agent.domain_packs.pinocchio_rnea.validators import validate_rnea
from scripts.bootstrap_pinocchio import bootstrap


def observations():
    # Synthetic observations test the judge, NOT actual Pinocchio physics or speed.
    records = []
    for point in samples():
        records.append({"input": point,
                        **{name: [1.0, 2.0] for name in
                           ("pinocchio_output", "source_output", "analytic_output",
                            "cppad_output", "codegen_output")},
                        **{name: [float(i + 1) for i in range(12)] for name in
                           ("analytic_jacobian", "finite_difference_jacobian",
                            "cppad_jacobian", "codegen_jacobian")}})
    return {"schema_version": 1, "nq": 2, "nv": 2,
            "urdf_sha256": model_contract()["urdf_sha256"], "samples": records,
            "codegen_compile_seconds": 1.0,
            "library_load_seconds": 0.01,
            "timing_spec": {"warmup": WARMUP, "repeats": REPEATS, "batch_size": BATCH_SIZE},
            "timings_ns": {"pinocchio": [1000.0] * REPEATS, "cppad": [2000.0] * REPEATS,
                           "codegen": [500.0] * REPEATS, "finite_difference": [3000.0] * REPEATS},
            "control_loop": {
                "spec": {"warmup": CONTROL_WARMUP, "repeats": CONTROL_REPEATS,
                         "steps": CONTROL_STEPS, "dt": CONTROL_DT,
                         "deadline_ns": CONTROL_DEADLINE_NS},
                "timings_ns": {"pinocchio": [10000.0] * CONTROL_REPEATS,
                               "codegen": [8000.0] * CONTROL_REPEATS},
                "pinocchio": {"final_state": [0.1, 0.2, 0.3, 0.4],
                              "checkpoints": [float(i) for i in range(48)],
                              "checksum": 1234.5},
                "codegen": {"final_state": [0.1, 0.2, 0.3, 0.4],
                            "checkpoints": [float(i) for i in range(48)],
                            "checksum": 1234.5}},
            "library_reused": False}


def test_pinocchio_pack_manifest_and_fixed_robot_contract():
    pack = get_domain_registry().get("pinocchio_rnea")
    assert pack.version == "0.2.0"
    assert set(pack.validators) == {"pinocchio_rnea.correctness", "pinocchio_rnea.codegen_benefit"}
    assert len(pack.tools) == 3
    assert samples() == samples() and len(samples()) == 24
    assert all(len(row) == 6 for row in samples())
    robot = ET.parse(MODEL_PATH).getroot()
    assert robot.attrib["name"] == model_contract()["name"]
    assert [joint.attrib["type"] for joint in robot.findall("joint")] == ["revolute"] * 2
    cases = load_cases("benchmarks/pinocchio_cases.json")
    assert len(cases) == 2 and all(case.domain_pack == pack.id for case in cases)
    assert inspect_dependencies() == inspect_dependencies()


def test_pinocchio_install_is_not_invalidated_by_ad_build_tool_cache_drift(
    monkeypatch, tmp_path
):
    dependencies = importlib.import_module("agent.domain_packs.pinocchio_rnea.dependencies")
    prefix = tmp_path / "install"
    for header in ("multibody/model.hpp", "parsers/urdf.hpp", "codegen/cppadcg.hpp"):
        path = prefix / "include/pinocchio" / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "installed.json").write_text(json.dumps({
        "commit": dependencies.PINOCCHIO_COMMIT,
        "version": dependencies.PINOCCHIO_VERSION,
        "ad_fingerprint": "fingerprint-before-ninja-was-discovered",
    }), encoding="utf-8")
    monkeypatch.setattr(dependencies, "ROOT", tmp_path)
    monkeypatch.setattr(dependencies, "PREFIX", prefix)
    monkeypatch.setattr(dependencies, "inspect_ad", lambda: {
        "available": True, "missing": [], "fingerprint": "new-build-tool-cache-key",
        "platform": "Linux",
    })

    result = dependencies.inspect_dependencies()
    assert result["available"]
    assert result["missing"] == []


def test_official_gate_recomputes_verdict_instead_of_trusting_pass_flag():
    raw = observations()
    assert evaluate_correctness(raw)["passed"]
    raw["passed"] = True
    raw["samples"][8]["codegen_jacobian"][5] = -99.0
    result = evaluate_correctness(raw)
    assert not result["passed"]
    assert result["first_failure"]["column"] == 5
    assert result["first_failure"]["sample_index"] == 8


@pytest.mark.parametrize("field", ["nq", "nv", "urdf_sha256", "schema_version"])
def test_contract_mismatch_fails_closed(field):
    raw = observations()
    raw[field] = "wrong"
    with pytest.raises(ValueError):
        evaluate_correctness(raw)


@pytest.mark.parametrize("fault", ["missing", "order", "input"])
def test_incomplete_or_changed_samples_cannot_pass(fault):
    raw = observations()
    if fault == "missing":
        raw["samples"].pop()
    elif fault == "order":
        raw["samples"].reverse()
    else:
        raw["samples"][0]["input"][0] = 0.1
    with pytest.raises(ValueError):
        evaluate_correctness(raw)


@pytest.mark.parametrize("value", [None, [float("nan")] * 12, [float("inf")] * 12, [True] * 12, []])
def test_nonfinite_and_malformed_derivatives_fail(value):
    raw = observations()
    raw["samples"][3]["analytic_jacobian"] = value
    result = evaluate_correctness(raw)
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


def test_fast_but_incorrect_never_receives_adoption_recommendation():
    result = evaluate_performance(observations(), correctness_passed=False)
    assert result["recommendation"] == "blocked_by_correctness"
    assert not result["measured_codegen_benefit"]


def test_control_loop_checks_full_trajectory_and_limited_scope():
    raw = observations()
    result = evaluate_control_loop(raw)
    assert result["passed"]
    assert result["recommendation"] == "candidate_for_repeated_process_validation"
    assert "MPC/Crocoddyl" in result["not_measured"]

    raw["control_loop"]["codegen"]["checkpoints"][17] += 0.01
    failed = evaluate_control_loop(raw)
    assert not failed["passed"]
    assert failed["failures"][0]["kind"] == "trajectory_mismatch"


@pytest.mark.parametrize("fault", ["spec", "count", "zero", "nan", "state", "checksum"])
def test_control_loop_malformed_evidence_fails_closed(fault):
    raw = observations()
    if fault == "spec":
        raw["control_loop"]["spec"]["steps"] = 1
    elif fault == "count":
        raw["control_loop"]["timings_ns"]["codegen"].pop()
    elif fault == "zero":
        raw["control_loop"]["timings_ns"]["pinocchio"][0] = 0
    elif fault == "nan":
        raw["control_loop"]["timings_ns"]["codegen"][0] = float("nan")
    elif fault == "state":
        raw["control_loop"]["codegen"]["final_state"] = []
    else:
        raw["control_loop"]["codegen"]["checksum"] += 10
    if fault in {"spec", "count", "zero", "nan"}:
        with pytest.raises(ValueError):
            evaluate_control_loop(raw)
    else:
        assert not evaluate_control_loop(raw)["passed"]


def test_three_process_aggregation_computes_break_even_and_requires_consistency():
    reports = [observations() for _ in range(3)]
    for report in reports:
        report["library_reused"] = True
    result = aggregate_control_processes(reports, compile_seconds=2.0)
    assert result["passed"]
    assert result["recommendation"] == "validated_end_to_end_candidate"
    assert result["process_count"] == 3
    assert result["samples_per_backend"] == 3 * CONTROL_REPEATS
    assert result["break_even_calls"] == 1_000_000
    assert result["break_even_seconds_at_1khz"] == 1000

    reports[1]["control_loop"]["codegen"]["final_state"][0] += 1
    blocked = aggregate_control_processes(reports, compile_seconds=2.0)
    assert not blocked["passed"]
    assert blocked["recommendation"] == "blocked_by_correctness"


def test_control_aggregation_rejects_too_few_processes_or_cross_process_noise():
    with pytest.raises(ValueError, match="three"):
        aggregate_control_processes([observations(), observations()], compile_seconds=1)
    reports = [observations() for _ in range(3)]
    for report in reports:
        report["library_reused"] = True
    reports[2]["control_loop"]["timings_ns"]["codegen"] = [40000.0] * CONTROL_REPEATS
    result = aggregate_control_processes(reports, compile_seconds=1)
    assert result["passed"]
    assert result["recommendation"] == "inconclusive_cross_process_variance"


def test_control_orchestrator_reuses_hashed_library_across_fresh_processes(
    monkeypatch, tmp_path
):
    module = importlib.import_module("agent.domain_packs.pinocchio_rnea.control_benchmark")
    build = tmp_path / "build"
    artifact = tmp_path / "run"
    build.mkdir()
    artifact.mkdir()
    (build / "rnea_library.so").write_bytes(b"fixed-generated-library")
    (artifact / "samples.txt").write_text("inputs", encoding="ascii")
    (artifact / "raw.json").write_text(json.dumps(observations()), encoding="utf-8")
    (artifact / "validation.json").write_text(json.dumps({
        "build": {"build_dir": str(build)},
    }), encoding="utf-8")
    monkeypatch.setattr(module, "resolve_within_root", lambda value: Path(value))
    monkeypatch.setattr(module, "relpath_for_display", str)
    monkeypatch.setattr(module, "validate_rnea", lambda _: {
        "passed": True, "artifacts": {"directory": str(artifact)},
        "control_loop": {"recommendation": "candidate_for_repeated_process_validation"},
    })
    calls = []

    def run(*_, args, **__):
        calls.append(args)
        raw = observations()
        raw["library_reused"] = True
        Path(args[1]).write_text(json.dumps(raw), encoding="utf-8")
        return {"passed": True, "stderr": ""}

    monkeypatch.setattr(module, "run_built_executable", run)
    result = module.run_control_loop_benchmark(3)
    assert result["passed"]
    assert result["recommendation"] == "validated_end_to_end_candidate"
    assert all(call[2] == "--reuse-library" for call in calls)
    assert len(calls) == 3
    assert (artifact / "control-loop-aggregate.json").is_file()


@pytest.mark.parametrize("latency,recommendation", [
    (500, "candidate_for_end_to_end_trial"), (980, "keep_pinocchio_analytic"),
    (1500, "keep_pinocchio_analytic"),
])
def test_performance_uses_analytic_baseline_and_does_not_infer_end_to_end(latency, recommendation):
    raw = observations()
    raw["timings_ns"]["codegen"] = [latency] * REPEATS
    result = evaluate_performance(raw, correctness_passed=True)
    assert result["recommendation"] == recommendation
    assert result["end_to_end_speedup"] is None
    assert result["backends"]["codegen"]["p95_ns"] == latency


@pytest.mark.parametrize("fault", ["warmup", "count", "nan", "zero", "noise", "tail"])
def test_bad_or_noisy_performance_evidence_is_not_accepted(fault):
    raw = observations()
    if fault == "warmup":
        raw["timing_spec"]["warmup"] = 0
    elif fault == "count":
        raw["timings_ns"]["codegen"].pop()
    elif fault == "nan":
        raw["timings_ns"]["codegen"][0] = float("nan")
    elif fault == "zero":
        raw["timings_ns"]["codegen"][0] = 0
    elif fault == "noise":
        raw["timings_ns"]["codegen"][0] = 100000
    else:
        # Median improves but the tail gets worse: do not recommend switching.
        raw["timings_ns"]["codegen"] = [900] * 45 + [1100] * 5
    result = evaluate_performance(raw, correctness_passed=True)
    assert not result["measured_codegen_benefit"]


def test_windows_bootstrap_and_missing_capability_are_explicit(monkeypatch, tmp_path):
    bootstrap_module = importlib.import_module("scripts.bootstrap_pinocchio")
    monkeypatch.setattr(bootstrap_module.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        bootstrap()
    module = importlib.import_module("agent.domain_packs.pinocchio_rnea.validators")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    monkeypatch.setattr(module, "inspect_dependencies", lambda: {
        "available": False, "missing": ["pinocchio_rnea_runtime"],
    })
    result = validate_rnea(str(PACK_ROOT / "harness/reference.hpp"))
    assert not result["passed"] and result["stage"] == "capability"
    assert len(list(tmp_path.glob("*/validation.json"))) == 1
    case = load_cases("benchmarks/pinocchio_cases.json")[0]
    assert domain_case_metadata(case)[1]["pinocchio_rnea_runtime"] is False


@pytest.mark.parametrize("fault,strict,stage", [
    (None, False, "complete"), ("wrong", False, "correctness"),
    ("slow", False, "complete"), ("slow", True, "performance"),
    ("missing", False, "report"), ("crash", False, "runtime"),
])
def test_validator_persists_evidence_and_distinguishes_adoption_policy(monkeypatch, tmp_path, fault, strict, stage):
    module = importlib.import_module("agent.domain_packs.pinocchio_rnea.validators")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    monkeypatch.setattr(module, "inspect_dependencies", lambda: {
        "available": True, "missing": [], "fingerprint": "test",
    })
    monkeypatch.setattr(module, "configure_and_build", lambda *a, **kw: BuildResult(
        passed=True, source_dir="source", build_dir="build", target="target", fingerprint="test",
        cache_hit=False, configure_seconds=0, build_seconds=0, stdout="", stderr="",
    ))

    def run(*a, args, **kw):
        raw = observations()
        if fault == "wrong":
            raw["samples"][8]["source_output"][0] = 100
        if fault == "slow":
            raw["timings_ns"]["codegen"] = [2000] * REPEATS
        if fault != "missing":
            Path(args[1]).write_text(json.dumps(raw), encoding="utf-8")
        return {"passed": fault != "crash", "stderr": "crashed" if fault == "crash" else ""}

    monkeypatch.setattr(module, "run_built_executable", run)
    result = validate_rnea(str(PACK_ROOT / "harness/reference.hpp"), require_benefit=strict)
    assert result["stage"] == stage
    assert result["passed"] == (stage == "complete")
    saved = next(tmp_path.glob("*/validation.json"))
    assert json.loads(saved.read_text(encoding="utf-8"))["source_sha256"]
    if fault == "wrong":
        assert saved.with_name("failing-input.json").is_file()
        assert result["performance"]["recommendation"] == "blocked_by_correctness"


@pytest.mark.cpp
@pytest.mark.pinocchio
def test_real_pinocchio_source_repairs_and_codegen_equivalence():
    if not inspect_dependencies()["available"]:
        pytest.skip("pinned Pinocchio + CppADCodeGen Linux runtime unavailable")
    result = validate_rnea(str(PACK_ROOT / "harness/reference.hpp"))
    assert result["passed"], result
    assert result["metrics"]["sample_count"] == 24
    assert "backends" in result["performance"], result
    assert result["control_loop"]["passed"], result
    for case in load_cases("benchmarks/pinocchio_cases.json"):
        workspace = prepare_workspace(case)
        try:
            broken = validate_rnea(case.workspace)
            assert broken["stage"] == "correctness" and not broken["passed"], broken
            workspace.write_text((PACK_ROOT / "harness/reference.hpp").read_text(encoding="utf-8"),
                                 encoding="utf-8")
            fixed = validate_rnea(case.workspace)
            assert fixed["passed"], fixed
        finally:
            workspace.unlink(missing_ok=True)
