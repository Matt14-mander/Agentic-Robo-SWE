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
from agent.domain_packs.pinocchio_rnea.report import evaluate_correctness, evaluate_performance
from agent.domain_packs.pinocchio_rnea.spec import (
    BATCH_SIZE, MODEL_PATH, PACK_ROOT, REPEATS, WARMUP, model_contract, samples,
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
            "timing_spec": {"warmup": WARMUP, "repeats": REPEATS, "batch_size": BATCH_SIZE},
            "timings_ns": {"pinocchio": [1000.0] * REPEATS, "cppad": [2000.0] * REPEATS,
                           "codegen": [500.0] * REPEATS, "finite_difference": [3000.0] * REPEATS}}


def test_pinocchio_pack_manifest_and_fixed_robot_contract():
    pack = get_domain_registry().get("pinocchio_rnea")
    assert pack.version == "0.1.0"
    assert set(pack.validators) == {"pinocchio_rnea.correctness", "pinocchio_rnea.codegen_benefit"}
    assert len(pack.tools) == 2
    assert samples() == samples() and len(samples()) == 24
    assert all(len(row) == 6 for row in samples())
    robot = ET.parse(MODEL_PATH).getroot()
    assert robot.attrib["name"] == model_contract()["name"]
    assert [joint.attrib["type"] for joint in robot.findall("joint")] == ["revolute"] * 2
    cases = load_cases("benchmarks/pinocchio_cases.json")
    assert len(cases) == 2 and all(case.domain_pack == pack.id for case in cases)
    assert inspect_dependencies() == inspect_dependencies()


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
