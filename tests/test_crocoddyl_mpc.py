"""Phase 6.4a report gates and optional native Crocoddyl integration."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from agent.domain import get_domain_registry
from agent.domain.toolchain import BuildResult
from agent.domain_packs.crocoddyl_mpc.dependencies import inspect_dependencies
from agent.domain_packs.crocoddyl_mpc.report import evaluate_correctness, evaluate_profile
from agent.domain_packs.crocoddyl_mpc.spec import REPEATS, TIMING_KEYS, benchmark_contract
from agent.domain_packs.crocoddyl_mpc.validators import validate_crocoddyl_mpc
from scripts.bootstrap_crocoddyl import bootstrap


def observations() -> dict:
    return {
        "schema_version": 1,
        "contract": benchmark_contract(),
        "correctness": {
            "derivative_max_abs_error": 1e-7,
            "analytic_initial_cost": 100.0,
            "analytic_final_cost": 10.0,
            "analytic_final_state_norm": 0.05,
            "trajectory_max_abs_error": 1e-12,
            "analytic_solved": True,
            "analytic_iterations": 2,
        },
        "timing_spec": {"warmup": 5, "repeats": REPEATS},
        "timings_ns": {
            "analytic_calc": [1_000.0] * REPEATS,
            "analytic_calc_diff": [6_000.0] * REPEATS,
            "analytic_backward_pass": [4_000.0] * REPEATS,
            "analytic_try_step": [2_000.0] * REPEATS,
            "analytic_solve": [20_000.0] * REPEATS,
            "analytic_rollout": [900.0] * REPEATS,
            "numdiff_calc_diff": [60_000.0] * REPEATS,
        },
    }


def test_pack_manifest_and_fixed_contract():
    pack = get_domain_registry().get("crocoddyl_mpc")
    assert pack.version == "0.4.0"
    assert set(pack.validators) == {"crocoddyl_mpc.solver_chain", "crocoddyl_mpc.robot_mpc",
                                    "crocoddyl_mpc.codegen_trial", "crocoddyl_mpc.codegen_processes"}
    assert len(pack.tools) == 5
    contract = benchmark_contract()
    assert contract["horizon"] == 40
    assert contract["codegen_replacement"] is False
    assert tuple(contract["timing_keys"]) == TIMING_KEYS
    assert inspect_dependencies() == inspect_dependencies()


def test_correctness_gate_recomputes_verdict():
    raw = observations()
    raw["passed"] = True
    assert evaluate_correctness(raw)["passed"]
    raw["correctness"]["derivative_max_abs_error"] = 0.5
    result = evaluate_correctness(raw)
    assert not result["passed"]
    assert result["first_failure"]["kind"] == "derivative_mismatch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("analytic_initial_cost", float("nan")),
        ("analytic_final_cost", float("inf")),
        ("trajectory_max_abs_error", None),
        ("analytic_final_state_norm", True),
    ],
)
def test_nonfinite_or_malformed_correctness_fails_closed(field, value):
    raw = observations()
    raw["correctness"][field] = value
    with pytest.raises(ValueError):
        evaluate_correctness(raw)


def test_profile_locates_derivative_bottleneck_without_claiming_codegen_speedup():
    raw = observations()
    result = evaluate_profile(raw, correctness_passed=True)
    assert result["recommendation"] == "candidate_for_codegen_action_model_trial"
    assert result["calc_diff_share_of_direction"] == 0.6
    assert result["numdiff_calc_diff_slowdown"] == 10
    assert result["codegen_end_to_end_speedup"] is None


def test_solver_dominated_and_incorrect_profiles_do_not_recommend_codegen():
    raw = observations()
    raw["timings_ns"]["analytic_calc_diff"] = [1_000.0] * REPEATS
    raw["timings_ns"]["analytic_backward_pass"] = [9_000.0] * REPEATS
    assert evaluate_profile(raw, correctness_passed=True)["recommendation"] == (
        "optimize_solver_or_rollout_before_codegen"
    )
    assert evaluate_profile(raw, correctness_passed=False)["recommendation"] == (
        "blocked_by_correctness"
    )


@pytest.mark.parametrize("fault", ["contract", "count", "zero", "nan", "noise"])
def test_changed_or_invalid_timing_evidence_is_rejected(fault):
    raw = observations()
    if fault == "contract":
        raw["timing_spec"]["warmup"] = 0
    elif fault == "count":
        raw["timings_ns"]["analytic_calc"].pop()
    elif fault == "zero":
        raw["timings_ns"]["analytic_solve"][0] = 0
    elif fault == "nan":
        raw["timings_ns"]["numdiff_calc_diff"][0] = float("nan")
    else:
        raw["timings_ns"]["analytic_calc"][0] = 1_000_000
        result = evaluate_profile(raw, correctness_passed=True)
        assert result["recommendation"] == "inconclusive_noisy_profile"
        return
    with pytest.raises(ValueError):
        evaluate_profile(raw, correctness_passed=True)


def test_windows_bootstrap_and_missing_capability_are_explicit(monkeypatch, tmp_path):
    bootstrap_module = importlib.import_module("scripts.bootstrap_crocoddyl")
    monkeypatch.setattr(bootstrap_module.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        bootstrap()
    module = importlib.import_module("agent.domain_packs.crocoddyl_mpc.validators")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    monkeypatch.setattr(module, "inspect_dependencies", lambda: {
        "available": False,
        "missing": ["crocoddyl_mpc_runtime"],
    })
    result = validate_crocoddyl_mpc()
    assert not result["passed"] and result["stage"] == "capability"
    assert len(list(tmp_path.glob("*/validation.json"))) == 1


@pytest.mark.parametrize("fault,stage", [
    (None, "complete"),
    ("wrong", "correctness"),
    ("missing", "report"),
    ("crash", "runtime"),
])
def test_validator_persists_evidence(monkeypatch, tmp_path, fault, stage):
    module = importlib.import_module("agent.domain_packs.crocoddyl_mpc.validators")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    monkeypatch.setattr(module, "inspect_dependencies", lambda: {
        "available": True,
        "missing": [],
        "fingerprint": "test",
    })
    build_options = {}

    def build(*args, **kwargs):
        build_options.update(kwargs)
        return BuildResult(
            passed=True,
            source_dir="source",
            build_dir="build",
            target="target",
            fingerprint="test",
            cache_hit=False,
            configure_seconds=0,
            build_seconds=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(module, "configure_and_build", build)

    def run(*args, **kwargs):
        raw = observations()
        if fault == "wrong":
            raw["correctness"]["trajectory_max_abs_error"] = 1
        if fault != "missing":
            Path(kwargs["args"][0]).write_text(json.dumps(raw), encoding="utf-8")
        return {"passed": fault != "crash", "stderr": "crashed" if fault == "crash" else ""}

    monkeypatch.setattr(module, "run_built_executable", run)
    result = validate_crocoddyl_mpc()
    assert build_options["timeout"] == 300
    assert build_options["build_timeout"] == 900
    assert result["stage"] == stage
    assert result["passed"] == (stage == "complete")
    assert next(tmp_path.glob("*/validation.json")).is_file()


@pytest.mark.cpp
@pytest.mark.crocoddyl
def test_real_crocoddyl_lqr_fddp_profile():
    if not inspect_dependencies()["available"]:
        pytest.skip("pinned Crocoddyl Linux runtime unavailable")
    result = validate_crocoddyl_mpc()
    assert result["passed"], result
    assert result["correctness"]["passed"], result
    assert result["profile"]["codegen_end_to_end_speedup"] is None
