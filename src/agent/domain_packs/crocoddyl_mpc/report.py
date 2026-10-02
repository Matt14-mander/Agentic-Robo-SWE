"""Independent correctness and profiling gates for Phase 6.4a."""

from __future__ import annotations

import math
import statistics
from typing import Any, TypeGuard

from agent.domain_packs.crocoddyl_mpc.spec import REPEATS, TIMING_KEYS, benchmark_contract


def _finite(value: Any) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(value)


def _timings(value: Any) -> TypeGuard[list[float]]:
    return (
        isinstance(value, list)
        and len(value) == REPEATS
        and all(_finite(item) and item > 0 for item in value)
    )


def evaluate_correctness(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported Crocoddyl profile schema")
    if raw.get("contract") != benchmark_contract():
        raise ValueError("Crocoddyl benchmark contract changed")
    evidence = raw.get("correctness")
    if not isinstance(evidence, dict):
        raise ValueError("Missing Crocoddyl correctness evidence")
    required = (
        "derivative_max_abs_error",
        "analytic_initial_cost",
        "analytic_final_cost",
        "analytic_final_state_norm",
        "trajectory_max_abs_error",
    )
    if not all(_finite(evidence.get(name)) for name in required):
        raise ValueError("Non-finite or missing Crocoddyl correctness evidence")
    failures: list[dict[str, Any]] = []

    def require(condition: bool, kind: str, **details: Any) -> None:
        if not condition:
            failures.append({"kind": kind, **details})

    derivative_error = float(evidence["derivative_max_abs_error"])
    initial_cost = float(evidence["analytic_initial_cost"])
    analytic_cost = float(evidence["analytic_final_cost"])
    analytic_norm = float(evidence["analytic_final_state_norm"])
    trajectory_error = float(evidence["trajectory_max_abs_error"])
    require(derivative_error <= 2e-4, "derivative_mismatch", actual=derivative_error)
    require(initial_cost > 0 and analytic_cost < initial_cost * 0.5,
            "insufficient_analytic_cost_reduction", initial=initial_cost, final=analytic_cost)
    require(analytic_norm < 0.25,
            "terminal_state_not_stabilized", analytic=analytic_norm)
    require(trajectory_error <= 1e-9, "rollout_mismatch", actual=trajectory_error)
    require(evidence.get("analytic_solved") is True, "analytic_solver_failed")
    return {
        "passed": not failures,
        "failures": failures,
        "first_failure": failures[0] if failures else None,
        "evidence": evidence,
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    return {
        "count": len(values),
        "p50_ns": statistics.median(values),
        "p95_ns": ordered[math.ceil(0.95 * len(values)) - 1],
        "mean_ns": mean,
        "stdev_ns": stdev,
        "cv": stdev / mean,
    }


def evaluate_profile(raw: dict[str, Any], *, correctness_passed: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": "deterministic LQR/FDDP solver chain; no Pinocchio RNEA CodeGen replacement",
        "codegen_end_to_end_speedup": None,
        "recommendation": "blocked_by_correctness",
    }
    if not correctness_passed:
        return result
    timing_spec = raw.get("timing_spec")
    contract = benchmark_contract()
    if timing_spec != {"warmup": contract["warmup"], "repeats": contract["repeats"]}:
        raise ValueError("Crocoddyl timing contract changed")
    timings = raw.get("timings_ns")
    if not isinstance(timings, dict):
        raise ValueError("Missing Crocoddyl timing samples")
    summaries: dict[str, dict[str, float | int]] = {}
    for name in TIMING_KEYS:
        values = timings.get(name)
        if not _timings(values):
            raise ValueError(f"Invalid timing samples: {name}")
        summaries[name] = _summary(values)
    calc_diff = float(summaries["analytic_calc_diff"]["p50_ns"])
    backward = float(summaries["analytic_backward_pass"]["p50_ns"])
    line_search = float(summaries["analytic_try_step"]["p50_ns"])
    direction_total = calc_diff + backward
    profile_noise = max(float(summary["cv"]) for summary in summaries.values())
    derivative_share = calc_diff / direction_total
    result.update({
        "stages": summaries,
        "analytic_direction_p50_ns": direction_total,
        "calc_diff_share_of_direction": derivative_share,
        "line_search_share_of_iteration_kernel": line_search / (direction_total + line_search),
        "numdiff_calc_diff_slowdown": (
            float(summaries["numdiff_calc_diff"]["p50_ns"]) / calc_diff
        ),
        "max_cv": profile_noise,
        "recommendation": (
            "inconclusive_noisy_profile"
            if profile_noise > 0.30
            else "candidate_for_codegen_action_model_trial"
            if derivative_share >= 0.50
            else "optimize_solver_or_rollout_before_codegen"
        ),
    })
    return result
