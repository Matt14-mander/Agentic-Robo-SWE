"""Independently check two-link plant transitions and real MPC evidence."""

from __future__ import annotations

import math
import statistics
from typing import Any, TypeGuard

from agent.domain_packs.crocoddyl_mpc.report import _finite, _summary
from agent.domain_packs.crocoddyl_mpc.spec import (
    MPC_REPEATS, MPC_STEPS, MPC_TIMING_KEYS, robot_contract,
)


def _vector(value: Any, size: int) -> TypeGuard[list[float]]:
    return isinstance(value, list) and len(value) == size and all(_finite(x) for x in value)


def plant_step(state: list[float], control: list[float]) -> list[float]:
    """Independent planar rigid-body equations for the fixed project-owned URDF."""
    q1, q2, v1, v2 = state
    coupling = 0.8 * 1.0 * 0.35
    m22 = 0.033 + 0.8 * 0.35**2
    m12 = m22 + coupling * math.cos(q2)
    m11 = 0.101 + 1.2 * 0.5**2 + m22 + 0.8 + 2 * coupling * math.cos(q2)
    c1 = -coupling * math.sin(q2) * (2 * v1 * v2 + v2**2)
    c2 = coupling * math.sin(q2) * v1**2
    g2 = -9.81 * 0.8 * 0.35 * math.cos(q1 + q2)
    g1 = -9.81 * (1.2 * 0.5 + 0.8) * math.cos(q1) + g2
    t1, t2 = control[0] - c1 - g1, control[1] - c2 - g2
    determinant = m11 * m22 - m12**2
    a1 = (m22 * t1 - m12 * t2) / determinant
    a2 = (m11 * t2 - m12 * t1) / determinant
    dt = robot_contract()["dt"]
    w1, w2 = v1 + dt * a1, v2 + dt * a2
    return [q1 + dt * w1, q2 + dt * w2, w1, w2]


def evaluate_robot_correctness(raw: dict[str, Any]) -> dict[str, Any]:
    contract = robot_contract()
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("Unsupported robot MPC schema")
    if raw.get("contract") != contract:
        raise ValueError("Changed official robot MPC contract")
    derivative_error = raw.get("derivative_max_abs_error")
    if not _finite(derivative_error) or derivative_error < 0:
        raise ValueError("Invalid robot action derivative evidence")
    failures: list[dict[str, Any]] = []
    if derivative_error > 2e-4:
        failures.append({"kind": "derivative_mismatch", "actual": derivative_error})
    runs = raw.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("Missing robot MPC traces")
    max_plant_error = 0.0
    final_norms: dict[str, list[float]] = {}
    iterations: dict[str, list[int]] = {}
    for strategy in contract["strategies"]:
        records = runs.get(strategy)
        if not isinstance(records, list) or len(records) != MPC_REPEATS:
            raise ValueError(f"Incomplete MPC repetitions: {strategy}")
        final_norms[strategy], iterations[strategy] = [], []
        for repeat, run in enumerate(records):
            if not isinstance(run, dict):
                raise ValueError("Malformed MPC run")
            states, controls = run.get("states"), run.get("controls")
            if (not isinstance(states, list) or len(states) != MPC_STEPS + 1
                    or not all(_vector(row, 4) for row in states)
                    or not isinstance(controls, list) or len(controls) != MPC_STEPS
                    or not all(_vector(row, 2) for row in controls)):
                raise ValueError("Incomplete or nonfinite MPC state/control trace")
            if states[0] != contract["initial_state"]:
                raise ValueError("MPC initial state changed")
            for key in ("costs", "initial_costs", "rollout_errors", "solve_ns", "iterations"):
                if not _vector(run.get(key), MPC_STEPS):
                    raise ValueError(f"Invalid MPC evidence: {key}")
            if any(type(n) is not int or not 1 <= n <= contract["max_iterations"]
                   for n in run["iterations"]):
                raise ValueError("Invalid MPC iteration counts")
            if any(n <= 0 for n in run["solve_ns"]):
                raise ValueError("Invalid per-step solve latencies")
            iterations[strategy].extend(run["iterations"])
            def fail(kind: str, **details: Any) -> None:
                failures.append({"kind": kind, "strategy": strategy, "repeat": repeat, **details})
            if run.get("all_solved") is not True:
                fail("solver_not_converged")
            for step, (state, control) in enumerate(zip(states, controls)):
                predicted = plant_step(state, control)
                error = max(abs(a - b) for a, b in zip(predicted, states[step + 1], strict=True))
                max_plant_error = max(max_plant_error, error)
                if error > 1e-8:
                    fail("plant_transition_mismatch", step=step, max_abs_error=error)
                cost, baseline = run["costs"][step], run["initial_costs"][step]
                if cost < 0 or baseline < 0 or cost > baseline + 1e-7 * max(1., baseline):
                    fail("cost_increased", step=step)
                if not 0 <= run["rollout_errors"][step] <= 1e-8:
                    fail("shooting_rollout_mismatch", step=step)
            norm = math.sqrt(sum(x*x for x in states[-1]))
            final_norms[strategy].append(norm)
            if norm > 0.15:
                fail("closed_loop_not_stabilized", final_norm=norm)
            if (max(abs(x) for row in states for x in row[:2]) > 2.5
                    or max(abs(x) for row in states for x in row[2:]) > 4
                    or max(abs(u) for row in controls for u in row) > 100):
                fail("robot_limits_exceeded")
    # Compare the two strategies at every closed-loop state, not only their final values.
    max_strategy_error = max(
        abs(a-b)
        for cold, warm in zip(runs["cold"], runs["shifted_warm"], strict=True)
        for x, y in zip(cold["states"], warm["states"], strict=True)
        for a, b in zip(x, y, strict=True)
    )
    if max_strategy_error > 0.01:
        failures.append({"kind": "warm_start_trajectory_regression",
                         "max_abs_error": max_strategy_error})
    return {"passed": not failures, "failures": failures,
            "first_failure": failures[0] if failures else None,
            "max_plant_abs_error": max_plant_error,
            "max_strategy_abs_error": max_strategy_error,
            "final_state_norms": final_norms,
            "iterations": {name: {"min": min(values), "max": max(values),
                                  "mean": statistics.mean(values), "total": sum(values)}
                           for name, values in iterations.items()}}


def evaluate_robot_profile(raw: dict[str, Any], *, correctness_passed: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": "Pinocchio two-link MPC: cold vs shifted warm start",
        "codegen_end_to_end_speedup": None, "recommendation": "blocked_by_correctness",
    }
    if not correctness_passed:
        return result
    summaries: dict[str, dict[str, float | int]] = {}
    for section, keys in (("stage_timings_ns", MPC_TIMING_KEYS),
                          ("mpc_timings_ns", ("cold", "shifted_warm"))):
        evidence = raw.get(section)
        if not isinstance(evidence, dict):
            raise ValueError(f"Missing {section}")
        for key in keys:
            values = evidence.get(key)
            if not _vector(values, MPC_REPEATS) or any(x <= 0 for x in values):
                raise ValueError(f"Invalid MPC timings: {key}")
            summaries[key] = _summary(values)
    diff = float(summaries["calc_diff"]["p50_ns"])
    backward = float(summaries["backward_pass"]["p50_ns"])
    share = diff / (diff + backward)
    speedup = summaries["cold"]["p50_ns"] / summaries["shifted_warm"]["p50_ns"]
    noisy = max(float(s["cv"]) for s in summaries.values()) > 0.3
    result.update(stages={key: summaries[key] for key in MPC_TIMING_KEYS},
                  mpc={key: summaries[key] for key in ("cold", "shifted_warm")},
                  calc_diff_share_of_direction=share, warm_start_p50_speedup=speedup,
                  max_cv=max(float(s["cv"]) for s in summaries.values()),
                  recommendation="inconclusive_noisy_profile" if noisy else (
                      "candidate_for_codegen_action_model_trial" if share >= 0.5
                      else "optimize_solver_or_rollout_before_codegen"),
                  warm_start_recommendation=("inconclusive_noisy_profile" if noisy else
                      "use_shifted_warm_start" if speedup >= 1.05
                      and summaries["shifted_warm"]["p95_ns"] <= summaries["cold"]["p95_ns"]
                      else "no_measured_warm_start_benefit"))
    result["stage_scope"] = "calc_diff is SolverFDDP.calcDiff including calc and feasibility work"
    result["mpc_scope"] = "32-step loop including setup, replay verification and ABA plant"
    result["per_step_solve"] = {
        name: _summary([latency for run in raw["runs"][name] for latency in run["solve_ns"]])
        for name in ("cold", "shifted_warm")
    }
    return result
