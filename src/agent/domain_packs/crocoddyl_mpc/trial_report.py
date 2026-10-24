"""Fail-closed independent judge for controlled ABA derivative replacement."""
from __future__ import annotations

import math
from typing import Any

from agent.domain_packs.crocoddyl_mpc.report import _finite, _summary
from agent.domain_packs.crocoddyl_mpc.robot_report import (
    _vector, evaluate_robot_correctness, plant_step,
)
from agent.domain_packs.crocoddyl_mpc.spec import (
    MPC_REPEATS, TRIAL_BACKENDS, robot_contract, trial_contract, trial_samples,
)

ACTION_SIZES = {"xnext": 4, "Fx": 16, "Fu": 8, "Lx": 4, "Lu": 2,
                "Lxx": 16, "Lxu": 8, "Luu": 4}


def acceleration(input_: list[float]) -> list[float]:
    predicted = plant_step(input_[:4], input_[4:])
    return [(predicted[i + 2] - input_[i + 2]) / .02 for i in range(2)]


def numerical_jacobian(input_: list[float]) -> list[float]:
    columns = []
    for j in range(6):
        plus, minus = input_.copy(), input_.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        columns.append([(a-b) / 2e-5 for a, b in zip(
            acceleration(plus), acceleration(minus), strict=True)])
    return [columns[j][i] for i in range(2) for j in range(6)]


def _error(actual: list[float], expected: list[float]) -> float:
    return max(abs(a-b) for a, b in zip(actual, expected, strict=True))


def expected_cost(input_: list[float], *, terminal: bool) -> dict[str, Any]:
    """Independent fixed quadratic cost, not a second call into the C++ action."""
    weights = [100., 100., 5., 5.]
    scale = 20. if terminal else .02
    residual = [input_[4] + 16.4808, input_[5] + 2.7468]
    effort = 0. if terminal else .001 * scale
    return {"cost": .5 * scale * sum(w*x*x for w, x in zip(weights, input_[:4], strict=True))
                    + .5 * effort * sum(u*u for u in residual),
            "Lx": [scale*w*x for w, x in zip(weights, input_[:4], strict=True)],
            "Lu": [effort*u for u in residual],
            "Lxx": [scale*weights[r] if r == c else 0. for r in range(4) for c in range(4)],
            "Lxu": [0.] * 8, "Luu": [effort, 0., 0., effort]}


def evaluate_trial_correctness(raw: dict[str, Any], *, repeats: int = MPC_REPEATS) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema_version") != 3:
        raise ValueError("Unsupported CodeGen trial schema")
    if raw.get("contract") != trial_contract(repeats):
        raise ValueError("Changed controlled CodeGen trial contract")
    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) != 25:
        raise ValueError("Incomplete action samples")
    failures: list[dict[str, Any]] = []
    maxima = {"value": 0., "jacobian": 0., "action": 0., "trajectory": 0., "control": 0.}

    def check(kind: str, error: float, tolerance: float, **context: Any) -> None:
        if kind in maxima:
            maxima[kind] = max(maxima[kind], error)
        if error > tolerance:
            failures.append({"kind": kind, "max_abs_error": error, **context})

    for index, (sample, input_) in enumerate(zip(samples, trial_samples(), strict=True)):
        if not isinstance(sample, dict) or not _vector(sample.get("input"), 6):
            raise ValueError("Invalid action sample input")
        if _error(sample["input"], input_) > 1e-14:
            raise ValueError("Changed sample input or order")
        evidence = sample.get("backends")
        if not isinstance(evidence, dict) or set(evidence) != set(TRIAL_BACKENDS):
            raise ValueError("Missing derivative backend")
        for name, obs in evidence.items():
            if not isinstance(obs, dict) or not _vector(obs.get("value"), 2) or not _vector(obs.get("jacobian"), 12):
                raise ValueError("Invalid ABA values/Jacobian")
            check("value", _error(obs["value"], acceleration(input_)), 1e-8, sample=index, backend=name)
            check("jacobian", _error(obs["jacobian"], numerical_jacobian(input_)), 2e-5,
                  sample=index, backend=name)
            for section in ("action", "terminal"):
                action = obs.get(section)
                if not isinstance(action, dict) or not _finite(action.get("cost")):
                    raise ValueError("Invalid action cost evidence")
                for key, size in ACTION_SIZES.items():
                    if not _vector(action.get(key), size):
                        raise ValueError(f"Invalid {section} quantity: {key}")
                    check("action", _error(action[key], evidence["native"][section][key]),
                          1e-8, sample=index, backend=name, quantity=key, section=section)
                check("action", abs(action["cost"] - evidence["native"][section]["cost"]),
                      1e-10, sample=index, backend=name, quantity="cost", section=section)
                expected = expected_cost(input_, terminal=section == "terminal")
                check("action", abs(action["cost"] - expected["cost"]), 1e-8,
                      sample=index, backend=name, quantity="cost_physics", section=section)
                for key in ("Lx", "Lu", "Lxx", "Lxu", "Luu"):
                    check("action", _error(action[key], expected[key]), 1e-8,
                          sample=index, backend=name, quantity=key+"_physics", section=section)
                if section == "terminal":
                    check("action", _error(action["xnext"], input_[:4]), 1e-12,
                          sample=index, backend=name, quantity="terminal_map")
                    identity = [1. if r == c else 0. for r in range(4) for c in range(4)]
                    check("action", _error(action["Fx"], identity), 1e-12,
                          sample=index, backend=name, quantity="terminal_Fx")
                    check("action", _error(action["Fu"], [0.] * 8), 1e-12,
                          sample=index, backend=name, quantity="terminal_Fu")
            # Independently check the integrated map and its state/control derivatives.
            predicted = plant_step(input_[:4], input_[4:])
            check("action", _error(obs["action"]["xnext"], predicted), 1e-8,
                  sample=index, backend=name, quantity="xnext_physics")
            jac = numerical_jacobian(input_)
            fx = [(1. if r == c else 0.) + (.02 if r < 2 and c == r+2 else 0.)
                  + (.0004 if r < 2 else .02) * jac[(r % 2)*6+c]
                  for r in range(4) for c in range(4)]
            fu = [(.0004 if r < 2 else .02)*jac[(r % 2)*6+c+4]
                  for r in range(4) for c in range(2)]
            check("action", _error(obs["action"]["Fx"], fx), 2e-6,
                  sample=index, backend=name, quantity="Fx_physics")
            check("action", _error(obs["action"]["Fu"], fu), 2e-6,
                  sample=index, backend=name, quantity="Fu_physics")
    runs = raw.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(TRIAL_BACKENDS):
        raise ValueError("Missing three-way MPC traces")
    judges = {}
    for name in TRIAL_BACKENDS:
        # Reuse all existing plant, convergence, limits and rollout gates; both slots
        # intentionally contain the SAME shifted-warm trace, not a cold comparison.
        judge = evaluate_robot_correctness({"schema_version": 2, "contract": robot_contract(),
            "derivative_max_abs_error": 0.,
            "runs": {"cold": runs[name], "shifted_warm": runs[name]}}, repeats=repeats)
        judges[name] = judge
        failures.extend({**fault, "backend": name} for fault in judge["failures"])
    for name in TRIAL_BACKENDS[1:]:
        for repeat, (base, candidate) in enumerate(zip(runs["native"], runs[name], strict=True)):
            for key, kind, tolerance in (("states", "trajectory", 1e-6), ("controls", "control", 1e-5)):
                error = max(_error(a, b) for a, b in zip(base[key], candidate[key], strict=True))
                check(kind, error, tolerance, backend=name, repeat=repeat)
            for a, b in zip(base["costs"], candidate["costs"], strict=True):
                check("closed_loop_cost", abs(a-b), 1e-6*max(1., abs(a)), backend=name, repeat=repeat)
    return {"passed": not failures, "failures": failures,
            "first_failure": failures[0] if failures else None,
            "max_errors": maxima, "sample_count": 25, "backends": judges}


def evaluate_trial_profile(
    raw: dict[str, Any], *, correctness_passed: bool, repeats: int = MPC_REPEATS,
) -> dict[str, Any]:
    contract = trial_contract(repeats)
    result: dict[str, Any] = {"recommendation": "blocked_by_correctness",
        "scope": "single-process paired 32-step shifted-warm MPC; verification workload, not hard real time",
        "codegen_end_to_end_speedup": None}
    if not correctness_passed:
        return result
    if raw.get("contract") != contract:
        raise ValueError("Changed controlled CodeGen trial contract")
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for section in ("calc_diff_timings_ns", "mpc_timings_ns"):
        timings = raw.get(section)
        if not isinstance(timings, dict) or set(timings) != set(TRIAL_BACKENDS):
            raise ValueError("Missing paired trial timings")
        summaries[section] = {}
        for name in TRIAL_BACKENDS:
            values = timings[name]
            if not _vector(values, repeats) or any(x <= 0 for x in values):
                raise ValueError("Invalid trial latencies")
            summaries[section][name] = _summary(values)
    setup = raw.get("generation_compile_load_seconds")
    if not _finite(setup) or setup <= 0:
        raise ValueError("Missing generation/compile/load duration")
    mpc, diff = summaries["mpc_timings_ns"], summaries["calc_diff_timings_ns"]
    speedup = mpc["native"]["p50_ns"] / mpc["codegen"]["p50_ns"]
    cv = max(float(mpc[name]["cv"]) for name in ("native", "codegen"))
    paired = [a/b for a, b in zip(raw["mpc_timings_ns"]["native"], raw["mpc_timings_ns"]["codegen"], strict=True)]
    wins = sum(r >= 1.05 for r in paired)
    minimum_wins = (4 * repeats + 4) // 5
    p95_delta = mpc["codegen"]["p95_ns"] - mpc["native"]["p95_ns"]
    gates = {
        "minimum_p50_speedup": {"actual": speedup, "minimum": 1.05, "passed": speedup >= 1.05},
        "minimum_paired_wins": {"actual": wins, "minimum": minimum_wins, "passed": wins >= minimum_wins},
        "p95_non_regression": {"delta_ns": p95_delta, "maximum_delta_ns": 0,
            "delta_percent": 100 * p95_delta / mpc["native"]["p95_ns"], "passed": p95_delta <= 0},
        "maximum_mpc_cv": {"actual": cv, "maximum": .3, "passed": cv <= .3},
    }
    rejection_reasons = [name for name, gate in gates.items() if not gate["passed"]]
    benefit = not rejection_reasons
    saved = mpc["native"]["p50_ns"] - mpc["codegen"]["p50_ns"]
    loops = math.ceil(setup * 1e9 / saved) if saved > 0 else None
    result.update(calc_diff=diff, mpc=mpc, codegen_end_to_end_speedup=speedup,
        codegen_calc_diff_speedup=diff["native"]["p50_ns"] / diff["codegen"]["p50_ns"],
        paired_speedups=paired, paired_wins=wins, max_mpc_cv=cv,
        repetitions=repeats, gates=gates, rejection_reasons=rejection_reasons,
        percentile_method="nearest_rank", p95_order_statistic=math.ceil(.95 * repeats),
        tail_sample_warning=f"P95 equals the sample maximum at {repeats} repeats; use 50-100 for follow-up"
            if math.ceil(.95 * repeats) == repeats else None,
        production_adoption=False, default_backend="native",
        generation_compile_load_seconds=setup, break_even_32_step_rollouts=loops,
        break_even_control_steps=loops * 32 if loops else None,
        amortization_scope="CG tape/source generation + gcc compile + load; excludes baseline build and CppAD baseline tape",
        recommendation="inconclusive_noisy_profile" if cv > .3 else (
            "candidate_for_repeated_process_trial" if benefit else "keep_native_no_measured_end_to_end_benefit"))
    result["per_step_solve"] = {
        name: _summary([ns for run in raw["runs"][name] for ns in run["solve_ns"]])
        for name in TRIAL_BACKENDS
    }
    result["calc_diff_scope"] = "SolverFDDP.calcDiff includes calc and feasibility work"
    result["calc_diff_noisy"] = any(float(diff[name]["cv"]) > .3 for name in ("native", "codegen"))
    result["mpc_scope"] = "32-step loop includes setup, replay verification and native ABA plant; excludes CodeGen setup"
    return result
