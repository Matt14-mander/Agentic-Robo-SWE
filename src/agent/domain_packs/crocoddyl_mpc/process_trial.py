"""Sequential independent native processes, not concurrent throughput testing."""
from __future__ import annotations

import json
import statistics
import uuid
from collections.abc import Callable
from typing import Any

from agent.domain_packs.crocoddyl_mpc import validators
from agent.domain_packs.crocoddyl_mpc.report import _summary
from agent.domain_packs.crocoddyl_mpc.spec import PACK_VERSION, trial_contract
from agent.domain_packs.crocoddyl_mpc.trial_report import (
    evaluate_trial_correctness, evaluate_trial_profile,
)
from agent.tools._paths import relpath_for_display


def check_process_count(processes: int) -> None:
    if type(processes) is not int or not 3 <= processes <= 20:
        raise ValueError("Independent process count must be an integer in [3, 20]")


def evaluate_process_trial(raws: list[dict[str, Any]], *, processes: int) -> dict[str, Any]:
    """Rejudge every raw trace; never trust a child report's pass/recommendation."""
    check_process_count(processes)
    if len(raws) != processes:
        raise ValueError("Incomplete independent process evidence")
    if any(not isinstance(raw, dict) for raw in raws):
        raise ValueError("Malformed independent process evidence")
    pids = [raw.get("process_id") for raw in raws]
    if any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != processes:
        raise ValueError("Missing or reused native process IDs")
    correctness = [evaluate_trial_correctness(raw) for raw in raws]
    passed = all(gate["passed"] for gate in correctness)
    profile: dict[str, Any] = {"recommendation": "blocked_by_correctness",
        "scope": "sequential fresh native processes on one machine; identical two-link workload",
        "independent_processes": processes, "repetitions_per_process": 10,
        "production_adoption": False}
    if not passed:
        return {"correctness": {"passed": False, "processes": correctness}, "profile": profile}
    # Check determinism across process boundaries as well as within each process.
    cross_error = max(abs(a-b) for raw in raws[1:]
        for first, current in zip(raws[0]["runs"]["native"], raw["runs"]["native"], strict=True)
        for x, y in zip(first["states"], current["states"], strict=True)
        for a, b in zip(x, y, strict=True))
    if cross_error > 1e-6:
        return {"correctness": {"passed": False, "processes": correctness,
                                "cross_process_state_max_abs_error": cross_error}, "profile": profile}
    profiles = [evaluate_trial_profile(raw, correctness_passed=True) for raw in raws]
    ratios = [float(p["codegen_end_to_end_speedup"]) for p in profiles]
    summary = _summary(ratios)
    noisy = any(p["recommendation"] == "inconclusive_noisy_profile" for p in profiles)
    unstable = float(summary["cv"]) > .15
    benefit = all(p["recommendation"] == "candidate_for_repeated_process_trial" for p in profiles)
    profile.update(
        recommendation="inconclusive_noisy_profile" if noisy or unstable else (
            "candidate_for_opt_in_codegen_pilot" if benefit else "keep_native_inconsistent_process_benefit"),
        process_speedups=ratios, speedup_min=min(ratios), speedup_median=statistics.median(ratios),
        speedup_max=max(ratios), speedup_cv=summary["cv"],
        beneficial_processes=sum(p["recommendation"] == "candidate_for_repeated_process_trial" for p in profiles),
        criteria={"every_process_minimum_p50_speedup": 1.05, "every_process_minimum_paired_wins": 8,
                  "every_process_p95_non_regression": True, "maximum_within_process_cv": .3,
                  "maximum_between_process_speedup_cv": .15},
        processes=[{"process_id": pid, **p} for pid, p in zip(pids, profiles, strict=True)],
        statistical_unit="one fresh native process; repetitions are not independent process samples",
    )
    return {"correctness": {"passed": True, "processes": correctness,
                            "cross_process_state_max_abs_error": cross_error}, "profile": profile}


def validate_process_trial(
    _: str = "", *, processes: int = 3, progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Each existing validator invocation launches a NEW C++ executable process.

    The harness build is reused through its fingerprint cache; tapes, generated
    libraries, warmup and measurement state are never reused between processes.
    """
    check_process_count(processes)
    directory = validators.ARTIFACT_ROOT / ("process-trial-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=True)
    context: dict[str, Any] = {"schema_version": 1, "pack_version": PACK_VERSION,
        "contract": trial_contract(), "requested_processes": processes, "processes": [],
        "execution_mode": "sequential_independent_native_processes"}

    def finish(passed: bool, stage: str, error: str | None = None) -> dict[str, Any]:
        context.update(passed=passed, stage=stage, error=error)
        validators._save(directory / "validation.json", context)
        return {"passed": passed, "stage": stage, "error": error,
                "artifacts": {"directory": relpath_for_display(directory),
                              "validation": relpath_for_display(directory / "validation.json")},
                "correctness": context.get("correctness", {}), "profile": context.get("profile", {})}

    raws = []
    identity = None
    try:
        validators._save(directory / "checkpoint.json", context)
        if progress:
            progress(f"process trial checkpoint: {relpath_for_display(directory / 'checkpoint.json')}")
        for index in range(processes):
            context["current_process"] = index+1
            validators._save(directory / "checkpoint.json", context)
            if progress:
                progress(f"independent process {index+1}/{processes}: fresh native run and unique CodeGen library")
            child = validators.validate_crocoddyl_mpc(codegen=True, progress=progress)
            context["processes"].append(child)
            validators._save(directory / "checkpoint.json", context)
            if not child["passed"]:
                return finish(False, "process_failure", f"Process {index+1}: {child['stage']}: {child.get('error')}")
            child_directory = validators.PROJECT_ROOT / child["artifacts"]["directory"]
            evidence = json.loads((child_directory / "validation.json").read_text(encoding="utf-8"))
            current = (evidence["dependencies"]["fingerprint"],
                       evidence["environment"]["toolchain"]["fingerprint"],
                       evidence["build"]["fingerprint"], evidence["execution"]["executable_sha256"])
            if identity is not None and current != identity:
                return finish(False, "identity", "Dependencies, toolchain or executable changed between processes")
            identity = current
            context["identity"] = dict(zip(
                ("dependencies", "toolchain", "build", "executable_sha256"), current, strict=True))
            raws.append(json.loads((child_directory / "raw.json").read_text(encoding="utf-8")))
            if progress:
                progress(f"saved process {index+1}/{processes}: {child['artifacts']['validation']}")
        aggregate = evaluate_process_trial(raws, processes=processes)
        context.update(aggregate)
        validators._save(directory / "correctness.json", aggregate["correctness"])
        validators._save(directory / "profile.json", aggregate["profile"])
        if progress:
            progress(f"process trial complete: {aggregate['profile']['recommendation']}")
        return finish(aggregate["correctness"]["passed"], "complete" if aggregate["correctness"]["passed"] else "correctness")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return finish(False, "report_error", f"{type(exc).__name__}: {exc}")
