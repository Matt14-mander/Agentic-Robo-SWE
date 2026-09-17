"""Recompute numerical verdicts and conservative microbenchmark decisions."""

from __future__ import annotations

import math
import statistics
from typing import Any, TypeGuard, cast

from agent.domain_packs.pinocchio_rnea.spec import (
    BATCH_SIZE, CONTROL_DEADLINE_NS, CONTROL_DT, CONTROL_REPEATS, CONTROL_STEPS,
    CONTROL_WARMUP, REPEATS, WARMUP, model_contract, samples,
)


BACKENDS = ("pinocchio", "cppad", "codegen", "finite_difference")


def _vector(value: Any, size: int) -> TypeGuard[list[float]]:
    return (isinstance(value, list) and len(value) == size
            and all(type(x) in (float, int) and math.isfinite(x) for x in value))


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(value)


def evaluate_correctness(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("RNEA observations must be a JSON object")
    if raw.get("schema_version") != 1 or raw.get("nq") != 2 or raw.get("nv") != 2:
        raise ValueError("Unsupported report schema or robot dimensions")
    if raw.get("urdf_sha256") != model_contract()["urdf_sha256"]:
        raise ValueError("URDF fingerprint does not match the fixed official model")
    records = raw.get("samples")
    expected = samples()
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("Incomplete official RNEA samples")
    failures: list[dict[str, Any]] = []
    max_error = 0.0
    for index, (record, point) in enumerate(zip(records, expected, strict=True)):
        if not isinstance(record, dict) or record.get("input") != point:
            raise ValueError("RNEA sample input/order does not match official contract")
        for quantity, size, reference, candidates, atol, rtol in (
            ("output", 2, "pinocchio_output",
             ("source_output", "analytic_output", "cppad_output", "codegen_output"),
             1e-9, 1e-8),
            ("jacobian", 12, "analytic_jacobian",
             ("finite_difference_jacobian", "cppad_jacobian", "codegen_jacobian"), 1e-7, 1e-5),
        ):
            target = record.get(reference)
            for name in (reference, *candidates):
                if not _vector(record.get(name), size):
                    failures.append({"sample_index": index, "input": point, "comparison": name,
                                     "kind": "dimension_or_nonfinite"})
            if not _vector(target, size):
                continue
            for name in candidates:
                actual = record.get(name)
                if not _vector(actual, size):
                    continue
                for offset, (value, baseline) in enumerate(zip(actual, target, strict=True)):
                    error = abs(value - baseline) / (atol + rtol * abs(baseline))
                    max_error = max(max_error, error)
                    if error > 1.0:
                        failures.append({
                            "sample_index": index, "input": point, "quantity": quantity,
                            "comparison": f"{name}_vs_{reference}", "kind": "value_mismatch",
                            "row": offset // 6 if quantity == "jacobian" else offset,
                            "column": offset % 6 if quantity == "jacobian" else None,
                            "expected": baseline, "actual": value,
                            "normalized_error": error if math.isfinite(error) else None,
                        })
    return {"passed": not failures, "sample_count": len(expected), "failures": failures,
            "max_normalized_error": max_error if math.isfinite(max_error) else None,
            "first_failure": failures[0] if failures else None}


def evaluate_performance(raw: dict[str, Any], *, correctness_passed: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": "RNEA output + full Jacobian microbenchmark, not an MPC solve",
        "end_to_end_speedup": None,
        "recommendation": "blocked_by_correctness",
        "measured_codegen_benefit": False,
    }
    if not correctness_passed:
        return result
    result["recommendation"] = "insufficient_evidence"
    compile_seconds = raw.get("codegen_compile_seconds")
    if (not isinstance(compile_seconds, (int, float)) or isinstance(compile_seconds, bool)
            or not math.isfinite(compile_seconds)
            or compile_seconds < 0):
        return result
    if raw.get("timing_spec") != {"warmup": WARMUP, "repeats": REPEATS, "batch_size": BATCH_SIZE}:
        return result
    timings = raw.get("timings_ns")
    if not isinstance(timings, dict):
        return result
    summaries = {}
    for name in BACKENDS:
        values = timings.get(name)
        if not _vector(values, REPEATS) or any(value <= 0 for value in values):
            return result
        ordered = sorted(values)
        mean = statistics.mean(values)
        stdev = statistics.pstdev(values)
        summaries[name] = {"count": len(values), "p50_ns": statistics.median(values),
                           "p95_ns": ordered[math.ceil(0.95 * len(values)) - 1],
                           "mean_ns": mean, "stdev_ns": stdev, "cv": stdev / mean}
    reference, generated = summaries["pinocchio"], summaries["codegen"]
    speedup = reference["p50_ns"] / generated["p50_ns"]
    result.update({"backends": summaries, "p50_speedup_vs_analytic": speedup,
                   "timing_unit": "ns/call (batch averages)",
                   "compile_seconds": raw.get("codegen_compile_seconds")})
    if max(reference["cv"], generated["cv"]) > 0.25:
        result["recommendation"] = "inconclusive_noisy_measurements"
    elif speedup >= 1.1 and generated["p95_ns"] <= reference["p95_ns"]:
        result["recommendation"] = "candidate_for_end_to_end_trial"
        result["measured_codegen_benefit"] = True
    else:
        result["recommendation"] = "keep_pinocchio_analytic"
    return result


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    mean = statistics.mean(values)
    return {
        "count": len(values), "p50_ns": statistics.median(values),
        "p95_ns": ordered[math.ceil(0.95 * len(values)) - 1],
        "mean_ns": mean, "stdev_ns": statistics.pstdev(values),
        "cv": statistics.pstdev(values) / mean,
        "batch_average_deadline_misses": sum(value > CONTROL_DEADLINE_NS for value in values),
    }


def evaluate_control_loop(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate one fresh-process closed-loop run without trusting C++ verdicts."""
    expected_spec = {
        "warmup": CONTROL_WARMUP, "repeats": CONTROL_REPEATS,
        "steps": CONTROL_STEPS, "dt": CONTROL_DT, "deadline_ns": CONTROL_DEADLINE_NS,
    }
    loop = raw.get("control_loop")
    if not isinstance(loop, dict) or loop.get("spec") != expected_spec:
        raise ValueError("Missing or changed official control-loop specification")
    timings = loop.get("timings_ns")
    if not isinstance(timings, dict):
        raise ValueError("Missing control-loop timings")
    summaries: dict[str, dict[str, float | int]] = {}
    raw_timings: dict[str, list[float]] = {}
    for backend in ("pinocchio", "codegen"):
        values = timings.get(backend)
        if not _vector(values, CONTROL_REPEATS) or any(value <= 0 for value in values):
            raise ValueError(f"Invalid {backend} control-loop timing samples")
        raw_timings[backend] = [float(value) for value in values]
        summaries[backend] = _latency_summary(raw_timings[backend])
    reference, generated = loop.get("pinocchio"), loop.get("codegen")
    if not isinstance(reference, dict) or not isinstance(generated, dict):
        raise ValueError("Missing control-loop state evidence")
    failures: list[dict[str, Any]] = []
    for field, size, atol, rtol in (
        ("final_state", 4, 1e-10, 1e-8),
        ("checkpoints", 48, 1e-9, 1e-8),
    ):
        expected, actual = reference.get(field), generated.get(field)
        if not _vector(expected, size) or not _vector(actual, size):
            failures.append({"kind": "dimension_or_nonfinite", "field": field})
            continue
        for index, (value, baseline) in enumerate(zip(actual, expected, strict=True)):
            normalized = abs(value - baseline) / (atol + rtol * abs(baseline))
            if normalized > 1.0:
                failures.append({"kind": "trajectory_mismatch", "field": field,
                                 "index": index, "expected": baseline, "actual": value,
                                 "normalized_error": normalized})
    expected_checksum, actual_checksum = reference.get("checksum"), generated.get("checksum")
    if not _finite_number(expected_checksum) or not _finite_number(actual_checksum):
        failures.append({"kind": "invalid_checksum"})
    else:
        expected_number = float(cast(float, expected_checksum))
        actual_number = float(cast(float, actual_checksum))
        checksum_error = abs(actual_number - expected_number) / (
            1e-8 + 1e-8 * abs(expected_number)
        )
        if checksum_error > 1.0:
            failures.append({"kind": "checksum_mismatch", "expected": expected_number,
                             "actual": actual_number, "normalized_error": checksum_error})
    speedup = summaries["pinocchio"]["p50_ns"] / summaries["codegen"]["p50_ns"]
    recommendation = "blocked_by_correctness"
    if not failures:
        if max(summaries[name]["cv"] for name in summaries) > 0.25:
            recommendation = "inconclusive_noisy_measurements"
        elif speedup >= 1.03 and summaries["codegen"]["p95_ns"] <= (
            summaries["pinocchio"]["p95_ns"] * 1.02
        ):
            recommendation = "candidate_for_repeated_process_validation"
        else:
            recommendation = "keep_pinocchio_analytic"
    return {
        "passed": not failures,
        "scope": "PD trajectory -> RNEA+Jacobian -> ABA -> semi-implicit Euler",
        "not_measured": "MPC/Crocoddyl solve, I/O, middleware, hardware actuation",
        "failures": failures, "backends": summaries, "timings_ns": raw_timings,
        "p50_speedup": speedup, "recommendation": recommendation,
        "library_reused": bool(raw.get("library_reused", False)),
    }


def aggregate_control_processes(
    process_reports: list[dict[str, Any]], *, compile_seconds: float
) -> dict[str, Any]:
    """Combine sequential fresh-process measurements and calculate amortization."""
    if len(process_reports) < 3:
        raise ValueError("At least three fresh process reports are required")
    if not math.isfinite(compile_seconds) or compile_seconds < 0:
        raise ValueError("A finite non-negative CodeGen compile duration is required")
    evaluated = [evaluate_control_loop(report) for report in process_reports]
    if not all(result["library_reused"] for result in evaluated):
        raise ValueError("Fresh process reports must reuse the immutable generated library")
    if not all(result["passed"] for result in evaluated):
        return {"passed": False, "recommendation": "blocked_by_correctness",
                "process_count": len(evaluated), "processes": evaluated}
    combined: dict[str, list[float]] = {"pinocchio": [], "codegen": []}
    process_medians: dict[str, list[float]] = {"pinocchio": [], "codegen": []}
    for result in evaluated:
        for backend in combined:
            combined[backend].extend(result["timings_ns"][backend])
            process_medians[backend].append(result["backends"][backend]["p50_ns"])
    summaries = {name: _latency_summary(values) for name, values in combined.items()}
    setup_seconds = [report.get("library_load_seconds") for report in process_reports]
    if not all(_finite_number(value) and value >= 0 for value in setup_seconds):
        raise ValueError("Fresh process library-load durations are invalid")
    setup_numbers = [float(cast(float, value)) for value in setup_seconds]
    between_process_cv = {
        name: statistics.pstdev(values) / statistics.mean(values)
        for name, values in process_medians.items()
    }
    reference = summaries["pinocchio"]
    generated = summaries["codegen"]
    speedup = reference["p50_ns"] / generated["p50_ns"]
    saved_ns = reference["p50_ns"] - generated["p50_ns"]
    break_even_calls = math.ceil(compile_seconds * 1e9 / saved_ns) if saved_ns > 0 else None
    recommendation = "keep_pinocchio_analytic"
    if max(between_process_cv.values()) > 0.15:
        recommendation = "inconclusive_cross_process_variance"
    elif max(summaries[name]["cv"] for name in summaries) > 0.25:
        recommendation = "inconclusive_noisy_measurements"
    elif (speedup >= 1.03 and generated["p95_ns"] <= reference["p95_ns"] * 1.02
          and generated["batch_average_deadline_misses"]
          <= reference["batch_average_deadline_misses"]):
        recommendation = "validated_end_to_end_candidate"
    return {
        "passed": True, "recommendation": recommendation,
        "process_count": len(evaluated), "samples_per_backend": len(combined["codegen"]),
        "backends": summaries, "process_medians_ns": process_medians,
        "between_process_cv": between_process_cv, "p50_speedup": speedup,
        "compile_seconds": compile_seconds, "break_even_calls": break_even_calls,
        "break_even_seconds_at_1khz": break_even_calls / 1000 if break_even_calls else None,
        "fresh_process_library_load_seconds": {
            "values": setup_numbers, "p50": statistics.median(setup_numbers),
            "max": max(setup_numbers),
        },
        "scope": "repeated-process RNEA control-loop benchmark",
        "not_measured": "MPC/Crocoddyl solve or deployed robot throughput",
    }
