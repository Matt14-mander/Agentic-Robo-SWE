"""Recompute numerical verdicts and conservative microbenchmark decisions."""

from __future__ import annotations

import math
import statistics
from typing import Any, TypeGuard

from agent.domain_packs.pinocchio_rnea.spec import (
    BATCH_SIZE, REPEATS, WARMUP, model_contract, samples,
)


BACKENDS = ("pinocchio", "cppad", "codegen", "finite_difference")


def _vector(value: Any, size: int) -> TypeGuard[list[float]]:
    return (isinstance(value, list) and len(value) == size
            and all(type(x) in (float, int) and math.isfinite(x) for x in value))


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
    failures = []
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
