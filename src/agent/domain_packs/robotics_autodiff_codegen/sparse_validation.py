"""Independent COO structure and numeric gates; never infer sparsity from values."""

from __future__ import annotations

import math
from typing import Any, cast

from agent.domain_packs.robotics_autodiff_codegen.dense_validation import (
    DEFAULT_DENSE_SPEC,
    DenseValidationSpec,
    deterministic_samples,
)


SPARSE_SCHEMA_VERSION = 1
MODEL_PATTERNS = {
    "coupled": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "triangular": ((0, 0), (1, 0), (1, 1)),
}


def reconstruct_coo(
    coo: object,
    *,
    shape: tuple[int, int],
    expected_pattern: tuple[tuple[int, int], ...],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Reject malformed/duplicate/missing coordinates, accepting any COO ordering."""
    matrix = [0.0] * (shape[0] * shape[1])
    failures: list[dict[str, Any]] = []
    if not isinstance(coo, dict):
        return matrix, [{"kind": "invalid_coo", "message": "COO must be an object"}]
    rows, columns, values = (coo.get(key) for key in ("rows", "columns", "values"))
    if not isinstance(rows, list) or not isinstance(columns, list) or not isinstance(values, list):
        return matrix, [{"kind": "invalid_coo", "message": "COO arrays are required"}]
    if not (len(rows) == len(columns) == len(values)):
        return matrix, [{"kind": "length_mismatch"}]
    seen: set[tuple[int, int]] = set()
    for position, (row, column, value) in enumerate(zip(rows, columns, values, strict=True)):
        if type(row) is not int or type(column) is not int:
            failures.append({"kind": "invalid_index", "position": position})
            continue
        if not (0 <= row < shape[0] and 0 <= column < shape[1]):
            failures.append({"kind": "out_of_bounds", "row": row, "column": column})
            continue
        coordinate = (row, column)
        if coordinate in seen:
            failures.append({"kind": "duplicate_coordinate", "row": row, "column": column})
        seen.add(coordinate)
        if type(value) not in (int, float) or not math.isfinite(value):
            failures.append({"kind": "non_finite_value", "row": row, "column": column})
            continue
        matrix[row * shape[1] + column] = float(value)
    expected = set(expected_pattern)
    for kind, coordinates in (
        ("missing_coordinate", expected - seen),
        ("unexpected_coordinate", seen - expected),
    ):
        for row, column in sorted(coordinates):
            failures.append({"kind": kind, "row": row, "column": column})
    return matrix, failures


def evaluate_sparse_report(
    payload: dict[str, Any],
    *,
    spec: DenseValidationSpec = DEFAULT_DENSE_SPEC,
    contract: str = "coupled",
) -> dict[str, Any]:
    """Recompute verdict from raw CppAD/CodeGen observations, not their pass flags."""
    pattern = MODEL_PATTERNS[contract]
    observations = payload.get("sparse_samples")
    if not isinstance(observations, list) or len(observations) != spec.sample_count:
        raise ValueError("Sparse observations missing or incomplete; refusing Dense-only success")
    failures: list[dict[str, Any]] = []
    max_absolute = 0.0
    max_normalized = 0.0
    max_relative = 0.0
    worst: dict[str, Any] | None = None
    for index, (sample, observation) in enumerate(zip(
        deterministic_samples(spec), observations, strict=True
    )):
        if not isinstance(observation, dict) or type(observation.get("index")) is not int:
            raise ValueError("Sparse observation index is invalid")
        if observation["index"] != index:
            raise ValueError("Sparse observation order/coverage does not match official samples")
        location = {"sample_index": index, "sample_kind": sample["kind"], "input": sample["input"]}
        matrices = {}
        for backend in ("cppad", "codegen"):
            matrix, problems = reconstruct_coo(
                observation.get(backend), shape=(2, 2), expected_pattern=pattern
            )
            matrices[backend] = matrix
            failures.extend({**location, "backend": backend, **problem} for problem in problems)
        x0, x1 = cast(list[float], sample["input"])
        references = {
            "analytic": [2.0 * x0, 0.0 if contract == "triangular" else 0.5, x1, x0 + 2 * x1],
            "cppad_dense": observation.get("cppad_dense"),
            "codegen_dense": observation.get("codegen_dense"),
            "finite_difference": observation.get("finite_difference"),
            "cppad_sparse": matrices["cppad"],
        }
        for reference_name, reference in references.items():
            if not isinstance(reference, list) or len(reference) != 4:
                failures.append({**location, "kind": "reference_dimension", "reference": reference_name})
                continue
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in reference):
                failures.append({**location, "kind": "non_finite_reference", "reference": reference_name})
                continue
            for backend, matrix in matrices.items():
                comparison = f"{backend}_sparse_vs_{reference_name}"
                absolute_errors = []
                for offset, (actual, expected) in enumerate(zip(matrix, reference, strict=True)):
                    absolute = abs(actual - expected)
                    normalized = absolute / (spec.jacobian_atol + spec.jacobian_rtol * abs(expected))
                    absolute_errors.append(absolute)
                    max_absolute = max(max_absolute, absolute)
                    max_normalized = max(max_normalized, normalized)
                    if normalized > 1.0:
                        failure = {
                            **location, "kind": "value_mismatch", "comparison": comparison,
                            "row": offset // 2, "column": offset % 2,
                            "actual": actual, "expected": expected,
                            "absolute_error": absolute, "normalized_error": normalized,
                        }
                        failures.append(failure)
                        if worst is None or normalized > worst["normalized_error"]:
                            worst = failure
                relative = math.hypot(*absolute_errors) / max(1.0, math.hypot(*reference))
                max_relative = max(max_relative, relative)
                if relative > spec.jacobian_rtol:
                    failures.append({
                        **location, "kind": "frobenius_error", "comparison": comparison,
                        "relative_error": relative,
                    })
    return _json_numbers({
        "schema_version": SPARSE_SCHEMA_VERSION,
        "passed": not failures,
        "contract": contract,
        "shape": [2, 2],
        "expected_pattern": [list(coordinate) for coordinate in pattern],
        "expected_nnz": len(pattern),
        "sample_count": spec.sample_count,
        "seed": spec.seed,
        "failure_count": len(failures),
        "structure_failures": sum(item["kind"] in {
            "invalid_coo", "length_mismatch", "invalid_index", "out_of_bounds",
            "duplicate_coordinate", "missing_coordinate", "unexpected_coordinate",
        } for item in failures),
        "max_absolute_error": max_absolute,
        "max_normalized_error": max_normalized,
        "max_frobenius_relative_error": max_relative,
        "worst_failure": worst or (failures[0] if failures else None),
        "failures": failures,
    })


def _json_numbers(value: Any) -> Any:
    """Keep overflowing failure diagnostics valid JSON without weakening the verdict."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_numbers(item) for item in value]
    return value
