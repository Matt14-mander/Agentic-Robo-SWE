"""Deterministic Dense Jacobian validation contract shared with the C++ harness."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from agent.domain_packs.robotics_autodiff_codegen.constants import (
    DENSE_RANDOM_SAMPLES,
    DENSE_VALIDATION_SEED,
    JACOBIAN_ABSOLUTE_TOLERANCE,
    JACOBIAN_RELATIVE_TOLERANCE,
    OUTPUT_ABSOLUTE_TOLERANCE,
    OUTPUT_RELATIVE_TOLERANCE,
)


_MASK_64 = (1 << 64) - 1
_UINT53_SCALE = 1.0 / 9_007_199_254_740_992.0
_REGRESSION_SAMPLES = (
    ("regression", (0.25, -0.75)),
    ("regression", (1.2, 0.4)),
    ("regression", (-0.9, 1.1)),
)
_BOUNDARY_SAMPLES = (
    ("boundary", (0.0, 0.0)),
    ("boundary", (2.0, 4.0)),
    ("boundary", (-2.0, -4.0)),
    ("boundary", (2.0, -4.0)),
    ("boundary", (-2.0, 4.0)),
)


@dataclass(frozen=True)
class DenseValidationSpec:
    schema_version: int = 1
    seed: int = DENSE_VALIDATION_SEED
    random_samples: int = DENSE_RANDOM_SAMPLES
    input_lower: tuple[float, float] = (-2.0, -4.0)
    input_upper: tuple[float, float] = (2.0, 4.0)
    input_names: tuple[str, str] = ("joint_position", "joint_velocity")
    output_names: tuple[str, str] = ("position_term", "coupled_velocity_term")
    output_atol: float = OUTPUT_ABSOLUTE_TOLERANCE
    output_rtol: float = OUTPUT_RELATIVE_TOLERANCE
    jacobian_atol: float = JACOBIAN_ABSOLUTE_TOLERANCE
    jacobian_rtol: float = JACOBIAN_RELATIVE_TOLERANCE

    @property
    def sample_count(self) -> int:
        return len(_REGRESSION_SAMPLES) + len(_BOUNDARY_SAMPLES) + self.random_samples

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "sample_count": self.sample_count, "fingerprint": self.fingerprint}


DEFAULT_DENSE_SPEC = DenseValidationSpec()


def deterministic_samples(
    spec: DenseValidationSpec = DEFAULT_DENSE_SPEC,
) -> list[dict[str, object]]:
    samples = [
        {"index": index, "kind": kind, "input": list(values)}
        for index, (kind, values) in enumerate((*_REGRESSION_SAMPLES, *_BOUNDARY_SAMPLES))
    ]
    state = spec.seed & _MASK_64 or 1
    for _ in range(spec.random_samples):
        values: list[float] = []
        for lower, upper in zip(spec.input_lower, spec.input_upper, strict=True):
            state, unit = _next_unit(state)
            values.append(lower + (upper - lower) * unit)
        samples.append({"index": len(samples), "kind": "random", "input": values})
    return samples


def load_dense_report(
    path: str | Path,
    spec: DenseValidationSpec = DEFAULT_DENSE_SPEC,
) -> dict[str, Any]:
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dense validation report: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Dense validation report schema_version must be 1")
    if payload.get("seed") != spec.seed:
        raise ValueError("Dense validation report seed does not match the official spec")
    if payload.get("random_samples") != spec.random_samples:
        raise ValueError("Dense validation random sample count does not match the official spec")
    if payload.get("sample_count") != spec.sample_count:
        raise ValueError("Dense validation declared sample count does not match the official spec")
    expected_thresholds = {
        "output_atol": spec.output_atol,
        "output_rtol": spec.output_rtol,
        "jacobian_atol": spec.jacobian_atol,
        "jacobian_rtol": spec.jacobian_rtol,
    }
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or any(
        not isinstance(thresholds.get(key), (int, float))
        or not math.isclose(float(thresholds[key]), target, rel_tol=0.0, abs_tol=1e-18)
        for key, target in expected_thresholds.items()
    ):
        raise ValueError("Dense validation thresholds do not match the official spec")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != spec.sample_count:
        raise ValueError("Dense validation report has an invalid sample count")
    expected_samples = deterministic_samples(spec)
    for actual, expected in zip(samples, expected_samples, strict=True):
        if not isinstance(actual, dict):
            raise ValueError("Dense validation sample must be an object")
        if actual.get("index") != expected["index"] or actual.get("kind") != expected["kind"]:
            raise ValueError("Dense validation sample identity does not match the official spec")
        values = actual.get("input")
        expected_values = cast(list[float], expected["input"])
        if not isinstance(values, list) or len(values) != len(expected_values):
            raise ValueError("Dense validation sample input dimension is invalid")
        if any(
            not isinstance(value, (int, float))
            or not math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=1e-15)
            for value, target in zip(values, expected_values, strict=True)
        ):
            raise ValueError("Dense validation sample values do not match the official generator")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Dense validation report metrics are missing")
    for key in ("failures", "non_finite", "dimension_failures"):
        if not isinstance(metrics.get(key), int) or metrics[key] < 0:
            raise ValueError(f"Dense validation metric {key} is invalid")
    if payload.get("passed") not in {True, False}:
        raise ValueError("Dense validation report passed must be boolean")
    if payload["passed"] and (
        metrics["failures"] != 0
        or metrics["non_finite"] != 0
        or metrics["dimension_failures"] != 0
        or payload.get("worst_failure") is not None
    ):
        raise ValueError("Dense validation report claims success with failure evidence")
    worst = payload.get("worst_failure")
    if worst is not None:
        if not isinstance(worst, dict) or worst.get("quantity") not in {"output", "jacobian"}:
            raise ValueError("Dense validation worst failure is invalid")
        sample_index = worst.get("sample_index")
        if not isinstance(sample_index, int) or not 0 <= sample_index < spec.sample_count:
            raise ValueError("Dense validation worst failure sample index is invalid")
        for key in ("row", "column"):
            if not isinstance(worst.get(key), int) or worst[key] < 0:
                raise ValueError(f"Dense validation worst failure {key} is invalid")
    return payload


def _next_unit(state: int) -> tuple[int, float]:
    state ^= state >> 12
    state &= _MASK_64
    state ^= (state << 25) & _MASK_64
    state &= _MASK_64
    state ^= state >> 27
    state &= _MASK_64
    result = (state * 2_685_821_657_736_338_717) & _MASK_64
    return state, (result >> 11) * _UINT53_SCALE
