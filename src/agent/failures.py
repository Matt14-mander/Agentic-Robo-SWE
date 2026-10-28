"""Versioned failure taxonomy based on explicit stage/protocol evidence, not prose guesses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Category = Literal[
    "none", "environment", "timeout", "budget_exhausted", "invalid_baseline",
    "compile_error", "runtime_error", "numerical_mismatch", "derivative_mismatch",
    "safety_violation", "behavioral_mismatch", "invalid_evidence", "tool_error",
    "model_error", "unclassified", "compatibility_error", "performance_regression",
]


@dataclass(frozen=True)
class Failure:
    category: Category
    stage: str
    reason: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(payload: dict[str, Any]) -> Failure:
    """A passing build is only a passing build, not repair acceptance.

    Unknown errors remain unknown. A compiler diagnostic in `error` is not an
    environment outage; explicit missing capabilities are an environment outage.
    """
    stage = str(payload.get("stage") or "unspecified")
    error = payload.get("error")
    if payload.get("timed_out") is True:
        category: Category = "timeout"
    elif payload.get("budget_exhausted") is True:
        category = "budget_exhausted"
    elif payload.get("capability_missing") or (payload.get("passed") is False and
                                               stage in {"capability", "environment"}):
        category = "environment"
    elif type(payload.get("passed")) is not bool:
        category = "invalid_evidence"
    elif payload["passed"] and error:
        category = "invalid_evidence"
    elif payload["passed"]:
        category = "none"
    else:
        mapping: dict[str, Category] = {
            "build": "compile_error", "compile": "compile_error",
            "runtime": "runtime_error", "correctness": "numerical_mismatch",
            "dense_correctness": "derivative_mismatch", "sparse_correctness": "derivative_mismatch",
            "derivatives": "derivative_mismatch", "safety": "safety_violation",
            "compatibility": "compatibility_error", "performance": "performance_regression",
            "control_loop_correctness": "behavioral_mismatch", "report": "invalid_evidence",
            "behavior": "behavioral_mismatch", "source": "invalid_evidence",
            "report_validation": "invalid_evidence", "protocol": "invalid_evidence",
            "tool": "tool_error", "model": "model_error",
        }
        category = mapping.get(stage, "unclassified")
    return Failure(category, stage, str(error or stage)[:1000])


def with_failure(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "failure": classify(payload).to_dict()}


def classify_trial(diagnosis: dict[str, Any]) -> Failure:
    """Adapt an existing simulation diagnosis without parsing its free-form text."""
    status = diagnosis.get("status")
    if not isinstance(status, str) or status not in {"pass", "fail"}:
        return Failure("invalid_evidence", "simulation", "invalid trial status")
    stage = "safety" if diagnosis.get("first_safety_event") else (
        "behavior" if diagnosis.get("category") == "tracking" else "simulation")
    return classify({"passed": status == "pass", "stage": stage})
