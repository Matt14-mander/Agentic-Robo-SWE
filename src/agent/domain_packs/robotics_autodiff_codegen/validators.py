"""Official multi-sample Dense Jacobian gate for the Phase 6.2a Domain Pack."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

from agent.domain.toolchain import configure_and_build, run_built_executable
from agent.domain_packs.robotics_autodiff_codegen.compatibility import inspect_source
from agent.domain_packs.robotics_autodiff_codegen.constants import PACK_VERSION
from agent.domain_packs.robotics_autodiff_codegen.dense_validation import (
    DEFAULT_DENSE_SPEC,
    deterministic_samples,
    load_dense_report,
)
from agent.domain_packs.robotics_autodiff_codegen.dependencies import (
    INSTALL_PREFIX,
    inspect_dependencies,
)
from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root


_HARNESS = Path(__file__).resolve().parent / "harness"
_TARGET = "autodiff_codegen_validator"
_ARTIFACT_ROOT = PROJECT_ROOT / ".agent_state" / "domain_artifacts" / "autodiff_codegen"
_DENSE_REPORT_NAME = "dense-validation.json"


def validate_autodiff_codegen(workspace: str) -> dict[str, object]:
    try:
        source = resolve_within_root(workspace)
    except ValueError as exc:
        return _failure(str(exc))
    if not source.is_file():
        return _failure(f"Workspace not found: {source}")

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dependencies = inspect_dependencies()
    artifact_dir = _artifact_directory(source_digest, str(dependencies["fingerprint"]))
    context: dict[str, object] = {
        "schema_version": 1,
        "source": relpath_for_display(source),
        "source_sha256": source_digest,
        "pack_version": PACK_VERSION,
        "dependency_fingerprint": dependencies["fingerprint"],
        "validation_spec": DEFAULT_DENSE_SPEC.to_dict(),
    }

    compatibility = inspect_source(source)
    if not compatibility["passed"]:
        findings = cast(list[dict[str, Any]], compatibility["findings"])
        details = [
            f"{item['rule']} line {item['line']}: {item['message']}"
            for item in findings
            if item["severity"] == "error"
        ]
        return _finalize(
            artifact_dir,
            context,
            passed=False,
            stage="compatibility",
            details=details,
            error=None,
            extra={"compatibility": compatibility},
        )

    if not dependencies["available"]:
        missing = cast(list[str], dependencies["missing"])
        return _finalize(
            artifact_dir,
            context,
            passed=False,
            stage="capability",
            details=[],
            error=(
                "Missing pinned autodiff capabilities: "
                + ", ".join(missing)
                + f". Bootstrap explicitly with: {dependencies['bootstrap_command']}"
            ),
            extra={"capabilities": dependencies},
        )

    spec = DEFAULT_DENSE_SPEC
    build = configure_and_build(
        str(_HARNESS),
        _TARGET,
        timeout=240,
        cmake_definitions={
            "WORKSPACE_HEADER": source.as_posix(),
            "WORKSPACE_SHA256": source_digest,
            "AUTODIFF_PREFIX": INSTALL_PREFIX.as_posix(),
            "AUTODIFF_DEPENDENCY_FINGERPRINT": str(dependencies["fingerprint"]),
            "DENSE_VALIDATION_SPEC": spec.fingerprint,
            "DOMAIN_PACK_VERSION": PACK_VERSION,
            "OUTPUT_ATOL": format(spec.output_atol, ".17g"),
            "OUTPUT_RTOL": format(spec.output_rtol, ".17g"),
            "JACOBIAN_ATOL": format(spec.jacobian_atol, ".17g"),
            "JACOBIAN_RTOL": format(spec.jacobian_rtol, ".17g"),
        },
    )
    context["build"] = build.to_dict()
    if not build.passed or not build.build_dir:
        return _finalize(
            artifact_dir,
            context,
            passed=False,
            stage="build",
            details=[build.stderr or build.stdout],
            error=build.error,
        )

    build_path = resolve_within_root(build.build_dir)
    for stale_report in build_path.rglob(_DENSE_REPORT_NAME):
        if stale_report.is_file():
            stale_report.unlink()
    execution = run_built_executable(
        build.build_dir,
        _TARGET,
        args=(
            "--seed",
            str(spec.seed),
            "--random-samples",
            str(spec.random_samples),
            "--report",
            _DENSE_REPORT_NAME,
        ),
        timeout=120,
    )
    context["execution"] = execution
    report_path = _find_dense_report(build_path)
    if report_path is None:
        output = str(execution["stdout"] or execution["stderr"]).strip()
        return _finalize(
            artifact_dir,
            context,
            passed=False,
            stage="runtime",
            details=[output] if output else [],
            error="Dense validator did not produce its structured report",
        )
    try:
        dense_report = load_dense_report(report_path, spec)
    except ValueError as exc:
        return _finalize(
            artifact_dir,
            context,
            passed=False,
            stage="report_validation",
            details=[],
            error=str(exc),
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, artifact_dir / _DENSE_REPORT_NAME)
    _write_json_atomic(artifact_dir / "samples.json", {
        "schema_version": 1,
        "spec_fingerprint": spec.fingerprint,
        "samples": deterministic_samples(spec),
    })
    worst = dense_report.get("worst_failure")
    if worst is not None:
        _write_json_atomic(artifact_dir / "failing-input.json", {
            "schema_version": 1,
            "seed": spec.seed,
            "failure": worst,
        })
    else:
        (artifact_dir / "failing-input.json").unlink(missing_ok=True)

    passed = bool(dense_report["passed"]) and bool(execution["passed"])
    metrics = cast(dict[str, Any], dense_report["metrics"])
    details = [_metrics_summary(metrics, spec.sample_count)]
    if not passed and isinstance(worst, dict):
        details.append(_failure_summary(worst, spec.input_names, spec.output_names))
    return _finalize(
        artifact_dir,
        context,
        passed=passed,
        stage="complete" if passed else "dense_correctness",
        details=details,
        error=None,
        extra={"dense_report": dense_report},
    )


def _artifact_directory(source_digest: str, dependency_fingerprint: str) -> Path:
    identity = json.dumps({
        "source": source_digest,
        "dependency": dependency_fingerprint,
        "spec": DEFAULT_DENSE_SPEC.fingerprint,
        "pack": PACK_VERSION,
    }, sort_keys=True)
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    return _ARTIFACT_ROOT / fingerprint[:20]


def _find_dense_report(build_dir: Path) -> Path | None:
    reports = sorted(path for path in build_dir.rglob(_DENSE_REPORT_NAME) if path.is_file())
    return reports[0] if len(reports) == 1 else None


def _metrics_summary(metrics: dict[str, Any], sample_count: int) -> str:
    return (
        f"Dense gate checked {sample_count} samples; failures={metrics.get('failures')}; "
        f"non_finite={metrics.get('non_finite')}; "
        f"max_output_abs={metrics.get('max_output_absolute_error')}; "
        f"max_jacobian_abs={metrics.get('max_jacobian_absolute_error')}; "
        f"max_jacobian_rel_fro={metrics.get('max_jacobian_frobenius_relative_error')}"
    )


def _failure_summary(
    failure: dict[str, Any],
    input_names: tuple[str, str],
    output_names: tuple[str, str],
) -> str:
    row = int(failure.get("row", 0))
    column = int(failure.get("column", 0))
    output_name = output_names[row] if 0 <= row < len(output_names) else f"output[{row}]"
    input_name = input_names[column] if 0 <= column < len(input_names) else f"input[{column}]"
    element = (
        output_name
        if failure.get("quantity") == "output"
        else f"{output_name}/{input_name}"
    )
    return (
        f"Worst failure: {failure.get('comparison')} sample={failure.get('sample_index')} "
        f"kind={failure.get('sample_kind')} input={failure.get('input')} "
        f"element={element} expected={failure.get('expected')} "
        f"actual={failure.get('actual')} normalized_error={failure.get('normalized_error')}"
    )


def _finalize(
    artifact_dir: Path,
    context: dict[str, object],
    *,
    passed: bool,
    stage: str,
    details: list[str],
    error: str | None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **context,
        "passed": passed,
        "stage": stage,
        "details": details,
        "error": error,
        **(extra or {}),
    }
    report_path = artifact_dir / "validation.json"
    _write_json_atomic(report_path, payload)
    dense_payload = (extra or {}).get("dense_report")
    metrics = dense_payload.get("metrics", {}) if isinstance(dense_payload, dict) else {}
    return {
        "passed": passed,
        "details": [*details, f"diagnostics={relpath_for_display(report_path)}"],
        "error": error,
        "stage": stage,
        "artifacts": {
            "directory": relpath_for_display(artifact_dir),
            "validation": relpath_for_display(report_path),
        },
        "metrics": metrics,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _failure(error: str) -> dict[str, object]:
    return {"passed": False, "details": [], "error": error}
