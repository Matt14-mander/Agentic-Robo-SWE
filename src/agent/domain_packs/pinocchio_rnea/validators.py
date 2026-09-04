"""Project-local RNEA build/run orchestration with persistent diagnostic artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from pathlib import Path
from typing import Any

from agent.domain.toolchain import configure_and_build, inspect_toolchain, run_built_executable
from agent.domain_packs.pinocchio_rnea.dependencies import PREFIX, inspect_dependencies
from agent.domain_packs.pinocchio_rnea.report import evaluate_correctness, evaluate_performance
from agent.domain_packs.pinocchio_rnea.spec import MODEL_PATH, PACK_ROOT, model_contract, samples
from agent.domain_packs.robotics_autodiff_codegen.dependencies import INSTALL_PREFIX
from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root


ARTIFACT_ROOT = PROJECT_ROOT / ".agent_state/domain_artifacts/pinocchio_rnea"
TARGET = "pinocchio_rnea_validator"


def _save(path: Path, data: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def validate_rnea(workspace: str, *, require_benefit: bool = False) -> dict[str, Any]:
    source = resolve_within_root(workspace)
    if not source.is_file():
        return {"passed": False, "stage": "source", "details": [], "error": "Source file missing"}
    deps = inspect_dependencies()
    # Unique run directory: failed builds cannot accidentally reuse an older successful report.
    directory = ARTIFACT_ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    spec = model_contract()
    context: dict[str, Any] = {
        "schema_version": 1, "pack_version": "0.1.0", "source": relpath_for_display(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "model": spec,
        "dependencies": deps, "require_codegen_benefit": require_benefit,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "processor": platform.processor(), "toolchain": inspect_toolchain().to_dict()},
    }

    def finish(passed: bool, stage: str, error: str | None = None) -> dict[str, Any]:
        context.update(passed=passed, stage=stage, error=error)
        _save(directory / "validation.json", context)
        return {"passed": passed, "stage": stage, "error": error,
                "details": [f"diagnostics={relpath_for_display(directory / 'validation.json')}",
                            f"adoption={context.get('performance', {}).get('recommendation', 'not_measured')}"],
                "artifacts": {"directory": relpath_for_display(directory),
                              "validation": relpath_for_display(directory / "validation.json")},
                "metrics": context.get("correctness", {}),
                "performance": context.get("performance", {})}

    _save(directory / "samples.json", {"contract": spec, "inputs": samples()})
    if not deps["available"]:
        return finish(False, "capability", "Missing: " + ", ".join(deps["missing"]))
    try:
        build = configure_and_build(
            str(PACK_ROOT / "harness"), TARGET, timeout=300,
            cmake_definitions={
                "CMAKE_PREFIX_PATH": f"{PREFIX.as_posix()};{INSTALL_PREFIX.as_posix()}",
                "AUTODIFF_PREFIX": INSTALL_PREFIX.as_posix(),
                "WORKSPACE_HEADER": source.as_posix(), "SOURCE_SHA256": context["source_sha256"],
                "ROBOT_URDF": MODEL_PATH.as_posix(), "URDF_SHA256": spec["urdf_sha256"],
                "DEPENDENCY_FINGERPRINT": deps["fingerprint"],
            },
        )
        context["build"] = build.to_dict()
        if not build.passed or not build.build_dir:
            return finish(False, "build", build.error)
        inputs = directory / "samples.txt"
        inputs.write_text("\n".join(" ".join(format(x, ".17g") for x in point)
                                   for point in samples()) + "\n", encoding="ascii")
        raw_path = directory / "raw.json"
        execution = run_built_executable(build.build_dir, TARGET,
                                         args=(str(inputs), str(raw_path)), timeout=120)
        context["execution"] = execution
        build_directory = resolve_within_root(build.build_dir)
        context["generated_libraries"] = {
            relpath_for_display(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in build_directory.rglob("rnea_library*.so") if path.is_file()
        }
        if not execution["passed"]:
            return finish(False, "runtime", str(execution["stderr"]))
        if not raw_path.is_file():
            return finish(False, "report", "Executable produced no raw observations")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        correctness = evaluate_correctness(raw)
        performance = evaluate_performance(raw, correctness_passed=correctness["passed"])
        context.update(correctness=correctness, performance=performance)
        _save(directory / "correctness.json", correctness)
        _save(directory / "performance.json", performance)
        if correctness["first_failure"]:
            _save(directory / "failing-input.json", correctness["first_failure"])
        if not correctness["passed"]:
            return finish(False, "correctness")
        if require_benefit and not performance["measured_codegen_benefit"]:
            return finish(False, "performance", "Measured CodeGen benefit was not established")
        return finish(True, "complete")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return finish(False, "report_or_build_error", f"{type(exc).__name__}: {exc}")


def validate_codegen_benefit(workspace: str) -> dict[str, Any]:
    """Stricter optional gate for tasks that explicitly promise a CodeGen speed benefit."""
    return validate_rnea(workspace, require_benefit=True)
