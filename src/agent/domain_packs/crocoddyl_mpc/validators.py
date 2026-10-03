"""Build and run the pinned Phase 6.4a Crocoddyl solver-chain profile."""

from __future__ import annotations

import json
import os
import platform
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.domain.toolchain import configure_and_build, inspect_toolchain, run_built_executable
from agent.domain_packs.crocoddyl_mpc.dependencies import PREFIX, inspect_dependencies
from agent.domain_packs.crocoddyl_mpc.report import evaluate_correctness, evaluate_profile
from agent.domain_packs.crocoddyl_mpc.robot_report import (
    evaluate_robot_correctness, evaluate_robot_profile,
)
from agent.domain_packs.crocoddyl_mpc.spec import (
    PACK_ROOT, PACK_VERSION, benchmark_contract, robot_contract,
)
from agent.domain_packs.pinocchio_rnea.spec import MODEL_PATH
from agent.domain_packs.pinocchio_rnea.dependencies import PREFIX as PINOCCHIO_PREFIX
from agent.domain_packs.robotics_autodiff_codegen.dependencies import INSTALL_PREFIX
from agent.tools._paths import PROJECT_ROOT, relpath_for_display


ARTIFACT_ROOT = PROJECT_ROOT / ".agent_state/domain_artifacts/crocoddyl_mpc"
TARGET = "crocoddyl_mpc_profile"


def _save(path: Path, data: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_crocoddyl_mpc(
    _: str = "", *, robot: bool = False, progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    dependencies = inspect_dependencies()
    directory = ARTIFACT_ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    context: dict[str, Any] = {
        "schema_version": 2 if robot else 1,
        "pack_version": PACK_VERSION,
        "contract": robot_contract() if robot else benchmark_contract(),
        "dependencies": dependencies,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "toolchain": inspect_toolchain().to_dict(),
        },
    }

    def finish(passed: bool, stage: str, error: str | None = None) -> dict[str, Any]:
        context.update(passed=passed, stage=stage, error=error)
        _save(directory / "validation.json", context)
        return {
            "passed": passed,
            "stage": stage,
            "error": error,
            "details": [
                f"diagnostics={relpath_for_display(directory / 'validation.json')}",
                f"recommendation={context.get('profile', {}).get('recommendation', 'not_measured')}",
            ],
            "artifacts": {
                "directory": relpath_for_display(directory),
                "validation": relpath_for_display(directory / "validation.json"),
            },
            "correctness": context.get("correctness", {}),
            "profile": context.get("profile", {}),
        }

    if not dependencies["available"]:
        return finish(False, "capability", "Missing: " + ", ".join(dependencies["missing"]))
    try:
        target = "crocoddyl_robot_mpc" if robot else TARGET
        if progress:
            progress(f"build {target}: configure <=300s, build <=900s")
        build = configure_and_build(
            str(PACK_ROOT / "harness"),
            target,
            timeout=300,
            build_timeout=900,
            cmake_definitions={
                "CMAKE_PREFIX_PATH": ";".join(
                    (PREFIX.as_posix(), PINOCCHIO_PREFIX.as_posix(), INSTALL_PREFIX.as_posix())
                ),
                "ROBOT_URDF": MODEL_PATH.as_posix(),
            },
        )
        context["build"] = build.to_dict()
        if not build.passed or not build.build_dir:
            return finish(False, "build", build.error)
        raw_path = directory / "raw.json"
        contract_path = directory / "contract.json"
        _save(contract_path, context["contract"])
        if progress:
            progress(f"run {target}: warmup + repeated measurements, timeout <=120s")
        execution = run_built_executable(
            build.build_dir, target,
            args=(str(raw_path), str(contract_path)) if robot else (str(raw_path),), timeout=120
        )
        context["execution"] = execution
        if not execution["passed"]:
            return finish(False, "runtime", str(execution["stderr"]))
        if not raw_path.is_file():
            return finish(False, "report", "Executable produced no raw profile")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        correctness = evaluate_robot_correctness(raw) if robot else evaluate_correctness(raw)
        profile = (evaluate_robot_profile if robot else evaluate_profile)(
            raw, correctness_passed=correctness["passed"]
        )
        context.update(correctness=correctness, profile=profile)
        _save(directory / "correctness.json", correctness)
        _save(directory / "profile.json", profile)
        if correctness["first_failure"]:
            _save(directory / "failing-evidence.json", correctness["first_failure"])
        if not correctness["passed"]:
            return finish(False, "correctness")
        if progress:
            progress(f"complete: {profile['recommendation']}")
        return finish(True, "complete")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return finish(False, "report_or_build_error", f"{type(exc).__name__}: {exc}")


def validate_robot_mpc(workspace: str = "") -> dict[str, Any]:
    return validate_crocoddyl_mpc(workspace, robot=True)
