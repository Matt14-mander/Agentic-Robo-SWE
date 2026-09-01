"""Official source-to-CodeGen numerical gate for the Phase 6.1 MVP."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

from agent.domain.toolchain import configure_and_build, run_built_executable
from agent.domain_packs.robotics_autodiff_codegen.compatibility import inspect_source
from agent.domain_packs.robotics_autodiff_codegen.constants import (
    JACOBIAN_RELATIVE_TOLERANCE,
    OUTPUT_TOLERANCE,
    PACK_VERSION,
)
from agent.domain_packs.robotics_autodiff_codegen.dependencies import (
    INSTALL_PREFIX,
    inspect_dependencies,
)
from agent.tools._paths import resolve_within_root


_HARNESS = Path(__file__).resolve().parent / "harness"
_TARGET = "autodiff_codegen_validator"


def validate_autodiff_codegen(workspace: str) -> dict[str, object]:
    try:
        source = resolve_within_root(workspace)
    except ValueError as exc:
        return _failure(str(exc))
    if not source.is_file():
        return _failure(f"Workspace not found: {source}")

    compatibility = inspect_source(source)
    if not compatibility["passed"]:
        findings = cast(list[dict[str, Any]], compatibility["findings"])
        details = [
            f"{item['rule']} line {item['line']}: {item['message']}"
            for item in findings
            if item["severity"] == "error"
        ]
        return {"passed": False, "details": details, "error": None}

    dependencies = inspect_dependencies()
    if not dependencies["available"]:
        missing = cast(list[str], dependencies["missing"])
        return _failure(
            "Missing pinned autodiff capabilities: "
            + ", ".join(missing)
            + f". Bootstrap explicitly with: {dependencies['bootstrap_command']}"
        )

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    build = configure_and_build(
        str(_HARNESS),
        _TARGET,
        timeout=240,
        cmake_definitions={
            "WORKSPACE_HEADER": source.as_posix(),
            "WORKSPACE_SHA256": source_digest,
            "AUTODIFF_PREFIX": INSTALL_PREFIX.as_posix(),
            "AUTODIFF_DEPENDENCY_FINGERPRINT": str(dependencies["fingerprint"]),
            "DOMAIN_PACK_VERSION": PACK_VERSION,
            "OUTPUT_TOLERANCE": format(OUTPUT_TOLERANCE, ".17g"),
            "JACOBIAN_RELATIVE_TOLERANCE": format(JACOBIAN_RELATIVE_TOLERANCE, ".17g"),
        },
    )
    if not build.passed or not build.build_dir:
        return {
            "passed": False,
            "details": [build.stderr or build.stdout],
            "error": build.error,
        }
    execution = run_built_executable(build.build_dir, _TARGET, timeout=120)
    output = str(execution["stdout"] or execution["stderr"]).strip()
    if not execution["passed"]:
        return {"passed": False, "details": [output], "error": None}
    return {
        "passed": True,
        "details": [
            output,
            f"dependency={str(dependencies['fingerprint'])[:12]}; "
            f"build_cache_hit={build.cache_hit}; seeds=3",
        ],
        "error": None,
    }


def _failure(error: str) -> dict[str, object]:
    return {"passed": False, "details": [], "error": error}
