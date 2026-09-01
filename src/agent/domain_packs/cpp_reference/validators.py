"""Official validator for the Phase 6.0 minimal C++ reference task."""

from __future__ import annotations

import hashlib
from pathlib import Path

from agent.domain.toolchain import configure_and_build, inspect_toolchain, run_built_executable
from agent.domain_packs.cpp_reference.constants import PACK_VERSION
from agent.tools._paths import resolve_within_root


_HARNESS = Path(__file__).resolve().parent / "harness"
_TARGET = "cpp_reference_validator"


def validate_cpp_reference(workspace: str) -> dict[str, object]:
    try:
        source = resolve_within_root(workspace)
    except ValueError as exc:
        return {"passed": False, "details": [], "error": str(exc)}
    if not source.is_file():
        return {"passed": False, "details": [], "error": f"Workspace not found: {source}"}
    toolchain = inspect_toolchain()
    if not toolchain.available:
        return {
            "passed": False,
            "details": [],
            "error": f"Missing C++ capabilities: {', '.join(toolchain.missing)}",
        }
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    build = configure_and_build(
        str(_HARNESS),
        _TARGET,
        timeout=120,
        cmake_definitions={
            "WORKSPACE_HEADER": source.as_posix(),
            "WORKSPACE_SHA256": digest,
            "DOMAIN_PACK_VERSION": PACK_VERSION,
        },
    )
    if not build.passed or not build.build_dir:
        return {
            "passed": False,
            "details": [build.stderr or build.stdout],
            "error": build.error,
        }
    execution = run_built_executable(build.build_dir, _TARGET, timeout=30)
    if not execution["passed"]:
        details = [str(execution["stdout"] or execution["stderr"])]
        return {"passed": False, "details": details, "error": None}
    return {
        "passed": True,
        "details": [
            f"C++ reference output verified; toolchain={toolchain.fingerprint[:12]}; "
            f"build_cache_hit={build.cache_hit}"
        ],
        "error": None,
    }
