"""Restricted build tools exposed by the Phase 6.0 reference Domain Pack."""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from agent.domain.toolchain import configure_and_build, inspect_toolchain, run_built_executable
from agent.domain_packs.cpp_reference.constants import PACK_VERSION


def _local_only_error() -> str | None:
    backend = (os.getenv("EXECUTOR_BACKEND") or "local").strip().lower()
    if backend == "local":
        return None
    return json.dumps({
        "passed": False,
        "error": f"cpp_reference is local-only; EXECUTOR_BACKEND={backend!r}",
    })


@tool
def inspect_cpp_toolchain() -> str:
    """Inspect the local CMake/C++ toolchain and return a reproducible JSON fingerprint."""
    if error := _local_only_error():
        return error
    return json.dumps(inspect_toolchain().to_dict(), ensure_ascii=False, indent=2)


@tool
def build_cpp_project(
    source_dir: str,
    target: str,
    build_type: str = "Release",
    timeout: int = 120,
) -> str:
    """Configure and build one allow-listed CMake target under the project artifact directory."""
    if error := _local_only_error():
        return error
    try:
        result = configure_and_build(
            source_dir,
            target,
            build_type=build_type,
            timeout=timeout,
            cmake_definitions={"DOMAIN_PACK_VERSION": PACK_VERSION},
        )
    except (OSError, ValueError) as exc:
        return json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


@tool
def run_cpp_target(
    build_dir: str,
    target: str,
    args: list[str] | None = None,
    timeout: int = 30,
) -> str:
    """Run one executable previously built inside the managed Domain Pack artifact directory."""
    if error := _local_only_error():
        return error
    try:
        result = run_built_executable(
            build_dir,
            target,
            args=tuple(args or ()),
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(result, ensure_ascii=False, indent=2)


CPP_REFERENCE_TOOLS = (inspect_cpp_toolchain, build_cpp_project, run_cpp_target)
