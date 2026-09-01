"""Offline dependency probe for the explicitly bootstrapped AD toolchain."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

from agent.domain.toolchain import inspect_toolchain
from agent.domain_packs.robotics_autodiff_codegen.constants import (
    CPPAD_CODEGEN_COMMIT,
    CPPAD_COMMIT,
    DEPENDENCY_LAYOUT_VERSION,
)
from agent.tools._paths import PROJECT_ROOT, relpath_for_display


DEPENDENCY_ROOT = (
    PROJECT_ROOT / ".agent_state" / "domain_deps" / "autodiff_codegen"
    / DEPENDENCY_LAYOUT_VERSION
)
INSTALL_PREFIX = DEPENDENCY_ROOT / "install"
LOCK_FILE = Path(__file__).resolve().parent / "dependencies.lock.json"


def inspect_dependencies() -> dict[str, object]:
    toolchain = inspect_toolchain()
    required_headers = (
        INSTALL_PREFIX / "include" / "cppad" / "cppad.hpp",
        INSTALL_PREFIX / "include" / "cppad" / "cg.hpp",
        INSTALL_PREFIX / "include" / "cppad" / "configure.hpp",
        INSTALL_PREFIX / "include" / "cppad" / "cg" / "configure.hpp",
    )
    marker = DEPENDENCY_ROOT / "installed.json"
    missing = list(toolchain.missing)
    if platform.system() != "Linux":
        missing.append("linux_codegen_runtime")
    missing.extend(
        f"header:{path.relative_to(INSTALL_PREFIX).as_posix()}"
        for path in required_headers if not path.is_file()
    )
    marker_data: dict[str, object] = {}
    if marker.is_file():
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            missing.append("valid_install_marker")
    else:
        missing.append("install_marker")
    if marker_data.get("cppad_commit") != CPPAD_COMMIT:
        missing.append("pinned_cppad_commit")
    if marker_data.get("cppad_codegen_commit") != CPPAD_CODEGEN_COMMIT:
        missing.append("pinned_cppad_codegen_commit")
    identity = {
        "toolchain": toolchain.fingerprint,
        "platform": platform.system(),
        "cppad_commit": CPPAD_COMMIT,
        "cppad_codegen_commit": CPPAD_CODEGEN_COMMIT,
        "lock_sha256": hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest(),
    }
    return {
        "available": not missing,
        "platform": platform.system(),
        "install_prefix": relpath_for_display(INSTALL_PREFIX),
        "missing": sorted(set(missing)),
        "cppad_commit": CPPAD_COMMIT,
        "cppad_codegen_commit": CPPAD_CODEGEN_COMMIT,
        "fingerprint": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "bootstrap_command": "uv run python scripts/bootstrap_autodiff_codegen.py",
    }
