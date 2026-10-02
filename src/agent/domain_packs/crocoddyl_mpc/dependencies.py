"""Offline Crocoddyl capability probe; installation is always explicit."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from agent.domain_packs.pinocchio_rnea.dependencies import inspect_dependencies as inspect_pinocchio
from agent.domain_packs.crocoddyl_mpc.spec import CROCODDYL_COMMIT, CROCODDYL_VERSION
from agent.tools._paths import PROJECT_ROOT


ROOT = PROJECT_ROOT / ".agent_state/domain_deps/crocoddyl_mpc/v1"
PREFIX = ROOT / "install"


def inspect_dependencies() -> dict[str, Any]:
    pinocchio = inspect_pinocchio()
    missing = list(cast(list[str], pinocchio["missing"]))
    marker: dict[str, Any] = {}
    try:
        loaded = json.loads((ROOT / "installed.json").read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            marker = loaded
    except (OSError, ValueError):
        pass
    headers = (
        "crocoddyl/core/actions/lqr.hpp",
        "crocoddyl/core/numdiff/action.hpp",
        "crocoddyl/core/solvers/fddp.hpp",
    )
    libraries = tuple(PREFIX.glob("lib/libcrocoddyl.*"))
    if (
        marker.get("commit") != CROCODDYL_COMMIT
        or not all((PREFIX / "include" / name).is_file() for name in headers)
        or not any(path.is_file() for path in libraries)
    ):
        missing.append("pinned_crocoddyl")
    if missing:
        missing.append("crocoddyl_mpc_runtime")
    identity = {
        "commit": CROCODDYL_COMMIT,
        "pinocchio": pinocchio["fingerprint"],
        "installation": marker,
    }
    return {
        "available": not missing,
        "missing": sorted(set(missing)),
        "version": CROCODDYL_VERSION,
        "commit": CROCODDYL_COMMIT,
        "fingerprint": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "bootstrap_command": "uv run python scripts/bootstrap_crocoddyl.py",
        "platform": pinocchio["platform"],
    }
