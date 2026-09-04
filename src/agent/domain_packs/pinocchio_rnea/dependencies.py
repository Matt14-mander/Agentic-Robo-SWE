"""Offline probe; Pinocchio installation is always explicit and project-local."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from agent.domain_packs.robotics_autodiff_codegen.dependencies import inspect_dependencies as inspect_ad
from agent.tools._paths import PROJECT_ROOT


PINOCCHIO_VERSION = "3.4.0"
PINOCCHIO_COMMIT = "187afafcfe22d7ac16a26241c0b13a76d04d82c1"
ROOT = PROJECT_ROOT / ".agent_state/domain_deps/pinocchio/v1"
PREFIX = ROOT / "install"


def inspect_dependencies() -> dict[str, Any]:
    ad = inspect_ad()
    missing = list(cast(list[str], ad["missing"]))
    marker = {}
    try:
        marker = json.loads((ROOT / "installed.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    headers = ("multibody/model.hpp", "parsers/urdf.hpp", "codegen/cppadcg.hpp")
    if (not isinstance(marker, dict) or marker.get("commit") != PINOCCHIO_COMMIT
            or marker.get("ad_fingerprint") != ad["fingerprint"]
            or not all((PREFIX / "include/pinocchio" / name).is_file() for name in headers)):
        missing.append("pinned_pinocchio")
    if missing:
        missing.append("pinocchio_rnea_runtime")
    identity = {"commit": PINOCCHIO_COMMIT, "ad": ad["fingerprint"], "installation": marker}
    return {"available": not missing, "missing": sorted(set(missing)),
            "pinocchio_version": PINOCCHIO_VERSION, "commit": PINOCCHIO_COMMIT,
            "fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "bootstrap_command": "uv run python scripts/bootstrap_pinocchio.py",
            "platform": ad["platform"]}
