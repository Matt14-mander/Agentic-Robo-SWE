"""Explicit Linux-only bootstrap for pinned Crocoddyl 3.0.1."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess

from agent.domain.toolchain import inspect_toolchain
from agent.domain_packs.crocoddyl_mpc.dependencies import PREFIX, ROOT, inspect_dependencies
from agent.domain_packs.crocoddyl_mpc.spec import CROCODDYL_COMMIT, CROCODDYL_VERSION
from agent.domain_packs.pinocchio_rnea.dependencies import PREFIX as PINOCCHIO_PREFIX
from agent.domain_packs.pinocchio_rnea.dependencies import inspect_dependencies as inspect_pinocchio
from agent.domain_packs.robotics_autodiff_codegen.dependencies import INSTALL_PREFIX


def bootstrap() -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("Crocoddyl profiling requires Linux/WSL; --check is safe on Windows")
    pinocchio = inspect_pinocchio()
    if not pinocchio["available"]:
        raise RuntimeError("Bootstrap the pinned CppAD/CppADCodeGen and Pinocchio stacks first")
    if inspect_dependencies()["available"]:
        return inspect_dependencies()
    source = ROOT / "source"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--branch", f"v{CROCODDYL_VERSION}",
                "https://github.com/loco-3d/crocoddyl.git", str(source),
            ],
            check=True,
        )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != CROCODDYL_COMMIT:
        raise RuntimeError(f"Crocoddyl source mismatch: {commit}")
    subprocess.run(
        ["git", "-C", str(source), "submodule", "update", "--init", "--depth", "1", "cmake"],
        check=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Crocoddyl source or submodule was modified; refusing pinned install")
    cmake = inspect_toolchain().cmake_path
    if not cmake:
        raise RuntimeError("CMake is required")
    prefix_path = f"{PINOCCHIO_PREFIX};{INSTALL_PREFIX}"
    subprocess.run(
        [
            cmake, "-S", str(source), "-B", str(ROOT / "build"),
            f"-DCMAKE_INSTALL_PREFIX={PREFIX}", f"-DCMAKE_PREFIX_PATH={prefix_path}",
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_PYTHON_INTERFACE=OFF",
            "-DBUILD_TESTING=OFF", "-DBUILD_BENCHMARK=OFF", "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_WITH_IPOPT=OFF", "-DBUILD_WITH_CODEGEN_SUPPORT=OFF",
            "-DBUILD_WITH_MULTITHREADS=OFF",
        ],
        check=True,
    )
    subprocess.run(
        [cmake, "--build", str(ROOT / "build"), "--target", "install", "--parallel", "2"],
        check=True,
    )
    marker = {
        "commit": CROCODDYL_COMMIT,
        "version": CROCODDYL_VERSION,
        "pinocchio_fingerprint": pinocchio["fingerprint"],
        "build_with_codegen_support": False,
    }
    (ROOT / "installed.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    result = inspect_dependencies()
    if not result["available"]:
        raise RuntimeError(f"Incomplete Crocoddyl installation: {result['missing']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Offline capability probe only")
    args = parser.parse_args()
    print(json.dumps(inspect_dependencies() if args.check else bootstrap(), indent=2))
