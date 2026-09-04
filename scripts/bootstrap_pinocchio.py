"""Explicit Linux-only Pinocchio source bootstrap. Never installs OS packages."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess

from agent.domain.toolchain import inspect_toolchain
from agent.domain_packs.pinocchio_rnea.dependencies import (
    PINOCCHIO_COMMIT, PINOCCHIO_VERSION, PREFIX, ROOT, inspect_dependencies,
)
from agent.domain_packs.robotics_autodiff_codegen.dependencies import (
    INSTALL_PREFIX, inspect_dependencies as inspect_ad,
)


def bootstrap() -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("Pinocchio CodeGen requires Linux/WSL; --check is safe on Windows")
    ad = inspect_ad()
    if not ad["available"]:
        raise RuntimeError("Bootstrap the existing CppAD/CppADCodeGen toolchain first")
    if inspect_dependencies()["available"]:
        return inspect_dependencies()
    source = ROOT / "source"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", f"v{PINOCCHIO_VERSION}",
                        "https://github.com/stack-of-tasks/pinocchio.git", str(source)], check=True)
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != PINOCCHIO_COMMIT:
        raise RuntimeError(f"Pinocchio source mismatch: {commit}")
    # Only the build-system submodule, not the large example-robot-data collection.
    subprocess.run(["git", "-C", str(source), "submodule", "update", "--init", "--depth", "1",
                    "cmake"], check=True)
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError("Pinocchio source or submodule was modified; refusing pinned install")
    cmake = inspect_toolchain().cmake_path
    if not cmake:
        raise RuntimeError("CMake is required")
    subprocess.run([
        cmake, "-S", str(source), "-B", str(ROOT / "build"),
        f"-DCMAKE_INSTALL_PREFIX={PREFIX}", f"-DCMAKE_PREFIX_PATH={INSTALL_PREFIX}",
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_PYTHON_INTERFACE=OFF", "-DBUILD_TESTING=OFF",
        "-DBUILD_WITH_COLLISION_SUPPORT=OFF", "-DBUILD_WITH_URDF_SUPPORT=ON",
        # Header-only Scalar algorithms are instantiated in our harness. Building
        # Pinocchio's entire AD libraries is unnecessary (and much more expensive).
        "-DBUILD_WITH_CODEGEN_SUPPORT=OFF", "-DBUILD_WITH_AUTODIFF_SUPPORT=OFF",
        "-DENABLE_TEMPLATE_INSTANTIATION=OFF",
    ], check=True)
    subprocess.run([cmake, "--build", str(ROOT / "build"), "--target", "install",
                    "--parallel", "2"], check=True)
    marker = {"commit": PINOCCHIO_COMMIT, "version": PINOCCHIO_VERSION,
              "ad_fingerprint": ad["fingerprint"]}
    (ROOT / "installed.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    result = inspect_dependencies()
    if not result["available"]:
        raise RuntimeError(f"Incomplete Pinocchio installation: {result['missing']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Offline capability probe only")
    args = parser.parse_args()
    print(json.dumps(inspect_dependencies() if args.check else bootstrap(), indent=2))
