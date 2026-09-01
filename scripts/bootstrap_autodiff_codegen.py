"""Explicitly install the pinned Phase 6.1 CppAD toolchain into .agent_state."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from agent.domain.toolchain import inspect_toolchain
from agent.domain_packs.robotics_autodiff_codegen.constants import (
    CPPAD_CODEGEN_COMMIT,
    CPPAD_CODEGEN_TAG,
    CPPAD_COMMIT,
    CPPAD_TAG,
)
from agent.domain_packs.robotics_autodiff_codegen.dependencies import (
    DEPENDENCY_ROOT,
    INSTALL_PREFIX,
    inspect_dependencies,
)


REPOSITORIES = (
    ("CppAD", "https://github.com/coin-or/CppAD.git", CPPAD_TAG, CPPAD_COMMIT),
    (
        "CppADCodeGen",
        "https://github.com/joaoleal/CppADCodeGen.git",
        CPPAD_CODEGEN_TAG,
        CPPAD_CODEGEN_COMMIT,
    ),
)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _checkout(name: str, repository: str, tag: str, commit: str) -> Path:
    source = DEPENDENCY_ROOT / "sources" / name
    if not (source / ".git").is_dir():
        source.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", tag, "--depth", "1", repository, str(source)])
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"{name} commit mismatch: expected {commit}, found {actual}")
    return source


def bootstrap(*, force: bool = False) -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError(
            "CppADCodeGen runtime compilation is supported only on Linux; "
            "run this bootstrap in WSL/Linux or rely on the autodiff-codegen CI job"
        )
    if force and DEPENDENCY_ROOT.is_dir():
        shutil.rmtree(DEPENDENCY_ROOT)
    toolchain = inspect_toolchain()
    if not toolchain.available or not toolchain.cmake_path:
        raise RuntimeError(f"Missing C++ toolchain: {', '.join(toolchain.missing)}")
    if not shutil.which("git"):
        raise RuntimeError("git is required for the explicit dependency bootstrap")

    sources = {
        name: _checkout(name, repository, tag, commit)
        for name, repository, tag, commit in REPOSITORIES
    }
    cmake = toolchain.cmake_path
    cppad_build = DEPENDENCY_ROOT / "build" / "cppad"
    codegen_build = DEPENDENCY_ROOT / "build" / "cppad_codegen"
    common = ["-DCMAKE_BUILD_TYPE=Release"]
    if toolchain.compiler_path:
        common.append(f"-DCMAKE_CXX_COMPILER={toolchain.compiler_path}")
        c_compiler = _matching_c_compiler(Path(toolchain.compiler_path))
        if c_compiler:
            common.append(f"-DCMAKE_C_COMPILER={c_compiler}")
    if toolchain.ninja_path:
        common.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={toolchain.ninja_path}"])
    _run([
        cmake, "-S", str(sources["CppAD"]), "-B", str(cppad_build),
        f"-Dcppad_prefix={INSTALL_PREFIX}", "-Dcppad_testvector=std", *common,
    ])
    _run([cmake, "--build", str(cppad_build), "--target", "install", "--config", "Release"])
    _run([
        cmake, "-S", str(sources["CppADCodeGen"]), "-B", str(codegen_build),
        f"-DCMAKE_INSTALL_PREFIX={INSTALL_PREFIX}", f"-DCMAKE_PREFIX_PATH={INSTALL_PREFIX}",
        "-DENABLE_THREAD_POOL_TESTS=OFF", *common,
    ])
    _run([
        cmake, "--build", str(codegen_build), "--target", "install", "--config", "Release",
    ])
    marker = {
        "schema_version": 1,
        "cppad_commit": CPPAD_COMMIT,
        "cppad_codegen_commit": CPPAD_CODEGEN_COMMIT,
        "toolchain_fingerprint": toolchain.fingerprint,
    }
    DEPENDENCY_ROOT.mkdir(parents=True, exist_ok=True)
    (DEPENDENCY_ROOT / "installed.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    return inspect_dependencies()


def _matching_c_compiler(cxx_compiler: Path) -> str | None:
    names = {
        "g++": "gcc",
        "g++.exe": "gcc.exe",
        "c++": "cc",
        "c++.exe": "cc.exe",
        "clang++": "clang",
        "clang++.exe": "clang.exe",
        "cl.exe": "cl.exe",
        "cl": "cl",
    }
    sibling_name = names.get(cxx_compiler.name.lower())
    if sibling_name and (sibling := cxx_compiler.with_name(sibling_name)).is_file():
        return str(sibling)
    return shutil.which("cc") or shutil.which("gcc")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="probe without network or writes")
    parser.add_argument("--force", action="store_true", help="replace the managed dependency cache")
    args = parser.parse_args()
    result = inspect_dependencies() if args.check else bootstrap(force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["available"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"bootstrap failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
