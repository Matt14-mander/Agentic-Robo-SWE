"""Restricted, reproducible CMake execution for local Domain Packs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root


_ARTIFACT_ROOT = PROJECT_ROOT / ".agent_state" / "domain_artifacts"
_TARGET_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_BUILD_TYPES = {"Debug", "Release", "RelWithDebInfo", "MinSizeRel"}
_MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True)
class ToolchainInfo:
    available: bool
    platform: str
    architecture: str
    cmake_path: str | None
    cmake_version: str | None
    compiler_path: str | None
    compiler_version: str | None
    ninja_path: str | None
    missing: tuple[str, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BuildResult:
    passed: bool
    source_dir: str
    build_dir: str | None
    target: str
    fingerprint: str
    cache_hit: bool
    configure_seconds: float
    build_seconds: float
    stdout: str
    stderr: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_toolchain() -> ToolchainInfo:
    cmake = _find_cmake()
    ninja = _find_ninja()
    compiler = _first_executable(("c++", "g++", "clang++", "cl"))
    cmake_version = _version(cmake, ("--version",)) if cmake else None
    compiler_args = ("--version",) if compiler and Path(compiler).stem.lower() != "cl" else ()
    compiler_version = _version(compiler, compiler_args) if compiler else None
    missing = tuple(
        name for name, value in (("cmake", cmake), ("cxx_compiler", compiler)) if not value
    )
    identity = json.dumps(
        {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cmake": cmake,
            "cmake_version": cmake_version,
            "compiler": compiler,
            "compiler_version": compiler_version,
            "ninja": ninja,
        },
        sort_keys=True,
    )
    return ToolchainInfo(
        available=not missing,
        platform=platform.system(),
        architecture=platform.machine(),
        cmake_path=cmake,
        cmake_version=cmake_version,
        compiler_path=compiler,
        compiler_version=compiler_version,
        ninja_path=ninja,
        missing=missing,
        fingerprint=hashlib.sha256(identity.encode()).hexdigest(),
    )


def configure_and_build(
    source_dir: str,
    target: str,
    *,
    build_type: str = "Release",
    timeout: int = 120,
    cmake_definitions: dict[str, str] | None = None,
) -> BuildResult:
    if not _TARGET_RE.fullmatch(target):
        raise ValueError(f"Invalid CMake target: {target!r}")
    if build_type not in _BUILD_TYPES:
        raise ValueError(f"Unsupported build_type: {build_type!r}")
    timeout = min(max(int(timeout), 1), 300)
    source = resolve_within_root(source_dir)
    if not source.is_dir() or not (source / "CMakeLists.txt").is_file():
        raise ValueError(f"CMake source directory is invalid: {source_dir}")
    definitions = dict(cmake_definitions or {})
    for key, value in definitions.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid CMake definition name: {key!r}")
        if any(character in value for character in ("\0", "\n", "\r")):
            raise ValueError(f"Invalid CMake definition value for {key}")

    toolchain = inspect_toolchain()
    source_fingerprint = _source_fingerprint(source)
    build_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "source_dir": str(source),
                "source": source_fingerprint,
                "toolchain": toolchain.fingerprint,
                "target": target,
                "build_type": build_type,
                "definitions": definitions,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if not toolchain.available or not toolchain.cmake_path:
        return BuildResult(
            passed=False,
            source_dir=relpath_for_display(source),
            build_dir=None,
            target=target,
            fingerprint=build_fingerprint,
            cache_hit=False,
            configure_seconds=0.0,
            build_seconds=0.0,
            stdout="",
            stderr="",
            error=f"Missing C++ toolchain capabilities: {', '.join(toolchain.missing)}",
        )

    build_dir = _ARTIFACT_ROOT / "cmake" / build_fingerprint[:20]
    marker = build_dir / ".build-complete.json"
    if marker.is_file():
        return BuildResult(
            passed=True,
            source_dir=relpath_for_display(source),
            build_dir=relpath_for_display(build_dir),
            target=target,
            fingerprint=build_fingerprint,
            cache_hit=True,
            configure_seconds=0.0,
            build_seconds=0.0,
            stdout="cached successful build",
            stderr="",
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    configure_command = [
        toolchain.cmake_path,
        "-S",
        str(source),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    if toolchain.ninja_path:
        configure_command.extend(("-G", "Ninja"))
    configure_command.extend(f"-D{key}={value}" for key, value in sorted(definitions.items()))
    env = _build_environment()
    if toolchain.compiler_path and Path(toolchain.compiler_path).stem.lower() != "cl":
        env["CXX"] = toolchain.compiler_path
    if toolchain.ninja_path:
        env["PATH"] = str(Path(toolchain.ninja_path).parent) + os.pathsep + env.get("PATH", "")
    configured = _run(configure_command, cwd=source, env=env, timeout=timeout)
    if configured[0] != 0:
        return _result_from_failure(
            source,
            build_dir,
            target,
            build_fingerprint,
            configured,
            stage="configure",
        )
    built = _run(
        [toolchain.cmake_path, "--build", str(build_dir), "--target", target, "--config", build_type],
        cwd=source,
        env=env,
        timeout=timeout,
    )
    stdout = _truncate(configured[1] + built[1])
    stderr = _truncate(configured[2] + built[2])
    passed = built[0] == 0
    result = BuildResult(
        passed=passed,
        source_dir=relpath_for_display(source),
        build_dir=relpath_for_display(build_dir),
        target=target,
        fingerprint=build_fingerprint,
        cache_hit=False,
        configure_seconds=configured[3],
        build_seconds=built[3],
        stdout=stdout,
        stderr=stderr,
        error=None if passed else f"CMake build failed with exit code {built[0]}",
    )
    (build_dir / "configure.log").write_text(configured[1] + configured[2], encoding="utf-8")
    (build_dir / "build.log").write_text(built[1] + built[2], encoding="utf-8")
    if passed:
        marker.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def find_built_executable(build_dir: str, target: str) -> Path:
    if not _TARGET_RE.fullmatch(target):
        raise ValueError(f"Invalid executable target: {target!r}")
    resolved = resolve_within_root(build_dir)
    try:
        resolved.relative_to(_ARTIFACT_ROOT)
    except ValueError as exc:
        raise ValueError("Build directory must be under .agent_state/domain_artifacts") from exc
    candidates = [path for name in (target, f"{target}.exe") for path in resolved.rglob(name)]
    files = [path for path in candidates if path.is_file()]
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one built executable for {target}, found {len(files)}")
    return files[0]


def run_built_executable(
    build_dir: str,
    target: str,
    *,
    args: tuple[str, ...] = (),
    timeout: int = 30,
) -> dict[str, object]:
    executable = find_built_executable(build_dir, target)
    if any("\0" in argument for argument in args):
        raise ValueError("Executable arguments may not contain NUL bytes")
    completed = _run(
        [str(executable), *args],
        cwd=executable.parent,
        env=_build_environment(),
        timeout=min(max(int(timeout), 1), 120),
    )
    return {
        "passed": completed[0] == 0,
        "exit_code": completed[0],
        "stdout": _truncate(completed[1]),
        "stderr": _truncate(completed[2]),
        "duration_seconds": completed[3],
        "executable": relpath_for_display(executable),
    }


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return process.returncode, process.stdout, process.stderr, time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return 124, stdout, stderr + f"\nTimed out after {timeout}s", time.perf_counter() - started
    except OSError as exc:
        return 127, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def _result_from_failure(
    source: Path,
    build_dir: Path,
    target: str,
    fingerprint: str,
    output: tuple[int, str, str, float],
    *,
    stage: str,
) -> BuildResult:
    (build_dir / f"{stage}.log").write_text(output[1] + output[2], encoding="utf-8")
    return BuildResult(
        passed=False,
        source_dir=relpath_for_display(source),
        build_dir=relpath_for_display(build_dir),
        target=target,
        fingerprint=fingerprint,
        cache_hit=False,
        configure_seconds=output[3] if stage == "configure" else 0.0,
        build_seconds=output[3] if stage == "build" else 0.0,
        stdout=_truncate(output[1]),
        stderr=_truncate(output[2]),
        error=f"CMake {stage} failed with exit code {output[0]}",
    )


def _source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    allowed_suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cmake", ".txt"}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or (path.name != "CMakeLists.txt" and path.suffix.lower() not in allowed_suffixes):
            continue
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_environment() -> dict[str, str]:
    allowed = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE")
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env["LC_ALL"] = "C"
    return env


def _first_executable(names: tuple[str, ...]) -> str | None:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def _find_cmake() -> str | None:
    discovered = shutil.which("cmake")
    if discovered:
        return discovered
    return _find_visual_studio_cmake_tool("CMake/CMake/bin/cmake.exe")


def _find_ninja() -> str | None:
    discovered = shutil.which("ninja")
    if discovered:
        return discovered
    return _find_visual_studio_cmake_tool("CMake/Ninja/ninja.exe")


def _find_visual_studio_cmake_tool(relative: str) -> str | None:
    roots = [
        Path(value) / "Microsoft Visual Studio"
        for key in ("ProgramFiles(x86)", "ProgramFiles")
        if (value := os.getenv(key))
    ]
    for root in roots:
        for edition in sorted(root.glob("*/*"), reverse=True):
            candidate = edition / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / relative
            if candidate.is_file():
                return str(candidate)
    return None


def _version(executable: str, args: tuple[str, ...]) -> str | None:
    result = _run([executable, *args], cwd=PROJECT_ROOT, env=_build_environment(), timeout=10)
    text = (result[1] or result[2]).strip().splitlines()
    return text[0][:300] if text else None


def _truncate(value: str) -> str:
    if len(value) <= _MAX_OUTPUT_CHARS:
        return value
    return value[:_MAX_OUTPUT_CHARS] + f"\n... [{len(value) - _MAX_OUTPUT_CHARS} chars truncated]"
