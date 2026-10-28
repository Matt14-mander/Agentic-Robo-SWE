"""Public eight-case C++ adapter. No LLM, shell input, or hidden-evaluation claims."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from agent.containers import run_bounded
from agent.domain_packs.cpp_robotics.spec import CASE_NAMES, samples
from agent.domain_packs.cpp_robotics.validators import PACK_ROOT, judge_observations
from agent.failures import with_failure


def compiler_identity(compiler: str) -> dict[str, Any]:
    path = shutil.which(compiler)
    if not path:
        return with_failure({"passed": False, "stage": "capability", "error": "C++ compiler missing"})
    result = run_bounded([path, "--version"], timeout=15)
    return with_failure({"passed": result.returncode == 0, "stage": "capability",
                         "compiler": str(Path(path).resolve()), "version": result.stdout[:4000],
                         "error": result.stderr if result.returncode else None})


def compile_candidate(source: Path, scenario: str, artifacts: Path, compiler: str) -> dict[str, Any]:
    if scenario not in CASE_NAMES:
        raise ValueError("Unsupported native calibration scenario")
    directory = artifacts / uuid.uuid4().hex
    directory.mkdir(parents=True)
    executable = directory / "candidate.exe"
    try:
        # Native argv parsers on Windows may not support Unicode absolute paths.
        # Stage the exact bytes and use ASCII filenames under a Unicode-safe cwd.
        shutil.copyfile(source, directory / "candidate.hpp")
        shutil.copyfile(PACK_ROOT / "harness/main.cpp", directory / "probe.cpp")
        result = run_bounded([
            compiler, "-std=c++17", "-O0", "probe.cpp", '-DWORKSPACE_HEADER="candidate.hpp"',
            f"-DROBOTICS_CASE={CASE_NAMES.index(scenario)}", "-o", "candidate.exe",
        ], cwd=directory, timeout=120)
        payload = {"passed": result.returncode == 0, "stage": "build",
                   "exit_code": result.returncode, "stdout": result.stdout[:12000],
                   "stderr": result.stderr[:12000], "timed_out": result.returncode == 124,
                   "error": None if result.returncode == 0 else "C++ compilation failed"}
    except OSError as exc:
        payload = {"passed": False, "stage": "capability", "error": type(exc).__name__}
    payload = with_failure(payload)
    (directory / "build.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "executable": str(executable), "artifact_dir": str(directory)}


def evaluate_candidate(source: Path, scenario: str, artifacts: Path, compiler: str) -> dict[str, Any]:
    built = compile_candidate(source, scenario, artifacts, compiler)
    if not built["passed"]:
        return built
    directory = Path(built["artifact_dir"])
    inputs = samples(scenario)
    input_path, output_path = directory / "inputs.txt", directory / "observations.txt"
    input_path.write_text(str(len(inputs)) + "\n" + "\n".join(
        str(len(row)) + " " + " ".join(format(v, ".17g") for v in row) for row in inputs), encoding="utf-8")
    try:
        result = run_bounded([built["executable"], "inputs.txt", "observations.txt"],
                             cwd=directory, timeout=30)
        if result.returncode:
            payload = {"passed": False, "stage": "runtime", "error": result.stderr[:12000],
                       "timed_out": result.returncode == 124}
        else:
            payload = {**judge_observations(scenario, output_path.read_text(encoding="utf-8")),
                       "stage": "correctness", "error": None}
    except (OSError, ValueError) as exc:
        payload = {"passed": False, "stage": "protocol", "error": str(exc)[:1000]}
    payload = with_failure(payload)
    (directory / "validation.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
