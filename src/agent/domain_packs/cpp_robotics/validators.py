"""Build source headers, preserve evidence and independently judge numeric observations."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any

from agent.domain.toolchain import configure_and_build, inspect_toolchain, run_built_executable
from agent.domain_packs.cpp_robotics.spec import CASE_NAMES, SEED, VERSION, expected, samples
from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root

PACK_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT_ROOT / ".agent_state/domain_artifacts/cpp_robotics"
TARGET = "robotics_cpp_validator"


def judge_observations(scenario: str, text: str) -> dict[str, Any]:
    inputs = samples(scenario)
    rows = text.splitlines()
    if len(rows) != len(inputs):
        raise ValueError("Incomplete or extra observation rows")
    failures = []
    max_error = 0.
    for index, (x, row) in enumerate(zip(inputs, rows, strict=True)):
        tokens = row.split()
        reference = expected(scenario, x)
        if not tokens or int(tokens[0]) != len(reference) or len(tokens) != len(reference)+1:
            raise ValueError(f"Invalid observation shape at sample {index}")
        actual = [float(t) for t in tokens[1:]]
        for component, (a, b) in enumerate(zip(actual, reference, strict=True)):
            error = abs(a-b)
            if not math.isfinite(a) or error > 1e-9 + 1e-8*abs(b):
                failures.append({"sample": index, "input": x, "component": component,
                                 "actual": a if math.isfinite(a) else str(a), "expected": b})
            if math.isfinite(a):
                max_error = max(max_error, error)
    return {"passed": not failures, "sample_count": len(inputs), "failures": failures,
            "first_failure": failures[0] if failures else None, "max_abs_error": max_error}


def validate_robotics_cpp(workspace: str, *, scenario: str) -> dict[str, object]:
    if scenario not in CASE_NAMES:
        raise ValueError(f"Unknown robotics scenario: {scenario}")
    directory = ARTIFACT_ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    context: dict[str, Any] = {"schema_version": 1, "pack_version": VERSION,
        "scenario": scenario, "seed": SEED + CASE_NAMES.index(scenario),
        "provenance": "synthetic_controlled", "sample_count": 28}

    def finish(passed: bool, stage: str, error: str | None = None) -> dict[str, object]:
        context.update(passed=passed, stage=stage, error=error)
        path = directory / "validation.json"
        path.write_text(json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return {"passed": passed, "stage": stage, "error": error,
                "details": [f"diagnostics={relpath_for_display(path)}",
                            f"scenario={scenario}; stage={stage}",
                            json.dumps(context.get("correctness", {}).get("first_failure"), ensure_ascii=False)],
                "artifacts": {"validation": relpath_for_display(path)}}
    try:
        source = resolve_within_root(workspace)
        if not source.is_file():
            return finish(False, "source", "Workspace header not found")
        toolchain = inspect_toolchain()
        context["toolchain"] = toolchain.to_dict()
        if not toolchain.available:
            return finish(False, "capability", "Missing: " + ", ".join(toolchain.missing))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        context["source_sha256"] = digest
        context["oracle_sha256"] = hashlib.sha256((PACK_ROOT / "spec.py").read_bytes()).hexdigest()
        build = configure_and_build(str(PACK_ROOT / "harness"), TARGET, timeout=120,
            cmake_definitions={"WORKSPACE_HEADER": source.as_posix(), "WORKSPACE_SHA256": digest,
                               "ROBOTICS_CASE": str(CASE_NAMES.index(scenario)), "PACK_VERSION": VERSION})
        context["build"] = build.to_dict()
        if not build.passed or not build.build_dir:
            return finish(False, "build", build.error or "C++ build failed")
        inputs = samples(scenario)
        input_path, output_path = directory / "inputs.txt", directory / "observations.txt"
        input_path.write_text(str(len(inputs)) + "\n" + "\n".join(
            str(len(x)) + " " + " ".join(format(v, ".17g") for v in x) for x in inputs), encoding="utf-8")
        execution = run_built_executable(build.build_dir, TARGET,
            args=(str(input_path), str(output_path)), timeout=30)
        context["execution"] = execution
        if not execution["passed"]:
            return finish(False, "runtime", str(execution.get("stderr") or "Native execution failed"))
        context["correctness"] = judge_observations(scenario, output_path.read_text(encoding="utf-8"))
        return finish(context["correctness"]["passed"], "correctness")
    except (OSError, ValueError, TypeError) as exc:
        return finish(False, "report_or_build_error", f"{type(exc).__name__}: {exc}")
