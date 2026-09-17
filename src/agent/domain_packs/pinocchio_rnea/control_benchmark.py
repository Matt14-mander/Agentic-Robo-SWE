"""Fresh-process orchestration for the Phase 6.3b closed-loop benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from agent.domain.toolchain import run_built_executable
from agent.domain_packs.pinocchio_rnea.report import aggregate_control_processes
from agent.domain_packs.pinocchio_rnea.spec import PACK_ROOT
from agent.domain_packs.pinocchio_rnea.validators import TARGET, validate_rnea
from agent.tools._paths import relpath_for_display, resolve_within_root


def _save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def run_control_loop_benchmark(processes: int = 3) -> dict[str, Any]:
    if not 3 <= processes <= 9:
        raise ValueError("processes must be between 3 and 9")
    seed = validate_rnea(str(PACK_ROOT / "harness/reference.hpp"))
    if not seed["passed"]:
        return {"passed": False, "stage": "seed_validation", "seed": seed}
    directory = resolve_within_root(seed["artifacts"]["directory"])
    envelope = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    raw_seed = json.loads((directory / "raw.json").read_text(encoding="utf-8"))
    build_dir = envelope["build"]["build_dir"]
    sample_file = directory / "samples.txt"
    libraries = sorted(resolve_within_root(build_dir).rglob("rnea_library*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one reusable generated library, found {len(libraries)}")
    library_hash = hashlib.sha256(libraries[0].read_bytes()).hexdigest()
    reports = []
    executions = []
    for process_index in range(processes):
        path = directory / f"control-process-{process_index + 1:02d}.json"
        execution = run_built_executable(
            build_dir, TARGET,
            args=(str(sample_file), str(path), "--reuse-library"), timeout=120,
        )
        executions.append(execution)
        if not execution["passed"] or not path.is_file():
            result = {"passed": False, "stage": "fresh_process_runtime",
                      "failed_process": process_index + 1, "execution": execution,
                      "seed": seed}
            _save(directory / "control-loop-aggregate.json", result)
            return result
        if hashlib.sha256(libraries[0].read_bytes()).hexdigest() != library_hash:
            raise RuntimeError("Generated library changed during measurement")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    aggregate = aggregate_control_processes(
        reports, compile_seconds=float(raw_seed["codegen_compile_seconds"])
    )
    result = {
        **aggregate, "stage": "complete" if aggregate["passed"] else "correctness",
        "artifact_directory": relpath_for_display(directory),
        "generated_library_sha256": library_hash,
        "executions": executions,
        "seed_recommendation": seed["control_loop"]["recommendation"],
    }
    _save(directory / "control-loop-aggregate.json", result)
    return result
