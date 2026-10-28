"""Operator-only hidden evaluation: candidate execution never mounts the oracle.

The trusted oracle image prepares inputs and judges observations in separate
containers. The candidate runner sees inputs, but neither expected values nor
the oracle image filesystem. This module is deliberately NOT an Agent tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from agent.containers import ProcessResult, run_container, validate_image
from agent.workspace import exported_source, source_digest


def strict_json(text: str) -> Any:
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject, object_pairs_hook=pairs)


def load_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    if len(raw) > 100_000:
        raise ValueError("Evaluation spec too large")
    spec = strict_json(raw.decode("utf-8"))
    required = {"schema_version", "task_id", "runner_image", "oracle_image",
                "runner_command", "oracle_command"}
    if not isinstance(spec, dict) or set(spec) - required - {"timeout_seconds"} or not required <= spec.keys():
        raise ValueError("Invalid evaluation spec fields")
    if spec["schema_version"] != 1 or not isinstance(spec["task_id"], str) or not spec["task_id"]:
        raise ValueError("Invalid evaluation task/schema")
    for key in ("runner_image", "oracle_image"):
        validate_image(spec[key])
    if spec["runner_image"] == spec["oracle_image"]:
        raise ValueError("Runner and oracle must be separate images")
    for key in ("runner_command", "oracle_command"):
        command = spec[key]
        if (not isinstance(command, list) or not command or
                not all(isinstance(arg, str) and arg and "\0" not in arg for arg in command)):
            raise ValueError("Evaluation commands must be non-empty argument arrays")
    timeout = spec.get("timeout_seconds", 120)
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise ValueError("timeout_seconds must be an integer in [1, 600]")
    return spec, hashlib.sha256(raw).hexdigest()


def evaluate_repository(root: Path, specification: Path, output: Path) -> dict[str, Any]:
    root, specification, output = root.resolve(strict=True), specification.resolve(strict=True), output.resolve()
    if not root.is_dir():
        raise ValueError("Candidate repository must be a directory")
    if (specification.is_relative_to(root) or output.is_relative_to(root) or root.is_relative_to(output)):
        raise ValueError("Private specification and evaluation artifacts must be outside candidate repository")
    spec, spec_hash = load_spec(specification)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="evaluation-", dir=output))
    if os.name != "nt":
        run_dir.chmod(0o700)
    report: dict[str, Any] = {
        "schema_version": 1, "task_id": spec["task_id"], "passed": False,
        "stage": "snapshot", "error": None, "spec_sha256": spec_hash,
        "runner_image": spec["runner_image"], "oracle_image": spec["oracle_image"],
        "artifact_dir": str(run_dir), "evaluation_scope": "isolated_final",
    }
    timeout = spec.get("timeout_seconds", 120)

    def decode(result: ProcessResult, stage: str):
        (run_dir / f"{stage}.stdout").write_text(result.stdout, encoding="utf-8")
        (run_dir / f"{stage}.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"{stage} failed (exit {result.returncode}); see private artifacts")
        return strict_json(result.stdout)

    try:
        original_hash = source_digest(root)
        with exported_source(root) as snapshot:
            snapshot_hash = source_digest(snapshot)
            if snapshot_hash != original_hash or source_digest(root) != original_hash:
                raise ValueError("Candidate changed while creating snapshot")
            report["source_sha256"] = snapshot_hash
            report["stage"] = "prepare"
            prepared = decode(run_container(spec["oracle_image"], [*spec["oracle_command"], "prepare"],
                                            [], stdin=json.dumps({"task_id": spec["task_id"]}),
                                            timeout=timeout), "prepare")
            if not isinstance(prepared, dict) or set(prepared) != {"cases"} or not isinstance(prepared["cases"], list):
                raise ValueError("Oracle prepare must return exactly a cases array")
            if not 1 <= len(prepared["cases"]) <= 10000:
                raise ValueError("Oracle cases must contain 1–10,000 entries")
            report["stage"] = "candidate"
            observed = decode(run_container(spec["runner_image"], spec["runner_command"],
                                           [(snapshot, "/candidate", False)],
                                           stdin=json.dumps(prepared, allow_nan=False), timeout=timeout), "candidate")
            if (not isinstance(observed, dict) or set(observed) != {"observations"} or
                    not isinstance(observed["observations"], list) or
                    len(observed["observations"]) != len(prepared["cases"])):
                raise ValueError("Candidate must return observations matching the case count, never passed")
            report["stage"] = "judge"
            judged = decode(run_container(spec["oracle_image"], [*spec["oracle_command"], "judge"], [],
                                         stdin=json.dumps({**prepared, **observed}, allow_nan=False),
                                         timeout=timeout), "judge")
            if (not isinstance(judged, dict) or set(judged) != {"passed", "checks"} or
                    type(judged["passed"]) is not bool or type(judged["checks"]) is not int or
                    judged["checks"] != len(prepared["cases"])):
                raise ValueError("Invalid independent oracle verdict")
            if source_digest(snapshot) != snapshot_hash or source_digest(root) != original_hash:
                raise ValueError("Candidate source changed during evaluation; verdict is stale")
            report.update(passed=judged["passed"], checks=judged["checks"], stage="complete")
    except (OSError, ValueError, RuntimeError) as exc:
        # No hidden inputs / expected values / untrusted logs in model-visible summaries.
        report["error"] = f"{type(exc).__name__}: evaluation failed at {report['stage']}; inspect private artifacts"
        (run_dir / "failure.txt").write_text(str(exc), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def receipt_matches(report: dict[str, Any], root: Path, specification: Path) -> bool:
    """A prior pass is usable only for the same source and evaluation specification."""
    _, identity = load_spec(specification)
    return (report.get("passed") is True and report.get("stage") == "complete" and
            report.get("source_sha256") == source_digest(root) and report.get("spec_sha256") == identity)
