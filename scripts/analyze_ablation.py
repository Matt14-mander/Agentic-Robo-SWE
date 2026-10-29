"""Audit a completed four-arm experiment and export portable observed-cost evidence.

No model calls or validators are run. Monetary pricing belongs to a dated report,
not this analyzer: token counts do not prove an account's actual charge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from agent.ablation import ARMS, Budget, fingerprint, summarize, worker_failure
from agent.repair_trace import inspect_trace
from agent.workspace import source_digest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory: Path) -> dict[str, Any]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or len(report["results"]) != report["planned_runs"]:
        raise ValueError("Analyze only completed experiments; incomplete runs need a separate report")
    if report["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Report/protocol identity mismatch")
    if fingerprint({key: value for key, value in protocol.items() if key != "protocol_id"}) != protocol["protocol_id"]:
        raise ValueError("Protocol digest mismatch")
    if any(report.get(key) != value for key, value in protocol.items()):
        raise ValueError("Report changed frozen protocol fields")
    expected = {(case["id"], repeat, arm) for case in protocol["catalog"]
                for repeat in range(1, protocol["repeats"] + 1) for arm in ARMS}
    observed = [(row["case_id"], row["repeat"], row["arm"]) for row in report["results"]]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Missing, duplicated or unexpected attempts")
    budget = Budget(**report["budget"])
    rows = []
    for row in report["results"]:
        slot = directory / "runs" / f"{row['case_id']}-{row['repeat']}-{row['arm']}"
        trace = slot / "trace"
        audit: dict[str, Any] = {"trace_available": trace.is_dir(), "issues": []}
        issues = audit["issues"]
        counters: Counter = Counter()
        models: Counter = Counter()
        fingerprints: Counter = Counter()
        tools: Counter = Counter()
        tool_failures: Counter = Counter()
        development: Counter = Counter()
        reasons: Counter = Counter()
        cached = 0
        cache_complete = True
        if trace.is_dir():
            inspected = inspect_trace(trace)
            audit.update(inspected)
            events = [json.loads(line) for line in (trace / "events.jsonl").read_text(
                encoding="utf-8").splitlines()[:inspected["events"]]]
            manifest = json.loads((trace / "manifest.json").read_text(encoding="utf-8"))
            if manifest["metadata"]["protocol_id"] != report["protocol_id"]:
                issues.append("trace protocol mismatch")
            if (not inspected["session_closed"] or inspected["partial_tail"] or
                    inspected["unfinished_spans"] or inspected["source_unavailable"]):
                issues.append("incomplete trace")
            if not events or events[0]["source"]["digest"] != row["source_before"]:
                issues.append("initial source mismatch")
            if not events or (events[-1].get("source") or {}).get("digest") != row["source_after"]:
                issues.append("final trace source mismatch")
            for event in events:
                data = event["data"]
                if event["kind"] == "agent_message":
                    usage = data.get("usage") or {}
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        counters[key] += usage.get(key, 0)
                    counters["model_responses"] += 1
                    cache_read = usage.get("input_token_details", {}).get("cache_read")
                    if type(cache_read) is int and 0 <= cache_read <= usage.get("input_tokens", 0):
                        cached += cache_read
                    else:
                        cache_complete = False
                    metadata = data.get("provider_response", {})
                    models[metadata.get("model_name", metadata.get("model", "unknown"))] += 1
                    fingerprints[metadata.get("system_fingerprint") or "unknown"] += 1
                    reasons[metadata.get("finish_reason") or "unknown"] += 1
                elif event["kind"] == "tool_result":
                    if data.get("status") != "skipped":
                        tools[data["tool"]] += 1
                    failure = data.get("failure") or {}
                    if failure.get("category", "none") != "none":
                        tool_failures[failure["category"]] += 1
                elif event["kind"] == "development_validation":
                    development["passed" if data.get("reported_passed") is True else "failed"] += 1
                    if row["arm"] != "feedback":
                        issues.append("development feedback outside feedback arm")
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if key in row.get("worker", {}) and row["worker"][key] != counters[key]:
                    issues.append(f"worker/trace {key} mismatch")
            audit["events_sha256"] = sha256(trace / "events.jsonl")
            audit["manifest_sha256"] = sha256(trace / "manifest.json")
        elif row["calibrated"]:
            issues.append("trace missing")
        if source_digest(slot / "workspace") != row["source_after"]:
            issues.append("saved source mismatch")
        if row.get("evaluated_source", row["source_after"]) != row["source_after"]:
            issues.append("final evaluation source mismatch")
        if row["calibrated"]:
            failure = worker_failure(row["worker"], budget)
            if failure and row["failure"] != failure:
                issues.append("parent failure mismatch")
        if row["success"] != (row["failure"]["category"] == "none"):
            issues.append("success/failure mismatch")
        if row["success"] and row.get("final_validation", {}).get("passed") is not True:
            issues.append("success without final validation")
        rows.append({"case_id": row["case_id"], "repeat": row["repeat"], "arm": row["arm"],
                     "protocol_id": report["protocol_id"],
                     "case_spec_sha256": fingerprint(next(c for c in protocol["catalog"]
                                                          if c["id"] == row["case_id"])),
                     "execution_order": row["execution_order"], "success": row["success"],
                     "calibrated": row["calibrated"], "failure": row["failure"],
                     "final_passed": row.get("final_validation", {}).get("passed"),
                     "attempt_seconds": row.get("attempt_seconds", 0),
                     "final_validation_seconds": row.get("final_validation_seconds", 0),
                     "usage": dict(counters), "cached_input_tokens": cached,
                     "cache_usage_complete": cache_complete and bool(counters["model_responses"]),
                     "served_models": dict(models), "system_fingerprints": dict(fingerprints),
                     "finish_reasons": dict(reasons), "tool_calls": dict(tools),
                     "development_validations": dict(development),
                     "intermediate_tool_failures": dict(tool_failures), "audit": audit,
                     "source_before": row["source_before"], "source_after": row["source_after"]})
    return {"schema_version": 1, "mode": report["mode"], "status": report["status"],
            "evaluation_scope": report["evaluation_scope"], "requested_model": report["environment"]["model"],
            "environment_id": report["environment_id"],
            "protocol_id": report["protocol_id"], "report_sha256": sha256(directory / "report.json"),
            "protocol_sha256": sha256(directory / "protocol.json"), "budget": report["budget"],
            "seed": report["seed"], "repeats": report["repeats"], "planned_runs": report["planned_runs"],
            "audit_ok": all(not row["audit"]["issues"] for row in rows),
            "cost_scope": "provider-reported returned responses; not a billing receipt; excludes preflight",
            "summary": summarize_rows(rows, report["mode"], report["seed"]), "results": rows}


def summarize_rows(rows: list[dict[str, Any]], mode: str, seed: int) -> dict[str, Any]:
    summary = summarize(rows, mode, seed)
    for arm in ARMS:
        members = [row for row in rows if row["arm"] == arm]
        attempted = [row for row in members if row["calibrated"]]
        usage = {key: sum(row["usage"].get(key, 0) for row in members)
                 for key in ("input_tokens", "output_tokens", "total_tokens", "model_responses")}
        tools_total: Counter = Counter()
        intermediate_failures: Counter = Counter()
        dev_total: Counter = Counter()
        for row in members:
            tools_total.update(row["tool_calls"])
            intermediate_failures.update(row["intermediate_tool_failures"])
            dev_total.update(row["development_validations"])
        successes = summary[arm]["successes"]
        summary[arm].update(usage=usage, cached_input_tokens=sum(r["cached_input_tokens"] for r in members),
                            cache_usage_complete=bool(attempted) and all(r["cache_usage_complete"] for r in attempted),
                            tool_calls=dict(tools_total),
                            intermediate_tool_failures=dict(intermediate_failures),
                            development_validations=dict(dev_total),
                            tokens_per_success=usage["total_tokens"] / successes if successes else None,
                            attempt_seconds=sum(r["attempt_seconds"] for r in members),
                            median_attempt_seconds=median([r["attempt_seconds"] for r in attempted])
                            if attempted else None,
                            final_validation_seconds=sum(r["final_validation_seconds"] for r in members),
                            trace_recording_seconds=sum(r["audit"].get("recording_seconds") or 0
                                                        for r in members))
    return summary


def merge_calibration_recovery(primary: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    """Fill only pre-model calibration blocks; never replace a paid failed attempt."""
    for key in ("budget", "mode", "environment_id", "requested_model", "evaluation_scope", "repeats"):
        if primary[key] != recovery[key]:
            raise ValueError(f"Recovery changed {key}")
    def identity(row):
        return row["case_id"], row["repeat"], row["arm"]
    old = {identity(row): row for row in primary["results"]}
    replacements = {identity(row): row for row in recovery["results"]}
    if [identity(row) for row in primary["results"] if identity(row) in replacements] != list(replacements):
        raise ValueError("Recovery changed arm order")
    for key, row in replacements.items():
        previous = old.get(key)
        if (previous is None or previous["calibrated"] or previous["audit"]["trace_available"] or
                previous["usage"].get("model_responses", 0) or not row["calibrated"]):
            raise ValueError("Recovery may only fill attempts blocked before model execution")
        if any(previous[field] != row[field] for field in ("case_spec_sha256", "source_before")):
            raise ValueError("Recovery changed task or starting source")
    rows = [replacements.get(identity(row), row) for row in primary["results"]]
    return {**{key: value for key, value in primary.items()
               if key not in {"protocol_id", "report_sha256", "protocol_sha256"}},
            "analysis_kind": "calibration_recovery_composite",
            "segments": [{key: segment[key] for key in ("protocol_id", "report_sha256", "protocol_sha256", "seed")}
                         for segment in (primary, recovery)],
            "recovered_pre_model_blocks": [old[key] for key in replacements],
            "audit_ok": primary["audit_ok"] and recovery["audit_ok"],
            "summary": summarize_rows(rows, primary["mode"], primary["seed"]), "results": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--recovery", type=Path, help="Fill only previously uncalibrated, unexecuted slots")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    if args.recovery:
        result = merge_calibration_recovery(result, analyze(args.recovery))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_ok": result["audit_ok"], "summary": result["summary"]}, indent=2))
    return 0 if result["audit_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
