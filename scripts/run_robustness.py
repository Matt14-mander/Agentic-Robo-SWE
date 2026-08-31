"""Run Phase 5.2 robustness and persist Phase 5.3 diagnostics without an LLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent.simulation import diagnose_trial, render_diagnostic_html, run_robustness_suite
from agent.tools._paths import PROJECT_ROOT, resolve_within_root


def _load_controller_factory(path: str):
    resolved = resolve_within_root(path)
    spec = importlib.util.spec_from_file_location("robustness_controller", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load controller: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def factory():
        controller = module.TwoJointTrajectoryController(
            kp=(28.0, 22.0),
            kd=(7.0, 5.5),
            torque_limits=(10.0, 8.0),
        )
        return controller.compute

    return factory


def _default_output() -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "benchmarks" / "results" / "robustness" / run_id / "report.json"


def _write_report(report, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_payload = report.to_dict()
    report_payload["diagnostics"] = [
        diagnose_trial(trial).to_dict() for trial in report.trials
    ]
    output.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    seed_directory = output.parent / "seeds"
    seed_directory.mkdir(parents=True, exist_ok=True)
    for trial in report.trials:
        seed_path = seed_directory / f"seed-{trial.scenario.seed:05d}.json"
        seed_payload = trial.to_dict()
        seed_payload["diagnosis"] = diagnose_trial(trial).to_dict()
        seed_path.write_text(
            json.dumps(seed_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    diagnostic_path = output.parent / "diagnostics.html"
    diagnostic_path.write_text(render_diagnostic_html(report), encoding="utf-8")
    return diagnostic_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the seeded robustness matrix and render diagnostics"
    )
    parser.add_argument(
        "controller",
        nargs="?",
        default="benchmarks/fixtures/sim_two_joint_trajectory.py",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--output", help="JSON report path inside the project")
    parser.add_argument(
        "--trace-stride",
        type=int,
        default=10,
        help="capture every N simulation steps (default: 10)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="disable time-series capture and render summary-only diagnostics",
    )
    args = parser.parse_args()
    try:
        factory = _load_controller_factory(args.controller)
        seeds = tuple(args.seeds) if args.seeds else None
        report = (
            run_robustness_suite(
                factory,
                seeds=seeds,
                capture_traces=not args.no_trace,
                trace_stride=args.trace_stride,
            )
            if seeds is not None
            else run_robustness_suite(
                factory,
                capture_traces=not args.no_trace,
                trace_stride=args.trace_stride,
            )
        )
        output = resolve_within_root(args.output) if args.output else _default_output()
        diagnostic_path = _write_report(report, output)
    except (ImportError, RuntimeError, ValueError, OSError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Robustness: {report.passed_trials}/{report.total_trials} passed "
        f"({report.pass_rate:.1%})\n"
        f"P50/P95 tracking RMSE: {report.p50_tracking_rmse:.5f}/"
        f"{report.p95_tracking_rmse:.5f}\n"
        f"Safety violations: {report.safety_violations}  "
        f"Score: {report.robustness_score:.2f}\n"
        f"Report: {output}\n"
        f"Diagnostics: {diagnostic_path}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
