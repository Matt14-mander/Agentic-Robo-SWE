"""Run the M1 deterministic robot-code benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys

from agent.benchmark import (
    domain_case_metadata,
    load_cases,
    prepare_workspace,
    run_suite,
    select_cases,
    validate_case,
)


def _timeout_value(value: str) -> float | None:
    seconds = float(value)
    if seconds < 0:
        raise argparse.ArgumentTypeError("timeout must be >= 0")
    return seconds or None


class _ConsoleProgress:
    def __init__(self) -> None:
        self.checkpoint: str | None = None

    def __call__(self, event: dict) -> None:
        kind = event["event"]
        if event.get("checkpoint"):
            self.checkpoint = str(event["checkpoint"])
        if kind == "suite_start":
            print(
                f"Starting benchmark: {event['planned_runs']} runs\n"
                f"Checkpoint: {event['checkpoint']}",
                flush=True,
            )
        elif kind == "case_start":
            print(
                f"[{event['position']}/{event['planned_runs']}] START "
                f"{event['case_id']} (repeat {event['repeat_index']}/{event['repeats']})",
                flush=True,
            )
        elif kind == "attempt_start":
            timeout = event["timeout_seconds"]
            timeout_text = f"{timeout:g}s" if timeout is not None else "disabled"
            print(
                f"  attempt {event['attempt']}/{event['max_attempts']} "
                f"(timeout {timeout_text})...",
                flush=True,
            )
        elif kind == "attempt_complete":
            status = "PASS" if event["passed"] else "TIMEOUT" if event["timed_out"] else "FAIL"
            suffix = f" — {event['error']}" if event.get("error") else ""
            print(
                f"  {status} in {event['duration_seconds']:.1f}s{suffix}",
                flush=True,
            )
        elif kind == "case_complete":
            status = "PASS" if event["passed"] else "FAIL"
            print(
                f"[{event['position']}/{event['planned_runs']}] {status} "
                f"{event['case_id']} — {event['duration_seconds']:.1f}s, "
                f"{event['tool_calls']}/{event['tool_budget']} tools, "
                f"{event['total_tokens']} tokens\n"
                f"  saved: {event['result_path']}",
                flush=True,
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agentic-Robo-SWE M1 benchmark cases")
    parser.add_argument("--list", action="store_true", help="列出案例，不调用 LLM")
    parser.add_argument(
        "--manifest",
        default="benchmarks/cases.json",
        help="案例清单；Phase 5 使用 benchmarks/sim_cases.json",
    )
    parser.add_argument(
        "--validate-fixtures",
        action="store_true",
        help="重置并验证原始题目均处于失败状态，不调用 LLM",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="case_ids",
        help="只运行指定案例，可重复传入",
    )
    parser.add_argument("--max-loops", type=int, default=12, help="每个案例的最大 Agent 循环数")
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=0,
        help="官方验证失败后允许的反馈修复次数（默认 0，保留单次基线）",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每个案例独立重复次数（默认 1）",
    )
    parser.add_argument(
        "--strategy",
        choices=("focused", "general"),
        default="focused",
        help="focused 直读目标并强制官方验证；general 保留通用探索流程",
    )
    parser.add_argument(
        "--task-timeout",
        type=_timeout_value,
        default=300.0,
        metavar="SECONDS",
        help="每次 Agent 尝试的硬超时，默认 300 秒；设为 0 可禁用",
    )
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=None,
        metavar="CALLS",
        help="覆盖每题工具预算；默认按难度使用 easy=5、medium=7、hard=9",
    )
    parser.add_argument("--output", help="报告输出目录，默认 benchmarks/results/<run-id>")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        cases = select_cases(load_cases(args.manifest), args.case_ids)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for case in cases:
            tags = ", ".join(case.tags)
            pack = f" pack={case.domain_pack}" if case.domain_pack else ""
            print(
                f"{case.id:28} {case.difficulty:6} {case.category:12} "
                f"{case.title} [{tags}]{pack}"
            )
        return 0

    if args.validate_fixtures:
        healthy = True
        for case in cases:
            _, capabilities, _ = domain_case_metadata(case)
            missing = sorted(name for name, available in capabilities.items() if not available)
            if missing:
                print(f"{case.id}: SKIPPED (missing capabilities: {', '.join(missing)})")
                continue
            prepare_workspace(case)
            result = validate_case(case)
            fixture_is_buggy = not result.passed and result.error is None
            healthy = healthy and fixture_is_buggy
            status = "OK (fails as designed)" if fixture_is_buggy else "INVALID"
            print(f"{case.id}: {status}")
            if result.error:
                print(f"  error: {result.error}")
        return 0 if healthy else 1

    blocked: list[str] = []
    for case in cases:
        _, capabilities, _ = domain_case_metadata(case)
        missing = sorted(name for name, available in capabilities.items() if not available)
        if missing:
            blocked.append(f"{case.id}: {', '.join(missing)}")
    if blocked:
        print("ERROR: required Domain Pack capabilities are unavailable:", file=sys.stderr)
        for item in blocked:
            print(f"  {item}", file=sys.stderr)
        return 2

    progress = _ConsoleProgress()
    if args.tool_budget is not None and args.tool_budget < 1:
        print("ERROR: --tool-budget must be positive", file=sys.stderr)
        return 2
    try:
        report, report_path = run_suite(
            cases,
            max_loop_steps=max(1, args.max_loops),
            repair_attempts=max(0, args.repair_attempts),
            repeats=max(1, args.repeats),
            strategy=args.strategy,
            tool_budget=args.tool_budget,
            task_timeout_seconds=args.task_timeout,
            output_dir=args.output,
            progress=progress,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; completed cases remain on disk.", file=sys.stderr)
        if progress.checkpoint:
            print(f"Checkpoint: {progress.checkpoint}", file=sys.stderr)
        return 130
    print(
        f"Benchmark complete: {report['passed']}/{report['case_count']} passed "
        f"({report['success_rate']:.1%})"
    )
    print(f"Tool calls: {report['total_tool_calls']}  Tokens: {report['total_tokens']}")
    print(
        f"Official validator compliance: {report['official_validator_compliance_rate']:.1%}  "
        f"False positives: {report['false_positives']}  Timeouts: {report['timeouts']}"
    )
    print(
        f"Averages/run: {report['average_tool_calls']:.1f} tool calls, "
        f"{report['average_tokens']:.0f} tokens"
    )
    print(f"Within tool budget: {report['tool_budget_compliance_rate']:.1%}")
    print(f"Report: {report_path}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
