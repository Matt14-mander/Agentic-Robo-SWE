"""Run the M1 deterministic robot-code benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys

from agent.benchmark import load_cases, prepare_workspace, run_suite, select_cases, validate_case


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agentic-Robo-SWE M1 benchmark cases")
    parser.add_argument("--list", action="store_true", help="列出案例，不调用 LLM")
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
    parser.add_argument("--output", help="报告输出目录，默认 benchmarks/results/<run-id>")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        cases = select_cases(load_cases(), args.case_ids)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for case in cases:
            tags = ", ".join(case.tags)
            print(f"{case.id:28} {case.difficulty:6} {case.category:12} {case.title} [{tags}]")
        return 0

    if args.validate_fixtures:
        healthy = True
        for case in cases:
            prepare_workspace(case)
            result = validate_case(case)
            fixture_is_buggy = not result.passed and result.error is None
            healthy = healthy and fixture_is_buggy
            status = "OK (fails as designed)" if fixture_is_buggy else "INVALID"
            print(f"{case.id}: {status}")
            if result.error:
                print(f"  error: {result.error}")
        return 0 if healthy else 1

    report, report_path = run_suite(
        cases,
        max_loop_steps=max(1, args.max_loops),
        output_dir=args.output,
    )
    print(
        f"Benchmark complete: {report['passed']}/{report['case_count']} passed "
        f"({report['success_rate']:.1%})"
    )
    print(f"Tool calls: {report['total_tool_calls']}  Tokens: {report['total_tokens']}")
    print(f"Report: {report_path}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
