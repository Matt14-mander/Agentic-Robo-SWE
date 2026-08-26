"""Run the deterministic Phase 3.1 source-retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import sys

from agent.rag import open_code_index
from agent.rag.evaluation import load_rag_cases, run_rag_evaluation, select_rag_cases


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 source-code retrieval")
    parser.add_argument("--list", action="store_true", help="列出查询案例，不运行检索")
    parser.add_argument("--case", action="append", default=[], dest="case_ids", help="只运行指定案例")
    parser.add_argument("--manifest", default="benchmarks/rag_cases.json", help="评测查询集")
    parser.add_argument("--root", action="append", dest="roots", help="索引源码根；默认 src")
    parser.add_argument("--top-k", action="append", type=int, dest="top_ks", help="统计 K，可重复传入")
    parser.add_argument(
        "--strategy",
        choices=("vector", "lexical", "hybrid"),
        default="hybrid",
        help="检索策略，默认 hybrid",
    )
    parser.add_argument("--persist-dir", default="chroma_db", help="Chroma 持久化目录")
    parser.add_argument("--rebuild", action="store_true", help="评测前完整重建索引")
    parser.add_argument("--no-sync", action="store_true", help="直接评测已有索引")
    parser.add_argument("--output", help="输出目录，默认 benchmarks/rag_results/<run-id>")
    return parser


def _progress(event: dict) -> None:
    if event["event"] == "suite_start":
        print(f"Starting RAG evaluation: {event['case_count']} queries", flush=True)
    elif event["event"] == "case_start":
        print(
            f"[{event['position']}/{event['case_count']}] {event['case_id']}...",
            end=" ",
            flush=True,
        )
    elif event["event"] == "case_complete":
        rank = event["first_relevant_rank"]
        print(f"rank={rank if rank is not None else 'MISS'} {event['latency_ms']:.1f}ms", flush=True)


def main() -> int:
    args = _build_parser().parse_args()
    try:
        cases = select_rag_cases(load_rag_cases(args.manifest), args.case_ids)
        top_ks = args.top_ks or [1, 3, 5]
        if args.list:
            for case in cases:
                targets = ", ".join(
                    f"{target.path}:{target.symbol or '*'}" for target in case.targets
                )
                print(f"{case.id:28} {case.query} -> {targets}")
            return 0

        index = open_code_index(args.persist_dir)
        if not args.no_sync:
            stats = index.sync(args.roots or ["src"], rebuild=args.rebuild)
            print(
                f"Index ready: {stats.files} files, {stats.chunks} chunks "
                f"({stats.added_or_updated} updated, {stats.unchanged} unchanged, "
                f"{stats.deleted} deleted)"
            )
        report, report_path = run_rag_evaluation(
            index,
            cases,
            top_ks=top_ks,
            strategy=args.strategy,
            output_dir=args.output,
            progress=_progress,
        )
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Strategy: {report['strategy']}")
    for k in report["top_ks"]:
        metrics = report["metrics"][f"at_{k}"]
        print(
            f"@{k}: hit={metrics['hit_rate']:.1%} recall={metrics['recall']:.1%} "
            f"MRR={metrics['mrr']:.3f} path={metrics['path_hit_rate']:.1%} "
            f"symbol={metrics['symbol_hit_rate']:.1%}"
        )
    print(f"Average query latency: {report['average_query_latency_ms']:.1f}ms")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
