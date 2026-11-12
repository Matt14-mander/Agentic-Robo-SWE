"""Run the deterministic Phase 3.1 source-retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import hashlib
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime

from agent.rag import open_code_index
from agent.rag.cli import add_embedding_arguments, embedding_config
from agent.rag.evaluation import load_rag_cases, run_rag_evaluation, select_rag_cases


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 source-code retrieval")
    add_embedding_arguments(parser)
    parser.add_argument("--compare", action="store_true",
                        help="Compare lexical, hash vector/hybrid and real semantic vector/hybrid; no fallback")
    parser.add_argument("--workspace", help="External target for an external-path query manifest")
    parser.add_argument("--artifact-root", help="External workspace artifact directory")
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
    parser.add_argument("--persist-dir", help="Harness Chroma 目录；外部模式使用 workspace artifacts")
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

        from agent.workspace import Workspace
        if args.artifact_root and not args.workspace:
            raise ValueError("--artifact-root requires --workspace")
        workspace = Workspace.create(args.workspace, args.artifact_root) if args.workspace else None
        config = embedding_config(args)
        if args.compare:
            if args.no_sync:
                raise ValueError("--compare requires sync to verify a common source snapshot")
            return compare(args, cases, config, workspace)
        index = open_code_index(args.persist_dir, workspace=workspace, embedding=config)
        if not args.no_sync:
            stats = index.sync(args.roots, rebuild=args.rebuild)
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


def compare(args, cases, config, workspace) -> int:
    """Run all arms against identical chunk hashes; semantic failures must fail the experiment."""
    from agent.tools._paths import resolve_within_root
    destination = resolve_within_root(args.output or (
        "benchmarks/rag_results/compare-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ))
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "status": "incomplete", "created_at": datetime.now(UTC).isoformat(),
               "case_count": len(cases), "retrieval_cache_enabled": False,
               "cases_sha256": hashlib.sha256(json.dumps(
                   [asdict(case) for case in cases], sort_keys=True
               ).encode()).hexdigest(), "arms": {}}
    (destination / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    source_revision = None
    for backend in ("hash", "semantic"):
        started = time.perf_counter()
        index = open_code_index(args.persist_dir, workspace=workspace,
                                embedding=replace(config, backend=backend))
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stats = index.sync(args.roots, rebuild=args.rebuild)
        sync_seconds = time.perf_counter() - started
        snapshot = index.retrieval_manifest()
        revision = snapshot["source_revision"]
        if source_revision is not None and revision != source_revision:
            raise RuntimeError("Source changed between embedding arms; comparison rejected")
        source_revision = revision
        print(f"{backend}: {stats.chunks} chunks; model load {load_seconds:.2f}s, sync {sync_seconds:.2f}s", flush=True)
        strategies = ("lexical", "vector", "hybrid") if backend == "hash" else ("vector", "hybrid")
        for strategy in strategies:
            name = "lexical" if strategy == "lexical" else f"{backend}_{strategy}"
            report, _ = run_rag_evaluation(index, cases, top_ks=args.top_ks or [1, 3, 5],
                                           strategy=strategy, output_dir=destination / name,
                                           progress=_progress)
            if report["retrieval"]["source_revision"] != revision:
                raise RuntimeError("Source changed during retrieval evaluation; comparison rejected")
            summary["arms"][name] = {
                "metrics": report["metrics"], "average_query_latency_ms": report["average_query_latency_ms"],
                "retrieval": report["retrieval"], "model_load_seconds": round(load_seconds, 3),
                "index_sync_seconds": round(sync_seconds, 3), "chunks_updated": stats.added_or_updated,
                "index_embedding": snapshot["embedding"],
                "ranks": {row["case_id"]: row["first_relevant_rank"] for row in report["results"]},
            }
            print(f"{name}: {report['metrics']}", flush=True)
    summary["source_revision"] = source_revision
    summary["status"] = "complete"
    (destination / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(f"Comparison: {destination / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
