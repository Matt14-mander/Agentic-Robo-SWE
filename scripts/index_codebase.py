"""Build, incrementally refresh and inspect the Phase 3 source-code index."""

from __future__ import annotations

import argparse
import sys

from agent.rag import open_code_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index source code into persistent ChromaDB")
    parser.add_argument("--workspace", help="External target repository; defaults to the harness")
    parser.add_argument("--artifact-root", help="External workspace artifacts, outside the target")
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="仓库相对源码根，可重复传入；Harness 默认 src，外部仓库默认整个目录",
    )
    parser.add_argument("--rebuild", action="store_true", help="清空 collection 后完整重建")
    parser.add_argument("--persist-dir", help="Harness Chroma 目录，默认 chroma_db；外部模式不可覆盖")
    parser.add_argument("--query", help="建索引后立即执行一次查询")
    parser.add_argument(
        "--repeat-query",
        type=int,
        default=1,
        metavar="N",
        help="在同一进程重复查询以检查缓存；默认 1",
    )
    parser.add_argument("--top-k", type=int, default=5, help="查询返回数量，默认 5")
    parser.add_argument("--path-prefix", help="查询结果路径前缀过滤")
    parser.add_argument(
        "--strategy",
        choices=("vector", "lexical", "hybrid"),
        default="hybrid",
        help="查询策略，默认 hybrid",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        from agent.workspace import Workspace
        if args.artifact_root and not args.workspace:
            raise ValueError("--artifact-root requires --workspace")
        if args.workspace and args.persist_dir:
            raise ValueError("--persist-dir cannot override workspace RAG artifacts")
        workspace = Workspace.create(args.workspace, args.artifact_root) if args.workspace else None
        index = open_code_index(args.persist_dir, workspace=workspace)
        stats = index.sync(args.roots, rebuild=args.rebuild)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Index ready: {stats.files} files, {stats.chunks} chunks "
        f"({stats.added_or_updated} updated, {stats.unchanged} unchanged, "
        f"{stats.deleted} deleted)"
    )
    if args.query:
        if args.repeat_query < 1:
            print("ERROR: --repeat-query must be positive", file=sys.stderr)
            return 2
        results = []
        try:
            for _ in range(args.repeat_query):
                results = index.search(
                    args.query,
                    top_k=max(1, args.top_k),
                    path_prefix=args.path_prefix,
                    strategy=args.strategy,
                )
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result.path}:{result.start_line}-{result.end_line} "
                f"{result.symbol} score={result.score:.3f}"
            )
        cache = index.cache_stats()
        print(
            f"Cache: {cache.hits} hits, {cache.misses} misses, "
            f"{cache.hit_rate:.1%} hit rate, {cache.invalidations} invalidations, "
            f"~{cache.estimated_saved_ms:.1f}ms saved, revision={cache.revision}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
