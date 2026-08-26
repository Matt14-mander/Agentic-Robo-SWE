"""Build, incrementally refresh and inspect the Phase 3 source-code index."""

from __future__ import annotations

import argparse
import sys

from agent.rag import open_code_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index source code into persistent ChromaDB")
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="仓库相对源码根，可重复传入；默认仅索引 src",
    )
    parser.add_argument("--rebuild", action="store_true", help="清空 collection 后完整重建")
    parser.add_argument("--persist-dir", default="chroma_db", help="Chroma 持久化目录")
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
    roots = args.roots or ["src"]
    try:
        index = open_code_index(args.persist_dir)
        stats = index.sync(roots, rebuild=args.rebuild)
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
        for _ in range(args.repeat_query):
            results = index.search(
                args.query,
                top_k=max(1, args.top_k),
                path_prefix=args.path_prefix,
                strategy=args.strategy,
            )
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
