"""Phase 3 hybrid source retrieval tool."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool

from agent.rag import CodeIndex, open_code_index

_MAX_RESULT_CHARS = 4_000


@lru_cache(maxsize=1)
def get_code_index() -> CodeIndex:
    return open_code_index()


@tool
def search_code_knowledge(
    query: str,
    top_k: int = 5,
    path_prefix: str | None = None,
) -> str:
    """混合检索源码索引，适合目标文件未知、概念匹配或跨文件定位。

    Args:
        query: 用自然语言描述要找的行为、算法、错误或相关实现。
        top_k: 返回结果数，默认 5，上限 20。
        path_prefix: 可选的仓库相对路径前缀，例如 ``src/agent``。

    精确符号或已知字符串优先使用 grep_codebase；已知文件直接 read_file_chunk。
    索引需先运行 ``uv run python scripts/index_codebase.py`` 创建。
    """
    try:
        results = get_code_index().search(
            query,
            top_k=top_k,
            path_prefix=path_prefix,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        return f"ERROR: {exc}"
    if not results:
        return "<no indexed code matches; run scripts/index_codebase.py or broaden the query>"

    sections: list[str] = []
    for rank, result in enumerate(results, start=1):
        content = result.content
        if len(content) > _MAX_RESULT_CHARS:
            content = content[:_MAX_RESULT_CHARS] + "\n... [chunk truncated]"
        sections.append(
            f"## {rank}. {result.path}:{result.start_line}-{result.end_line} "
            f"{result.symbol} ({result.kind}, score={result.score:.3f})\n"
            f"```python\n{content.rstrip()}\n```"
        )
    return "# Hybrid code search results\n\n" + "\n\n".join(sections)
