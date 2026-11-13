"""Phase 3 hybrid source retrieval tool."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from agent.rag import CodeIndex, open_code_index
from agent.rag.embedding import EmbeddingConfig
from agent.rag.settings import SearchStrategy, resolve_strategy
from agent.rag.workspace_source import source_language
from agent.workspace import Workspace, current_workspace

_MAX_RESULT_CHARS = 4_000


def get_code_index() -> CodeIndex:
    return _get_code_index(current_workspace(), EmbeddingConfig.from_env())


@lru_cache(maxsize=8)
def _get_code_index(workspace: Workspace | None, embedding: EmbeddingConfig) -> CodeIndex:
    return open_code_index(workspace=workspace, embedding=embedding)


@tool
def search_code_knowledge(
    query: str,
    config: RunnableConfig,
    top_k: int = 5,
    path_prefix: str | None = None,
    strategy: SearchStrategy | None = None,
) -> str:
    """选择关键词、向量或混合检索源码索引，适合目标未知或跨文件定位。

    Args:
        query: 用自然语言描述要找的行为、算法、错误或相关实现。
        top_k: 返回结果数，默认 5，上限 20。
        path_prefix: 可选的仓库相对路径前缀，例如 ``src/agent``。
        strategy: lexical 适合标识符/术语，vector 适合语义描述，hybrid 融合两者。
            省略时使用会话默认策略；向量可能来自语义模型或离线 hash 回退。

    精确符号或已知字符串优先使用 grep_codebase；已知文件直接 read_file_chunk。
    索引需先运行 ``uv run python scripts/index_codebase.py`` 创建；外部仓库加 --workspace。
    自动选择当前 workspace，返回片段行号；C++ 符号行列需读文件确认后再用 cpp_semantic_query。
    """
    try:
        selected = resolve_strategy(strategy, config.get("configurable", {}).get("rag_strategy"))
        results = get_code_index().search(
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            strategy=selected,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        return f"ERROR: {exc}"
    if not results:
        if current_workspace():
            return (
                "<no indexed code matches; build/sync with scripts/index_codebase.py "
                "--workspace <target> (and the session's --artifact-root), or broaden the query>"
            )
        return "<no indexed code matches; run scripts/index_codebase.py or broaden the query>"

    sections: list[str] = []
    for rank, result in enumerate(results, start=1):
        content = result.content
        if len(content) > _MAX_RESULT_CHARS:
            content = content[:_MAX_RESULT_CHARS] + "\n... [chunk truncated]"
        sections.append(
            f"## {rank}. {result.path}:{result.start_line}-{result.end_line} "
            f"{result.symbol} ({result.kind}, score={result.score:.3f})\n"
            f"```{source_language(result.path)}\n{content.rstrip()}\n```"
        )
        if result.parent_context:
            parent = result.parent_context
            sections[-1] += (
                f"\nMatched child: {result.start_line}:{result.start_column}"
                f"–{result.end_line}:{result.end_column}; "
                f"parent: {result.path}:{parent.start_line}-{parent.end_line}"
            )
            remaining = max(0, _MAX_RESULT_CHARS - len(content))
            if remaining:
                context = parent.content[:remaining]
                note = " (excerpt; read full parent range for more)" if len(context) < len(parent.content) else ""
                sections[-1] += (
                    f"\nParent context{note}:\n"
                    f"```{source_language(result.path)}\n{context.rstrip()}\n```"
                )
    return f"# {selected.capitalize()} code search results\n\n" + "\n\n".join(sections)
