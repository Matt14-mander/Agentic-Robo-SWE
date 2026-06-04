"""AgentState 定义 —— LangGraph StateGraph 的共享状态契约。

设计要点:
1. 用 ``add_messages`` reducer 让消息历史天然追加 (与 LangGraph 主流模式一致)。
2. ``task`` 是不变快照, ``current_file`` 是当前游标 —— 方便后续多文件遍历。
3. 预留字段从一开始定义、Phase 1 留空,
   避免 Phase 2 改 schema 触发 checkpointer 不兼容。
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # —— 核心三字段 (用户明确指定) ——
    messages: Annotated[list[BaseMessage], add_messages]
    current_file: str | None
    loop_step: int

    # —— Phase 1 必需的辅助字段 ——
    task: str
    file_content: str | None
    suggestion: str | None

    # —— 预留字段 (Phase 2+ 启用, Phase 1 默认 None) ——
    error_log: str | None
    sandbox_id: str | None
    next_action: Literal["read", "search", "execute", "respond"] | None
