"""AgentState 定义 —— LangGraph StateGraph 的共享状态契约。

Phase 2 起 ``messages`` 成为 ReAct 循环的主线; ``task`` 仅作初始化入口,
``current_file`` / ``file_content`` / ``suggestion`` 这些 Phase 1 字段不再被节点直接写,
但保留下来供 Studio 调试或后续 Phase 复用 (向后兼容)。

设计要点:
1. ``add_messages`` reducer 让消息历史天然追加, 是 ReAct 的状态基础。
2. ``loop_step`` 在每轮 planner 节点 +1; ``max_loop_steps`` 提供熔断。
3. 新字段一律 ``total=False``, 旧 invoke 调用不会因为缺字段崩。
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # —— ReAct 主线 ——
    messages: Annotated[list[BaseMessage], add_messages]
    loop_step: int
    max_loop_steps: int  # Phase 2 新增, 默认 15

    # —— 用户入口 ——
    task: str

    # —— Phase 1 兼容字段 (节点不再主动写, 但 Studio/旧测试仍可读) ——
    current_file: str | None
    file_content: str | None
    suggestion: str | None

    # —— Phase 3+ 预留 ——
    error_log: str | None
    sandbox_id: str | None
    next_action: Literal["read", "search", "execute", "respond"] | None
