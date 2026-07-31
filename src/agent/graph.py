"""图编排 —— Phase 2 ReAct 循环。

```
                  ┌────────────────────────┐
START → planner → │ should_continue?       │
   ↑              │  - has tool_calls & OK  → "tools"
   │              │  - no tool_calls        → END (自然完成, suggestion 已填)
   │              │  - budget 用完 & 有 tool_calls → "finalize" (强制总结)
   │              └────────────────────────┘
   │                       │ "tools"
   └─────── tools (ToolNode) ◀───────────────┘

                  finalize → END
```

迭代逻辑:
- ``planner`` 调一次 LLM, 输出 ``AIMessage`` (可能含 tool_calls), ``loop_step += 1``。
- ``should_continue`` 根据 tool_calls 存在性 + loop_step 上限决定走向。
- ``ToolNode`` 自动执行 AIMessage.tool_calls 里的每个调用, 写回 ``ToolMessage``。
- 若预算耗尽时 LLM 还挂着未执行的 tool_calls, 不能直接 END —— 那样 ``suggestion``
  永远是空的, 用户拿到的是一堆没有结论的消息。转去 ``finalize`` 补一次不带工具的
  LLM 调用, 强制输出总结。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import finalize, planner
from agent.state import AgentState
from agent.tools import ALL_TOOLS

_DEFAULT_MAX_LOOP_STEPS = 15


def should_continue(state: AgentState) -> Literal["tools", "finalize", "__end__"]:
    """条件路由 —— ReAct 循环的"刹车"。"""
    messages = state.get("messages") or []
    if not messages:
        return "__end__"  # 不该发生, 但兜底

    last = messages[-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))

    # 熔断: 超出预算就停, 即使 LLM 还想继续
    loop_step = state.get("loop_step", 0)
    max_steps = state.get("max_loop_steps", _DEFAULT_MAX_LOOP_STEPS)
    if loop_step >= max_steps:
        # 挂着未执行的 tool_calls 说明 LLM 本想继续, 被硬停 —— 需要补一次总结。
        # 没有 tool_calls 说明 planner 这一轮已自然完成, suggestion 已填, 直接 END。
        return "finalize" if has_tool_calls else "__end__"

    return "tools" if has_tool_calls else "__end__"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {"tools": "tools", "finalize": "finalize", END: END},
    )
    workflow.add_edge("tools", "planner")  # 工具执行完回到 planner
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Studio / langgraph.json 通过此顶层名发现入口
graph = build_graph()
