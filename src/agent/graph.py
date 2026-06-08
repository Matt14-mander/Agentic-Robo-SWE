"""图编排 —— Phase 2 ReAct 循环。

```
                  ┌────────────────────────┐
START → planner → │ should_continue?       │
   ↑              │  - has tool_calls → "tools"
   │              │  - else → END (loop budget OK)
   │              │  - loop_step >= max → END (force stop)
   │              └────────────────────────┘
   │                       │ "tools"
   └─────── tools (ToolNode) ◀───────────────┘
```

迭代逻辑:
- ``planner`` 调一次 LLM, 输出 ``AIMessage`` (可能含 tool_calls), ``loop_step += 1``。
- ``should_continue`` 根据 tool_calls 存在性 + loop_step 上限决定走 tools 还是 END。
- ``ToolNode`` 自动执行 AIMessage.tool_calls 里的每个调用, 写回 ``ToolMessage``。
- 回到 ``planner`` 进入下一轮。
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import planner
from agent.state import AgentState
from agent.tools import ALL_TOOLS

_DEFAULT_MAX_LOOP_STEPS = 15


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """条件路由 —— ReAct 循环的"刹车"。"""
    messages = state.get("messages") or []
    if not messages:
        return END  # 不该发生, 但兜底

    last = messages[-1]
    has_tool_calls = bool(getattr(last, "tool_calls", None))

    # 熔断: 超出预算就停, 即使 LLM 还想继续
    loop_step = state.get("loop_step", 0)
    max_steps = state.get("max_loop_steps", _DEFAULT_MAX_LOOP_STEPS)
    if loop_step >= max_steps:
        return END

    return "tools" if has_tool_calls else END


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "planner")  # 工具执行完回到 planner

    return workflow.compile()


# Studio / langgraph.json 通过此顶层名发现入口
graph = build_graph()
