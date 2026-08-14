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

from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import benchmark_validate, finalize, planner
from agent.state import AgentState
from agent.tools import ALL_TOOLS

_DEFAULT_MAX_LOOP_STEPS = 15


def _is_official_validator_call(call: Any, validator: str | None) -> bool:
    if not isinstance(call, dict) or call.get("name") != "execute_python" or not validator:
        return False
    args = call.get("args") or {}
    code = args.get("code", "") if isinstance(args, dict) else ""
    return "benchmarks.validators" in str(code) and validator in str(code)


def official_validation_status(state: AgentState) -> tuple[bool, bool]:
    """Return whether the official validator ran and whether any run passed."""
    called = bool(state.get("benchmark_validator_calls"))
    passed = bool(state.get("benchmark_validation_passed"))
    validator = state.get("benchmark_validator")
    official_ids: set[str] = set()
    for message in state.get("messages") or []:
        for call in getattr(message, "tool_calls", None) or []:
            if _is_official_validator_call(call, validator):
                called = True
                call_id = call.get("id")
                if call_id:
                    official_ids.add(str(call_id))
        if isinstance(message, ToolMessage) and message.tool_call_id in official_ids:
            from agent.nodes.benchmark import validator_output_passed

            passed = passed or validator_output_passed(str(message.content))
    return called, passed


def _focused_benchmark(state: AgentState) -> bool:
    return bool(
        state.get("benchmark_mode")
        and state.get("benchmark_strategy", "focused") == "focused"
        and state.get("benchmark_validator")
        and state.get("benchmark_workspace")
    )


def should_continue(
    state: AgentState,
) -> Literal["tools", "benchmark_validate", "finalize", "__end__"]:
    """条件路由 —— ReAct 循环的"刹车"。"""
    messages = state.get("messages") or []
    if not messages:
        return "__end__"  # 不该发生, 但兜底

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    has_tool_calls = bool(tool_calls)
    benchmark_focused = _focused_benchmark(state)
    _, validation_passed = official_validation_status(state)

    # 熔断: 超出预算就停, 即使 LLM 还想继续
    loop_step = state.get("loop_step", 0)
    max_steps = state.get("max_loop_steps", _DEFAULT_MAX_LOOP_STEPS)
    if loop_step >= max_steps:
        # The official validator is allowed to cross the ordinary loop boundary;
        # otherwise the budget could suppress the evidence required for completion.
        if benchmark_focused and has_tool_calls and any(
            _is_official_validator_call(call, state.get("benchmark_validator"))
            for call in tool_calls
        ):
            return "tools"
        # 挂着未执行的 tool_calls 说明 LLM 本想继续, 被硬停 —— 需要补一次总结。
        # 没有 tool_calls 说明 planner 这一轮已自然完成, suggestion 已填, 直接 END。
        if has_tool_calls:
            return "finalize"
        return "benchmark_validate" if benchmark_focused and not validation_passed else "__end__"

    if has_tool_calls:
        return "tools"
    return "benchmark_validate" if benchmark_focused and not validation_passed else "__end__"


def after_benchmark_validation(state: AgentState) -> Literal["planner", "finalize", "__end__"]:
    """End on proof, otherwise let the model repair while normal budget remains."""
    if state.get("benchmark_validation_passed"):
        return "__end__"
    if state.get("loop_step", 0) >= state.get("max_loop_steps", _DEFAULT_MAX_LOOP_STEPS):
        return "finalize"
    return "planner"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.add_node("benchmark_validate", benchmark_validate)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "tools": "tools",
            "benchmark_validate": "benchmark_validate",
            "finalize": "finalize",
            END: END,
        },
    )
    workflow.add_edge("tools", "planner")  # 工具执行完回到 planner
    workflow.add_conditional_edges(
        "benchmark_validate",
        after_benchmark_validation,
        {"planner": "planner", "finalize": "finalize", END: END},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Studio / langgraph.json 通过此顶层名发现入口
graph = build_graph()
