"""ReAct 条件路由与预算耗尽收尾节点的离线测试。"""

from __future__ import annotations

import importlib

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent.graph import after_benchmark_validation, should_continue


def _tool_call(call_id: str = "call-1") -> dict:
    return {
        "name": "read_file_chunk",
        "args": {"path": "README.md"},
        "id": call_id,
        "type": "tool_call",
    }


def _validator_call(call_id: str = "validator-1") -> dict:
    return {
        "name": "execute_python",
        "args": {
            "code": "from benchmarks.validators import validate\n"
            "print(validate('angle_units', 'benchmarks/workspaces/angle_units.py'))"
        },
        "id": call_id,
        "type": "tool_call",
    }


def _write_call(call_id: str = "write-1") -> dict:
    return {
        "name": "write_patch",
        "args": {"path": "README.md", "old_string": "old", "new_string": "new"},
        "id": call_id,
        "type": "tool_call",
    }


def _benchmark_state(messages, loop_step=1, max_steps=4):
    return {
        "messages": messages,
        "loop_step": loop_step,
        "max_loop_steps": max_steps,
        "benchmark_mode": True,
        "benchmark_strategy": "focused",
        "benchmark_validator": "angle_units",
        "benchmark_workspace": "benchmarks/workspaces/angle_units.py",
    }


def test_should_continue_routes_tool_call_with_budget():
    state = {
        "messages": [AIMessage(content="", tool_calls=[_tool_call()])],
        "loop_step": 1,
        "max_loop_steps": 2,
    }

    assert should_continue(state) == "tools"


def test_should_continue_routes_interactive_write_to_approval():
    state = {
        "messages": [AIMessage(content="", tool_calls=[_write_call()])],
        "loop_step": 1,
        "max_loop_steps": 2,
        "hitl_enabled": True,
    }

    assert should_continue(state) == "approval"


def test_benchmark_write_never_blocks_on_hitl():
    state = _benchmark_state([AIMessage(content="", tool_calls=[_write_call()])])
    state["hitl_enabled"] = True

    assert should_continue(state) == "tools"


def test_should_continue_routes_pending_call_to_finalize_at_budget():
    state = {
        "messages": [AIMessage(content="", tool_calls=[_tool_call()])],
        "loop_step": 2,
        "max_loop_steps": 2,
    }

    assert should_continue(state) == "finalize"


def test_should_continue_ends_after_natural_text_response():
    state = {
        "messages": [AIMessage(content="done")],
        "loop_step": 2,
        "max_loop_steps": 2,
    }

    assert should_continue(state) == END


def test_should_continue_empty_messages_is_safe():
    assert should_continue({"messages": []}) == END


def test_benchmark_final_answer_is_gated_until_official_validation():
    state = _benchmark_state([AIMessage(content="fixed")])

    assert should_continue(state) == "benchmark_validate"


def test_successful_model_validator_call_allows_benchmark_to_end():
    call = _validator_call()
    state = _benchmark_state([
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(
            content="exit_code: 0\n--- stdout ---\n{'passed': True, 'details': []}",
            tool_call_id=call["id"],
            name="execute_python",
        ),
        AIMessage(content="verified"),
    ])

    assert should_continue(state) == END


def test_official_validator_call_can_cross_loop_budget():
    state = _benchmark_state(
        [AIMessage(content="", tool_calls=[_validator_call()])],
        loop_step=4,
        max_steps=4,
    )

    assert should_continue(state) == "tools"


def test_benchmark_gate_routes_by_result_and_remaining_budget():
    assert after_benchmark_validation({"benchmark_validation_passed": True}) == END
    assert after_benchmark_validation({"loop_step": 1, "max_loop_steps": 2}) == "planner"
    assert after_benchmark_validation({"loop_step": 2, "max_loop_steps": 2}) == "finalize"


def test_finalize_balances_pending_tool_calls_and_writes_suggestion(monkeypatch):
    finalize_module = importlib.import_module("agent.nodes.finalize")
    pending = AIMessage(
        content="",
        tool_calls=[_tool_call("call-1"), _tool_call("call-2")],
    )
    captured_messages = []

    class FakeModel:
        def invoke(self, messages):
            captured_messages.extend(messages)
            return AIMessage(content="forced final summary")

    monkeypatch.setattr(
        finalize_module,
        "get_chat_model",
        lambda temperature=0.0: FakeModel(),
    )

    update = finalize_module.finalize(
        {"messages": [HumanMessage(content="fix the bug"), pending]}
    )

    skipped = update["messages"][:-1]
    assert len(skipped) == 2
    assert all(isinstance(message, ToolMessage) for message in skipped)
    assert [message.tool_call_id for message in skipped] == ["call-1", "call-2"]
    assert all(message.content.startswith("SKIPPED:") for message in skipped)
    assert captured_messages[-2:] == skipped
    assert update["suggestion"] == "forced final summary"


def test_focused_planner_binds_only_targeted_tools(monkeypatch):
    planner_module = importlib.import_module("agent.nodes.planner")
    bound_tool_names = []

    class FakeModel:
        def bind_tools(self, tools):
            bound_tool_names.extend(tool.name for tool in tools)
            return self

        def invoke(self, messages):
            return AIMessage(content="done")

    monkeypatch.setattr(
        planner_module,
        "get_chat_model",
        lambda temperature=0.0: FakeModel(),
    )
    planner_module.planner({
        "task": "fix target",
        "benchmark_mode": True,
        "benchmark_strategy": "focused",
        "benchmark_tool_budget": 5,
    })

    assert bound_tool_names == ["read_file_chunk", "execute_python", "write_patch"]


def test_general_planner_exposes_semantic_rag_tool(monkeypatch):
    planner_module = importlib.import_module("agent.nodes.planner")
    bound_tool_names = []

    class FakeModel:
        def bind_tools(self, tools):
            bound_tool_names.extend(tool.name for tool in tools)
            return self

        def invoke(self, messages):
            return AIMessage(content="done")

    monkeypatch.setattr(
        planner_module,
        "get_chat_model",
        lambda temperature=0.0: FakeModel(),
    )
    planner_module.planner({"task": "locate anti-windup logic"})

    assert "search_code_knowledge" in bound_tool_names
