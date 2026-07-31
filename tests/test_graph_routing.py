"""ReAct 条件路由与预算耗尽收尾节点的离线测试。"""

from __future__ import annotations

import importlib

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent.graph import should_continue


def _tool_call(call_id: str = "call-1") -> dict:
    return {
        "name": "read_file_chunk",
        "args": {"path": "README.md"},
        "id": call_id,
        "type": "tool_call",
    }


def test_should_continue_routes_tool_call_with_budget():
    state = {
        "messages": [AIMessage(content="", tool_calls=[_tool_call()])],
        "loop_step": 1,
        "max_loop_steps": 2,
    }

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
