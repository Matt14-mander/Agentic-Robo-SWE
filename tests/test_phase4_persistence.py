"""Phase 4 SQLite checkpoint and resumable write-approval tests."""

from __future__ import annotations

import importlib

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.persistence import open_persistent_graph
from agent.tools._paths import PROJECT_ROOT


def _fake_planner_for_patch(relative: str):
    def fake_planner(state):
        messages = list(state.get("messages") or [])
        loop_step = state.get("loop_step", 0)
        if messages and isinstance(messages[-1], ToolMessage):
            return {
                "messages": [AIMessage(content="review handled")],
                "loop_step": loop_step + 1,
                "suggestion": "review handled",
            }
        return {
            "messages": [AIMessage(
                content="",
                tool_calls=[{
                    "name": "write_patch",
                    "args": {
                        "path": relative,
                        "old_string": "value = 1",
                        "new_string": "value = 2",
                    },
                    "id": "write-1",
                    "type": "tool_call",
                }],
            )],
            "loop_step": loop_step + 1,
        }

    return fake_planner


def test_hitl_interrupt_approves_write_and_resumes(monkeypatch):
    graph_module = importlib.import_module("agent.graph")
    path = PROJECT_ROOT / "tests" / "_scratch_phase4_approve.py"
    relative = "tests/_scratch_phase4_approve.py"
    path.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(graph_module, "planner", _fake_planner_for_patch(relative))
    graph = graph_module.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approve"}, "recursion_limit": 20}
    try:
        graph.invoke(
            {"task": "change value", "loop_step": 0, "max_loop_steps": 4, "hitl_enabled": True},
            config,
        )
        snapshot = graph.get_state(config)
        assert snapshot.interrupts
        assert snapshot.interrupts[0].value["kind"] == "write_patch_approval"
        assert path.read_text(encoding="utf-8") == "value = 1\n"

        result = graph.invoke(Command(resume={"approved": True}), config)
        assert path.read_text(encoding="utf-8") == "value = 2\n"
        assert result["suggestion"] == "review handled"
        assert not graph.get_state(config).interrupts
    finally:
        path.unlink(missing_ok=True)


def test_hitl_interrupt_denies_write_and_returns_feedback(monkeypatch):
    graph_module = importlib.import_module("agent.graph")
    path = PROJECT_ROOT / "tests" / "_scratch_phase4_deny.py"
    relative = "tests/_scratch_phase4_deny.py"
    path.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(graph_module, "planner", _fake_planner_for_patch(relative))
    graph = graph_module.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "deny"}, "recursion_limit": 20}
    try:
        graph.invoke(
            {"task": "change value", "loop_step": 0, "max_loop_steps": 4, "hitl_enabled": True},
            config,
        )
        result = graph.invoke(
            Command(resume={"approved": False, "feedback": "keep the original"}),
            config,
        )

        assert path.read_text(encoding="utf-8") == "value = 1\n"
        denied = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        assert any("DENIED" in str(message.content) for message in denied)
        assert result["suggestion"] == "review handled"
    finally:
        path.unlink(missing_ok=True)


def test_sqlite_checkpoint_survives_graph_reopen(monkeypatch):
    graph_module = importlib.import_module("agent.graph")
    database = PROJECT_ROOT / "benchmarks" / "workspaces" / "_test_phase4.sqlite"
    config = {"configurable": {"thread_id": "persistent"}, "recursion_limit": 20}

    def complete_planner(state):
        return {
            "messages": [AIMessage(content="persisted summary")],
            "loop_step": state.get("loop_step", 0) + 1,
            "suggestion": "persisted summary",
        }

    monkeypatch.setattr(graph_module, "planner", complete_planner)
    try:
        with open_persistent_graph(database) as first_graph:
            result = first_graph.invoke({"task": "persist me", "loop_step": 0}, config)
            assert result["suggestion"] == "persisted summary"

        with open_persistent_graph(database) as reopened_graph:
            snapshot = reopened_graph.get_state(config)
            assert snapshot.values["suggestion"] == "persisted summary"
            assert snapshot.values["task"] == "persist me"
    finally:
        database.unlink(missing_ok=True)
        database.with_name(database.name + "-shm").unlink(missing_ok=True)
        database.with_name(database.name + "-wal").unlink(missing_ok=True)
