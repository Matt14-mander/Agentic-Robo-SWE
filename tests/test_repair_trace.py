"""Offline trace integration: real graph/tool dispatch, no model or compiler needed."""

import importlib
import uuid
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.repair_trace import (
    TraceEvent, bounded, inspect_trace, trace_node, use_repair_trace,
)
from agent.workspace import Workspace, source_digest, use_workspace


def events(directory):
    return [TraceEvent.model_validate_json(line) for line in
            (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.cpp").write_text("int value = 1;\n", encoding="utf-8")
    return Workspace.create(root, tmp_path / "artifacts", trusted_local=True)


def patch_planner(state):
    step = state.get("loop_step", 0)
    calls = [
        {"name": "read_file_chunk", "args": {"path": "main.cpp"}, "id": "read"},
        {"name": "write_patch", "args": {"path": "main.cpp", "old_string": "value = 1",
                                         "new_string": "value = 2"}, "id": "patch"},
        {"name": "cpp_build", "args": {}, "id": "build"},
    ]
    response = AIMessage(content=[{"type": "text", "text": "Check the value"},
                                  {"type": "thinking", "thinking": "private-reasoning"}],
                         tool_calls=[calls[step]] if step < len(calls) else [],
                         usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                         "total_tokens": 15})
    return {"messages": [response], "loop_step": step + 1, "suggestion": "done"}


def test_graph_records_patch_build_and_source_without_changing_behavior(monkeypatch, workspace):
    graph_module = importlib.import_module("agent.graph")
    cpp_module = importlib.import_module("agent.tools.cpp_ops")
    monkeypatch.setattr(graph_module, "planner", patch_planner)
    monkeypatch.setattr(cpp_module, "configure_build", lambda *_: {
        "passed": True, "stage": "build", "exit_code": 0, "stdout": "", "stderr": ""})
    graph = graph_module.build_graph()
    state = {"task": "fix", "loop_step": 0, "max_loop_steps": 8,
             "cpp_tools_enabled": True, "workspace_binding": workspace.binding()}
    directory = workspace.artifacts / "trace"
    with use_workspace(workspace):
        initial = source_digest(workspace.root)
        with use_repair_trace(directory, workspace.root, {"task": "fix"}):
            traced = graph.invoke(state)
        final = source_digest(workspace.root)
        (workspace.root / "main.cpp").write_text("int value = 1;\n", encoding="utf-8")
        plain = graph.invoke(state)
    assert traced["suggestion"] == plain["suggestion"]
    assert traced["loop_step"] == plain["loop_step"] == 4
    assert len(traced["messages"]) == len(plain["messages"])
    rows = events(directory)
    assert rows[0].source.digest == initial
    assert rows[-1].source.digest == final != initial
    assert sum(e.data.get("source_changed") is True for e in rows) == 1
    results = [e for e in rows if e.kind == "tool_result"]
    assert [e.data["tool_call_id"] for e in results] == ["read", "patch", "build"]
    assert results[-1].data["observation"] == {
        "kind": "build", "reported_passed": True, "stage": "build", "exit_code": 0}
    assert all(e.data["source_attribution"] == "node_batch" for e in results)
    assert "private-reasoning" not in (directory / "events.jsonl").read_text(encoding="utf-8")
    assert [e.data["usage"]["total_tokens"] for e in rows if e.kind == "agent_message"] == [15] * 4
    summary = inspect_trace(directory)
    assert summary["session_closed"] and not summary["unfinished_spans"]
    assert summary["source_changes"] == 1
    assert summary["skill_eligible"] is False
    assert "repair_trace" not in traced


def test_approval_resume_segments_and_denial(monkeypatch, workspace):
    graph_module = importlib.import_module("agent.graph")
    monkeypatch.setattr(graph_module, "planner", patch_planner)
    graph = graph_module.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "trace-resume"}}
    state = {"task": "fix", "loop_step": 1, "max_loop_steps": 3,
             "workspace_binding": workspace.binding(), "hitl_enabled": True,
             "cpp_tools_enabled": True}
    # After the denial, finish without invoking a model/finalizer.
    def finish(_state):
        return {"messages": [AIMessage(content="denied")], "loop_step": 2}

    with use_workspace(workspace):
        with use_repair_trace(workspace.artifacts / "first", workspace.root,
                              {"thread_id": "trace-resume"}):
            graph.invoke(state, config)
        assert graph.get_state(config).interrupts
        # Recompile with the same checkpointer, as a fresh CLI process would.
        monkeypatch.setattr(graph_module, "planner", finish)
        resumed = graph_module.build_graph(checkpointer=graph.checkpointer)
        with use_repair_trace(workspace.artifacts / "second", workspace.root,
                              {"thread_id": "trace-resume", "resumed": True}):
            resumed.invoke(Command(resume={"approved": False}), config)
    assert any(e.kind == "node_interrupted" for e in events(workspace.artifacts / "first"))
    assert any(e.kind == "tool_result" and e.data["status"] == "denied"
               for e in events(workspace.artifacts / "second"))
    assert (workspace.root / "main.cpp").read_text() == "int value = 1;\n"


def test_budget_skips_are_not_executed_or_accepted(monkeypatch, workspace):
    from agent.nodes.finalize import finalize
    module = importlib.import_module("agent.nodes.finalize")
    class Model:
        def invoke(self, _messages):
            return AIMessage(content="not verified")
    monkeypatch.setattr(module, "get_chat_model", lambda **_: Model())
    directory = workspace.artifacts / "trace"
    with use_repair_trace(directory, workspace.root, {}):
        trace_node("finalize", finalize)(patch_planner({"loop_step": 1}), {})
    result = next(e for e in events(directory) if e.kind == "tool_result")
    assert result.data["status"] == "skipped"
    assert result.data["observation"] is None


def test_gate_records_development_evidence_not_final_evaluation(monkeypatch, workspace):
    module = importlib.import_module("agent.nodes.benchmark")
    class Executor:
        def invoke(self, _args):
            return "exit_code: 0\n{'passed': True, 'details': {'max_error': 0.0}}"
    monkeypatch.setattr(module, "execute_python", Executor())
    directory = workspace.artifacts / "trace"
    with use_repair_trace(directory, workspace.root, {}):
        trace_node("benchmark_validate", module.benchmark_validate)({
            "benchmark_validator": "public", "benchmark_workspace": "public.cpp"}, {})
    rows = events(directory)
    validation = next(e for e in rows if e.kind == "development_validation")
    assert validation.data["reported_passed"] is True
    enclosing = next(e for e in rows if e.kind == "node_completed")
    assert validation.span_id == enclosing.span_id
    assert enclosing.source.digest == source_digest(workspace.root)
    assert all(e.scope == "development" for e in rows)


def test_error_and_incomplete_tail_survive_for_inspection(workspace):
    directory = workspace.artifacts / "error"
    def fail(_state):
        raise RuntimeError("example")
    with pytest.raises(RuntimeError):
        with use_repair_trace(directory, workspace.root, {}):
            trace_node("planner", fail)({}, {})
    assert any(e.kind == "node_error" for e in events(directory))
    assert events(directory)[-1].data["status"] == "error"
    # Simulate a worker killed mid-write, including an incomplete UTF-8 character.
    path = directory / "events.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:2]) + b'{"text":"\xe4')
    summary = inspect_trace(directory)
    assert summary["partial_tail"] and not summary["session_closed"]
    assert len(summary["unfinished_spans"]) == 1


def test_trace_is_bounded_and_refuses_scope_or_manifest_tampering(workspace):
    payload = bounded({"api_key": "private", "input_tokens": 10, "text": "x" * 13000})
    assert payload["api_key"] == "[REDACTED]"
    assert payload["input_tokens"] == 10
    assert payload["text"]["truncated"]
    directory = workspace.artifacts / "trace"
    with use_repair_trace(directory, workspace.root, {}):
        pass
    path = directory / "events.jsonl"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace('"scope":"development"', '"scope":"hidden"'), encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_trace(directory)
    path.write_text(original, encoding="utf-8")
    source_path = next((directory / "sources").glob("*.json"))
    source_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        inspect_trace(directory)


def test_output_cannot_pollute_source_identity_and_missing_source_is_explicit(workspace):
    with pytest.raises(ValueError, match="overlap"):
        with use_repair_trace(workspace.root / "trace", workspace.root, {}):
            pass
    source = workspace.root / "main.cpp"
    directory = workspace.artifacts / "trace"
    with use_repair_trace(directory, source, {}):
        source.unlink()
    assert events(directory)[-1].source.status == "unavailable"
    assert inspect_trace(directory)["source_unavailable"] == 1


def test_benchmark_reports_trace_and_keeps_parent_validation_outside(tmp_path):
    from agent.benchmark import load_cases, run_suite
    from agent.tools._paths import PROJECT_ROOT
    base = next(c for c in load_cases() if c.id == "angle_units")
    case = replace(base, workspace="benchmarks/workspaces/_trace_angle.py")
    source = PROJECT_ROOT / case.workspace
    output = PROJECT_ROOT / "benchmarks/results" / f"trace-test-{uuid.uuid4().hex}"
    def fake_invoke(state, config):
        source.write_text("import math\ndef degrees_to_radians(x): return x * math.pi / 180\n")
        return {"messages": [], "loop_step": 1}
    report, _ = run_suite([case], invoke=fake_invoke, record_traces=True, output_dir=output)
    assert report["repair_traces_enabled"]
    trace_dirs = list((output / "traces").iterdir())
    assert len(trace_dirs) == 1
    assert inspect_trace(trace_dirs[0])["development_validations"] == 0
    assert [e.kind for e in events(trace_dirs[0])] == ["session_started", "session_closed"]
    assert report["results"][0]["attempt_history"][0]["repair_trace"] == str(trace_dirs[0])


def test_real_worker_persists_failed_attempt_before_returning(workspace):
    from agent.benchmark import _invoke_graph_subprocess
    directory = workspace.artifacts / "worker"
    # Invalid task fails before constructing an LLM client. Trace context must
    # cross the process boundary separately from planner-visible state/config.
    with pytest.raises(RuntimeError, match="must contain either"):
        _invoke_graph_subprocess({"task": ""}, {}, 30, {
            "directory": str(directory), "source": str(workspace.root),
            "metadata": {"case_id": "offline-worker"},
        })
    rows = events(directory)
    assert any(e.kind == "node_error" for e in rows)
    assert rows[-1].kind == "session_closed"
    assert rows[-1].data["status"] == "error"
    assert inspect_trace(directory)["unfinished_spans"] == []
