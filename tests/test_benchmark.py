"""M1 benchmark manifest, validators and runner tests."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent.benchmark as benchmark_module
from agent.benchmark import (
    BenchmarkAttemptTimeout,
    build_task,
    load_cases,
    prepare_workspace,
    run_case,
    run_suite,
    select_cases,
    validate_case,
)
from agent.nodes.planner import build_system_prompt, focused_message_context
from agent.tools._paths import PROJECT_ROOT
from benchmarks.validators import validate


def test_manifest_loads_unique_robotics_cases():
    cases = load_cases()

    assert len(cases) == 5
    assert len({case.id for case in cases}) == len(cases)
    assert {case.category for case in cases} >= {"kinematics", "geometry", "control"}


def test_all_fixture_baselines_fail_as_designed():
    for case in load_cases():
        prepare_workspace(case)
        result = validate_case(case)
        assert result.error is None, f"{case.id}: {result.error}"
        assert not result.passed, f"{case.id} fixture unexpectedly passes"


def test_select_cases_preserves_requested_order_and_rejects_unknown():
    cases = load_cases()

    selected = select_cases(cases, ["trajectory_endpoints", "angle_units"])
    assert [case.id for case in selected] == ["trajectory_endpoints", "angle_units"]

    with pytest.raises(ValueError, match="Unknown benchmark"):
        select_cases(cases, ["missing-case"])


def test_build_task_contains_workspace_and_deterministic_validator():
    case = load_cases()[0]
    task = build_task(case)

    assert case.workspace in task
    assert "benchmarks.validators" in task
    assert case.validator in task
    assert "只修改目标文件" in task


def test_run_case_collects_metrics_and_passes_after_fake_fix():
    base_case = next(case for case in load_cases() if case.id == "angle_units")
    workspace = "benchmarks/workspaces/_test_angle_units.py"
    case = replace(base_case, workspace=workspace)
    workspace_path = PROJECT_ROOT / workspace

    def fake_invoke(state, config):
        assert state["max_loop_steps"] == 4
        assert state["benchmark_mode"] is True
        assert state["benchmark_strategy"] == "focused"
        assert state["benchmark_tool_budget"] == 5
        assert state["benchmark_validator"] == case.validator
        assert config["recursion_limit"] == 50
        workspace_path.write_text(
            "import math\n\n"
            "def degrees_to_radians(angle_degrees: float) -> float:\n"
            "    return angle_degrees * math.pi / 180.0\n",
            encoding="utf-8",
        )
        message = AIMessage(
            content="fixed",
            tool_calls=[{
                "name": "write_patch",
                "args": {"path": workspace},
                "id": "call-1",
                "type": "tool_call",
            }],
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )
        return {
            "messages": [message],
            "loop_step": 2,
            "suggestion": "fixed and verified",
            "benchmark_validator_calls": 1,
            "benchmark_validation_passed": True,
        }

    try:
        result = run_case(case, max_loop_steps=4, invoke=fake_invoke)
    finally:
        workspace_path.unlink(missing_ok=True)

    assert result.success
    assert not result.baseline_passed
    assert result.workspace_modified
    assert result.loop_steps == 2
    assert result.tool_calls == 2
    assert result.tool_budget == 5
    assert result.within_tool_budget
    assert result.tool_calls_by_name == {"execute_python": 1, "write_patch": 1}
    assert result.total_tokens == 14
    assert result.official_validator_called
    assert result.validator_compliance_rate == 1.0
    assert not result.false_positive
    assert result.error is None


def test_focused_prompt_overrides_redundant_repository_exploration():
    focused = build_system_prompt({
        "benchmark_mode": True,
        "benchmark_strategy": "focused",
        "max_loop_steps": 6,
    })
    general = build_system_prompt({
        "benchmark_mode": True,
        "benchmark_strategy": "general",
        "max_loop_steps": 6,
    })

    assert "Do not call list_dir or grep_codebase" in focused
    assert "official validator reports passed=True" in focused
    assert "Tool-call budget: 6" in focused
    assert "Benchmark focused mode" not in general


def test_focused_context_keeps_task_and_recent_complete_tool_rounds():
    messages = [HumanMessage(content="task")]
    for index in range(4):
        call_id = f"call-{index}"
        messages.extend([
            AIMessage(content="", tool_calls=[{
                "name": "read_file_chunk",
                "args": {"path": "target.py"},
                "id": call_id,
                "type": "tool_call",
            }]),
            ToolMessage(content=f"round {index}", tool_call_id=call_id),
        ])

    context = focused_message_context(messages, tool_rounds=2)

    assert context[0].content == "task"
    assert "Earlier benchmark tool rounds omitted" in str(context[1].content)
    assert [message.content for message in context if isinstance(message, ToolMessage)] == [
        "round 2",
        "round 3",
    ]


def test_validator_resolves_workspace_from_project_root(monkeypatch, tmp_path):
    workspace = "benchmarks/workspaces/_test_validator_root.py"
    workspace_path = PROJECT_ROOT / workspace
    workspace_path.write_text(
        "import math\n\n"
        "def degrees_to_radians(angle_degrees: float) -> float:\n"
        "    return angle_degrees * math.pi / 180.0\n",
        encoding="utf-8",
    )
    try:
        monkeypatch.chdir(tmp_path)
        result = validate("angle_units", workspace)
    finally:
        workspace_path.unlink(missing_ok=True)

    assert result["passed"] is True
    assert result["error"] is None


def test_failed_claim_is_recorded_and_validator_feedback_can_repair():
    base_case = next(case for case in load_cases() if case.id == "quaternion_normalization")
    workspace = "benchmarks/workspaces/_test_quaternion.py"
    case = replace(base_case, workspace=workspace)
    workspace_path = PROJECT_ROOT / workspace
    calls = 0

    def fake_invoke(state, config):
        nonlocal calls
        calls += 1
        validator_call = {
            "name": "execute_python",
            "args": {
                "code": "from benchmarks.validators import validate\n"
                f"print(validate({case.validator!r}, {workspace!r}))"
            },
            "id": f"validator-{calls}",
            "type": "tool_call",
        }
        if calls == 1:
            workspace_path.write_text(
                "import math\n\n"
                "def normalize_quaternion(q):\n"
                "    norm = math.sqrt(sum(value * value for value in q))\n"
                "    return tuple(value / norm for value in q)\n",
                encoding="utf-8",
            )
            suggestion = "修复完成，验证通过"
        else:
            assert "previous attempt failed" in state["task"]
            assert "ZeroDivisionError" in state["task"]
            workspace_path.write_text(
                "import math\n\n"
                "def normalize_quaternion(q):\n"
                "    norm = math.sqrt(sum(value * value for value in q))\n"
                "    if norm == 0:\n"
                "        raise ValueError('zero quaternion')\n"
                "    return tuple(value / norm for value in q)\n",
                encoding="utf-8",
            )
            suggestion = "Fix is complete and verified"
        return {
            "messages": [AIMessage(content=suggestion, tool_calls=[validator_call])],
            "loop_step": 2,
            "suggestion": suggestion,
        }

    try:
        result = run_case(case, max_loop_steps=4, repair_attempts=1, invoke=fake_invoke)
    finally:
        workspace_path.unlink(missing_ok=True)

    assert result.success
    assert result.attempts == 2
    assert result.repair_attempts_used == 1
    assert result.official_validator_calls == 2
    assert result.validator_compliance_rate == 1.0
    assert result.false_positive
    assert result.attempt_history[0]["false_positive"] is True
    assert result.attempt_history[1]["validation_passed"] is True


def test_run_suite_aggregates_repeated_runs():
    base_case = next(case for case in load_cases() if case.id == "angle_units")
    workspace = "benchmarks/workspaces/_test_repeat_angle.py"
    output = "benchmarks/results/_test_m11_repeat"
    case = replace(base_case, workspace=workspace)
    workspace_path = PROJECT_ROOT / workspace
    output_path = PROJECT_ROOT / output
    progress_events = []

    def fake_invoke(state, config):
        workspace_path.write_text(
            "import math\n\n"
            "def degrees_to_radians(angle_degrees: float) -> float:\n"
            "    return angle_degrees * math.pi / 180.0\n",
            encoding="utf-8",
        )
        validator_call = {
            "name": "execute_python",
            "args": {
                "code": "from benchmarks.validators import validate\n"
                f"print(validate({case.validator!r}, {workspace!r}))"
            },
            "id": "validator",
            "type": "tool_call",
        }
        return {
            "messages": [AIMessage(content="verified", tool_calls=[validator_call])],
            "loop_step": 1,
            "suggestion": "verified",
        }

    try:
        report, report_path = run_suite(
            [case],
            repeats=2,
            output_dir=output,
            progress=progress_events.append,
            invoke=fake_invoke,
        )
        assert report_path.is_file()
        checkpoint = json.loads((output_path / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["status"] == "completed"
        assert checkpoint["completed_runs"] == 2
        assert (output_path / "cases" / "repeat-01__angle_units.json").is_file()
        assert (output_path / "cases" / "repeat-02__angle_units.json").is_file()
    finally:
        workspace_path.unlink(missing_ok=True)
        (output_path / "report.json").unlink(missing_ok=True)
        (output_path / "checkpoint.json").unlink(missing_ok=True)
        for repeat_index in (1, 2):
            (output_path / "cases" / f"repeat-{repeat_index:02d}__angle_units.json").unlink(
                missing_ok=True
            )
        if (output_path / "cases").exists():
            (output_path / "cases").rmdir()
        if output_path.exists():
            output_path.rmdir()

    assert report["schema_version"] == 4
    assert report["status"] == "completed"
    assert report["case_count"] == 2
    assert report["completed_runs"] == 2
    assert report["success_rate"] == 1.0
    assert report["official_validator_compliance_rate"] == 1.0
    assert report["tool_budget_compliance_rate"] == 1.0
    assert report["per_case"][case.id]["runs"] == 2
    assert [event["event"] for event in progress_events].count("case_complete") == 2
    assert progress_events[0]["event"] == "suite_start"
    assert progress_events[-1]["event"] == "suite_complete"


def test_isolated_attempt_timeout_is_reported(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0]), timeout=0.01)

    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_run)

    with pytest.raises(BenchmarkAttemptTimeout, match="exceeded 0.01s"):
        benchmark_module._invoke_graph_subprocess({}, {}, 0.01)


def test_run_case_records_timeout_and_continues_to_validation(monkeypatch):
    base_case = next(case for case in load_cases() if case.id == "angle_units")
    workspace = "benchmarks/workspaces/_test_timeout_angle.py"
    case = replace(base_case, workspace=workspace)
    workspace_path = PROJECT_ROOT / workspace
    events = []

    def fake_timeout(state, config, timeout_seconds):
        assert timeout_seconds == 0.25
        raise BenchmarkAttemptTimeout("Agent attempt exceeded 0.25s and was terminated")

    monkeypatch.setattr(benchmark_module, "_invoke_graph_subprocess", fake_timeout)
    try:
        result = run_case(case, task_timeout_seconds=0.25, progress=events.append)
    finally:
        workspace_path.unlink(missing_ok=True)

    assert not result.success
    assert result.timed_out
    assert result.timeout_attempts == 1
    assert result.attempt_history[0]["timed_out"] is True
    assert "BenchmarkAttemptTimeout" in (result.error or "")
    assert [event["event"] for event in events] == ["attempt_start", "attempt_complete"]


def test_interruption_keeps_checkpoint_on_disk():
    base_case = next(case for case in load_cases() if case.id == "angle_units")
    workspace = "benchmarks/workspaces/_test_interrupt_angle.py"
    output = "benchmarks/results/_test_m12_interrupt"
    case = replace(base_case, workspace=workspace)
    workspace_path = PROJECT_ROOT / workspace
    output_path = PROJECT_ROOT / output

    def interrupted_invoke(state, config):
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            run_suite([case], output_dir=output, invoke=interrupted_invoke)
        checkpoint = json.loads((output_path / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["status"] == "interrupted"
        assert checkpoint["completed_runs"] == 0
        assert checkpoint["fatal_error"] == "KeyboardInterrupt: "
        assert not (output_path / "report.json").exists()
    finally:
        workspace_path.unlink(missing_ok=True)
        (output_path / "checkpoint.json").unlink(missing_ok=True)
        if (output_path / "cases").exists():
            (output_path / "cases").rmdir()
        if output_path.exists():
            output_path.rmdir()
