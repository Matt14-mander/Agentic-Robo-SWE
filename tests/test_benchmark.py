"""M1 benchmark manifest, validators and runner tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage

from agent.benchmark import (
    build_task,
    load_cases,
    prepare_workspace,
    run_case,
    select_cases,
    validate_case,
)
from agent.tools._paths import PROJECT_ROOT


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
        return {"messages": [message], "loop_step": 2, "suggestion": "fixed and verified"}

    try:
        result = run_case(case, max_loop_steps=4, invoke=fake_invoke)
    finally:
        workspace_path.unlink(missing_ok=True)

    assert result.success
    assert not result.baseline_passed
    assert result.workspace_modified
    assert result.loop_steps == 2
    assert result.tool_calls == 1
    assert result.tool_calls_by_name == {"write_patch": 1}
    assert result.total_tokens == 14
    assert result.error is None
