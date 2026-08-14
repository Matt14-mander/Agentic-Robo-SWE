"""M1.2 deterministic validator gate tests."""

from __future__ import annotations

import importlib

from langchain_core.messages import HumanMessage

from agent.nodes.benchmark import validator_output_passed
from agent.tools import execute_python
from agent.tools._paths import PROJECT_ROOT


def test_validator_output_passed_requires_clean_exit_and_passed_payload():
    assert validator_output_passed("exit_code: 0\n{'passed': True, 'details': []}")
    assert not validator_output_passed("exit_code: 1\n{'passed': True}")
    assert not validator_output_passed("exit_code: 0\n{'passed': False}")


def test_benchmark_gate_runs_exact_validator_and_records_result(monkeypatch):
    module = importlib.import_module("agent.nodes.benchmark")
    captured = {}

    class FakeExecutor:
        def invoke(self, payload):
            captured.update(payload)
            return "exit_code: 0\n--- stdout ---\n{'passed': True, 'details': []}"

    monkeypatch.setattr(module, "execute_python", FakeExecutor())
    update = module.benchmark_validate({
        "benchmark_validator": "angle_units",
        "benchmark_workspace": "benchmarks/workspaces/angle_units.py",
        "benchmark_validator_calls": 1,
    })

    assert "benchmarks.validators" in captured["code"]
    assert "angle_units" in captured["code"]
    assert update["benchmark_validator_calls"] == 2
    assert update["benchmark_validation_passed"] is True
    assert isinstance(update["messages"][0], HumanMessage)


def test_official_validator_runs_from_execute_python_tempdir(monkeypatch):
    workspace = "benchmarks/workspaces/_test_gate_angle.py"
    workspace_path = PROJECT_ROOT / workspace
    workspace_path.write_text(
        "import math\n\n"
        "def degrees_to_radians(angle_degrees: float) -> float:\n"
        "    return angle_degrees * math.pi / 180.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXECUTOR_BACKEND", "local")
    code = (
        "from benchmarks.validators import validate\n"
        f"print(validate('angle_units', {workspace!r}))"
    )
    try:
        output = str(execute_python.invoke({"code": code}))
    finally:
        workspace_path.unlink(missing_ok=True)

    assert validator_output_passed(output)
    assert "Workspace not found" not in output
