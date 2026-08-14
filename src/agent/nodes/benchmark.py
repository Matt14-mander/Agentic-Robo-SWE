"""Deterministic benchmark-only validation gate."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from agent.state import AgentState
from agent.tools import execute_python

_PASSED_RE = re.compile(r"[\"']passed[\"']\s*:\s*(?:True|true)")


def validator_output_passed(output: str) -> bool:
    """Recognize a successful official validator result in tool output."""
    return "exit_code: 0" in output and bool(_PASSED_RE.search(output))


def benchmark_validate(state: AgentState) -> dict:
    """Run the official validator when the model tries to finish without proof."""
    validator = state.get("benchmark_validator")
    workspace = state.get("benchmark_workspace")
    if not validator or not workspace:
        raise ValueError("Benchmark validation gate requires validator and workspace")

    code = (
        "from benchmarks.validators import validate\n"
        f"print(validate({validator!r}, {workspace!r}))"
    )
    output = str(execute_python.invoke({"code": code}))
    passed = validator_output_passed(output)
    if passed:
        message = "M1.2 validator gate: official validator passed."
    else:
        message = (
            "M1.2 validator gate: official validator failed. Use this evidence to fix the "
            f"remaining defect, then validate again.\n\n{output}"
        )
    return {
        "messages": [HumanMessage(content=message)],
        "benchmark_validator_calls": state.get("benchmark_validator_calls", 0) + 1,
        "benchmark_validation_passed": passed,
    }
