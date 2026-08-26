"""Human approval gate for state-changing tool calls."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from agent.state import AgentState


def _pending_calls(state: AgentState) -> list[dict[str, Any]]:
    messages = state.get("messages") or []
    if not messages:
        return []
    return [call for call in (getattr(messages[-1], "tool_calls", None) or []) if isinstance(call, dict)]


def requires_write_approval(state: AgentState) -> bool:
    """Only interactive, non-benchmark write operations require approval."""
    if not state.get("hitl_enabled") or state.get("benchmark_mode"):
        return False
    return any(call.get("name") == "write_patch" for call in _pending_calls(state))


def _parse_decision(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"approve", "approved", "yes", "y", "true"}, value
    if isinstance(value, dict):
        approved = value.get("approved", False)
        feedback = value.get("feedback")
        return bool(approved), str(feedback) if feedback is not None else None
    return False, "Unsupported approval response; denied safely."


def request_write_approval(state: AgentState) -> dict[str, Any]:
    """Pause before write_patch and resume with an explicit human decision."""
    write_calls = [call for call in _pending_calls(state) if call.get("name") == "write_patch"]
    if not write_calls:
        return {"hitl_decision": True, "hitl_feedback": None}

    response = interrupt({
        "kind": "write_patch_approval",
        "question": "Approve the proposed repository modification?",
        "tool_calls": [
            {
                "id": str(call.get("id", "")),
                "name": str(call.get("name", "")),
                "args": call.get("args") or {},
            }
            for call in write_calls
        ],
    })
    approved, feedback = _parse_decision(response)
    return {"hitl_decision": approved, "hitl_feedback": feedback}


def after_write_approval(state: AgentState) -> Literal["tools", "reject_tools"]:
    return "tools" if state.get("hitl_decision") else "reject_tools"


def reject_pending_tools(state: AgentState) -> dict[str, Any]:
    """Balance rejected tool calls and return the human feedback to the planner."""
    feedback = state.get("hitl_feedback") or "No feedback provided."
    messages = [
        ToolMessage(
            content=f"DENIED by human reviewer. Feedback: {feedback}",
            tool_call_id=str(call.get("id", "unknown")),
            name=str(call.get("name", "unknown")),
        )
        for call in _pending_calls(state)
    ]
    return {
        "messages": messages,
        "hitl_decision": None,
        "hitl_feedback": None,
    }
