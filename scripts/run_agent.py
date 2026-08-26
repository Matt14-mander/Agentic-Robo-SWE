"""Run, inspect and resume a Phase 4 persistent Agent session."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent.persistence import open_persistent_graph


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a resumable Phase 4 Agent session")
    parser.add_argument("task", nargs="?", help="新会话任务；恢复或查看状态时省略")
    parser.add_argument("--thread-id", help="持久会话 ID；新会话未提供时自动生成")
    parser.add_argument("--checkpoint-db", default=".agent_state/checkpoints.sqlite")
    parser.add_argument("--max-loops", type=int, default=12)
    parser.add_argument("--hitl", action="store_true", help="write_patch 前暂停并请求审批")
    parser.add_argument("--stream", action="store_true", help="逐节点显示新增消息")
    parser.add_argument("--status", action="store_true", help="只查看指定会话状态")
    parser.add_argument("--continue", dest="continue_run", action="store_true", help="从普通待执行 checkpoint 继续")
    parser.add_argument("--resume", choices=("approve", "deny"), help="恢复待审批 write_patch")
    parser.add_argument("--feedback", help="审批说明，尤其适合拒绝时提供修改方向")
    return parser


def _format_message(message: Any) -> str:
    if isinstance(message, HumanMessage):
        return f"[HUMAN] {message.content}"
    if isinstance(message, AIMessage):
        content = message.content if isinstance(message.content, str) else str(message.content)
        calls = getattr(message, "tool_calls", None) or []
        suffix = "\n" + "\n".join(
            f"  → {call.get('name')}({call.get('args') or {}})" for call in calls
        ) if calls else ""
        return f"[AI] {content}{suffix}"
    if isinstance(message, ToolMessage):
        return f"[TOOL:{message.name}] {message.content}"
    return f"[{type(message).__name__}] {message}"


def _print_status(snapshot: Any, thread_id: str) -> None:
    values = snapshot.values or {}
    print(f"Thread: {thread_id}")
    print(f"Next nodes: {', '.join(snapshot.next) if snapshot.next else '<complete>'}")
    print(f"Loop step: {values.get('loop_step', 0)}")
    print(f"HITL enabled: {bool(values.get('hitl_enabled'))}")
    if snapshot.interrupts:
        print("Pending interrupts:")
        for item in snapshot.interrupts:
            print(json.dumps(item.value, ensure_ascii=False, indent=2, default=str))
    if values.get("suggestion"):
        print(f"Suggestion: {values['suggestion']}")


def _run(agent: Any, graph_input: Any, config: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    if not stream:
        result = agent.invoke(graph_input, config)
        return dict(result or {})

    printed = 0
    latest: dict[str, Any] = {}
    for chunk in agent.stream(graph_input, config, stream_mode="values"):
        latest = dict(chunk or {})
        messages = latest.get("messages") or []
        for message in messages[printed:]:
            print(_format_message(message))
        printed = len(messages)
    return latest


def main() -> int:
    args = _build_parser().parse_args()
    modes = sum(bool(item) for item in (args.status, args.continue_run, args.resume))
    if modes > 1:
        print("ERROR: --status, --continue and --resume are mutually exclusive", file=sys.stderr)
        return 2
    if (args.status or args.continue_run or args.resume) and not args.thread_id:
        print("ERROR: restoring or inspecting requires --thread-id", file=sys.stderr)
        return 2
    if not (args.status or args.continue_run or args.resume) and not args.task:
        print("ERROR: provide a task for a new session", file=sys.stderr)
        return 2

    thread_id = args.thread_id or f"agent-{uuid.uuid4().hex[:12]}"
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }

    try:
        with open_persistent_graph(Path(args.checkpoint_db)) as agent:
            snapshot = agent.get_state(config)
            if args.status:
                if not snapshot.values:
                    print(f"ERROR: thread not found: {thread_id}", file=sys.stderr)
                    return 1
                _print_status(snapshot, thread_id)
                return 0

            if args.resume:
                if not snapshot.interrupts:
                    print(f"ERROR: thread has no pending approval: {thread_id}", file=sys.stderr)
                    return 2
                graph_input: Any = Command(resume={
                    "approved": args.resume == "approve",
                    "feedback": args.feedback,
                })
            elif args.continue_run:
                if snapshot.interrupts:
                    print("ERROR: pending HITL approval; use --resume approve|deny", file=sys.stderr)
                    return 2
                if not snapshot.values or not snapshot.next:
                    print(f"ERROR: thread has no pending work: {thread_id}", file=sys.stderr)
                    return 2
                graph_input = None
            else:
                if snapshot.values:
                    print(
                        f"ERROR: thread already exists: {thread_id}; choose a new ID or resume it",
                        file=sys.stderr,
                    )
                    return 2
                graph_input = {
                    "task": args.task,
                    "loop_step": 0,
                    "max_loop_steps": max(1, args.max_loops),
                    "hitl_enabled": args.hitl,
                }

            print(f"Thread: {thread_id}")
            result = _run(agent, graph_input, config, stream=args.stream)
            snapshot = agent.get_state(config)
            if snapshot.interrupts:
                _print_status(snapshot, thread_id)
                print(
                    "Resume with:\n"
                    f"  uv run python scripts/run_agent.py --thread-id {thread_id} "
                    "--resume approve|deny"
                )
                return 3

            if not args.stream:
                for message in result.get("messages") or []:
                    print(_format_message(message))
            if result.get("suggestion"):
                print("\n=== Final summary ===\n")
                print(result["suggestion"])
            return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
