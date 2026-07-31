"""命令行 demo —— 跑 Phase 2 ReAct 循环, 打印每圈推理 + 工具调用。

用法:
    uv run python scripts/run_demo.py
    uv run python scripts/run_demo.py "<task>"
    uv run python scripts/run_demo.py "<task>" --stream

``--stream`` 模式下逐节点打印, 适合实时观察 agent 思考过程。
"""

from __future__ import annotations

import argparse
import os
import sys

# Windows PowerShell 默认 GBK, Python 输出 UTF-8 会乱码 ——
# 在打印前把 stdout/stderr 强制切到 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import shutil

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph import graph
from agent.tools._paths import PROJECT_ROOT

_TEMPLATE = PROJECT_ROOT / "tests" / "fixtures" / "buggy_ik_template.py"
_WORKSPACE = PROJECT_ROOT / "tests" / "fixtures" / "_buggy_ik_workspace.py"

_DEFAULT_TASK = (
    "请分析 tests/fixtures/_buggy_ik_workspace.py 中的 bug, "
    "用 write_patch 修复, 用 execute_python 跑 "
    "`inverse_kinematics_2link(1.5, 0.5)` 与 `inverse_kinematics_2link(3, 3)` 验证 "
    "(前者返回合理弧度, 后者抛清晰的 ValueError 而非 math domain error)。"
    "完成后给出简短总结。"
)


def _reset_workspace() -> None:
    """每次 demo 启动时, 从 template 拷出一份新鲜的 buggy 副本作为 workspace。"""
    if not _TEMPLATE.exists():
        raise FileNotFoundError(f"template missing: {_TEMPLATE}")
    shutil.copy2(_TEMPLATE, _WORKSPACE)


def _format_message(msg) -> str:
    """把一条 LangChain 消息渲染成可读多行字符串。"""
    cls = type(msg).__name__
    if isinstance(msg, SystemMessage):
        return "[SYSTEM] (omitted)"
    if isinstance(msg, HumanMessage):
        return f"[HUMAN] {msg.content}"
    if isinstance(msg, AIMessage):
        parts = ["[AI]"]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if content.strip():
            parts.append(content.strip())
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            # 给 args 一个紧凑预览, 长字符串截短
            preview = {k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                       for k, v in (args or {}).items()}
            parts.append(f"  → tool_call: {name}({preview})")
        return "\n".join(parts)
    if isinstance(msg, ToolMessage):
        body = msg.content if isinstance(msg.content, str) else str(msg.content)
        if len(body) > 800:
            body = body[:800] + f"\n... [+{len(msg.content) - 800} chars]"
        return f"[TOOL:{msg.name}]\n{body}"
    return f"[{cls}] {msg}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic-Robo-SWE Phase 2 ReAct demo")
    parser.add_argument("task", nargs="?", default=_DEFAULT_TASK)
    parser.add_argument("--stream", action="store_true",
                        help="逐节点 print, 实时观察 ReAct 循环 (推荐学习时用)")
    parser.add_argument("--max-loops", type=int, default=12,
                        help="最大循环步数熔断, 默认 12")
    args = parser.parse_args()

    # 从 template 重置 workspace, 确保每次跑 demo 时 agent 都遇到完整的 4 个 bug
    _reset_workspace()

    provider = os.getenv("MODEL_PROVIDER", "anthropic")
    print(f"=== Agentic-Robo-SWE Phase 2 / Provider={provider} / max_loops={args.max_loops} ===")
    print(f"Workspace reset from template: {_WORKSPACE.relative_to(PROJECT_ROOT)}")
    print(f"Task: {args.task}\n")

    initial = {
        "task": args.task,
        "loop_step": 0,
        "max_loop_steps": args.max_loops,
    }

    if args.stream:
        printed = 0
        for chunk in graph.stream(initial, config={"recursion_limit": 50}, stream_mode="values"):
            messages = chunk.get("messages") or []
            # 只打印新增的 message (上一次 print 之后追加的)
            for msg in messages[printed:]:
                print(_format_message(msg))
                print()
            printed = len(messages)
        # 最后取一次终态 (stream_mode="values" 已发完, printed == 完整长度)
        print(f"\n--- 终态 loop_step={chunk.get('loop_step')} ---")
        if chunk.get("suggestion"):
            print("\n=== 最终总结 ===\n")
            print(chunk["suggestion"])
    else:
        result = graph.invoke(initial, config={"recursion_limit": 50})
        print(f"--- 循环步数: {result.get('loop_step')} ---\n")
        for msg in result.get("messages", []):
            print(_format_message(msg))
            print()
        if result.get("suggestion"):
            print("=== 最终总结 ===\n")
            print(result["suggestion"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
