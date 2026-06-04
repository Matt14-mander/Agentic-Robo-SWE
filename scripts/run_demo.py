"""命令行 demo —— 不进 Studio, 直接跑 graph.invoke 并打印结果。

用法:
    uv run python scripts/run_demo.py "修复 tests/fixtures/buggy_ik.py 中的 bug"
"""

from __future__ import annotations

import argparse
import os
import sys

# Windows PowerShell 默认 GBK 编码, Python 输出 UTF-8 会乱码 ——
# 在打印前把 stdout/stderr 强制切到 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agent.graph import graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic-Robo-SWE Phase 1 demo")
    parser.add_argument(
        "task",
        nargs="?",
        default="请分析并修复 tests/fixtures/buggy_ik.py 中的 bug",
        help="用户指令 (默认: 修复内置 buggy_ik fixture)",
    )
    args = parser.parse_args()

    provider = os.getenv("MODEL_PROVIDER", "anthropic")
    print(f"=== Agentic-Robo-SWE / Provider={provider} ===")
    print(f"Task: {args.task}\n")

    result = graph.invoke(
        {
            "task": args.task,
            "loop_step": 0,
            "current_file": None,
            "file_content": None,
            "suggestion": None,
        }
    )

    print("--- 执行轨迹 ---")
    for msg in result.get("messages", []):
        content = msg.content if hasattr(msg, "content") else str(msg)
        print(f"\n{content}")

    print("\n--- 最终建议 ---\n")
    print(result.get("suggestion") or "<empty>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
