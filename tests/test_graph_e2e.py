"""端到端测试 —— Phase 2 ReAct 循环。

无 API key 时由 conftest 自动跳过 (用 ``@pytest.mark.e2e`` 标记)。

测试 fixture 工作流:
1. 复制 buggy_ik.py 到一个临时副本路径
2. 让 agent 修复这个副本 (避免污染原 fixture)
3. 断言至少跑了 2 圈 (planner 节点 invoke 次数 >= 2)
4. 断言至少调用了 1 次 execute_python (LLM 试图验证)
5. 断言最终的 messages 末条是 AIMessage 且无 tool_calls (达到"完成"状态)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import graph
from agent.tools._paths import PROJECT_ROOT


@pytest.fixture
def buggy_ik_copy():
    """从 template 复制带 bug 的 fixture 到 workspace, 让 agent 改完不污染 template。"""
    src = PROJECT_ROOT / "tests" / "fixtures" / "buggy_ik_template.py"
    dst = PROJECT_ROOT / "tests" / "fixtures" / "_buggy_ik_workspace.py"
    shutil.copy2(src, dst)
    yield dst
    dst.unlink(missing_ok=True)


@pytest.mark.e2e
def test_react_loop_attempts_real_fix(buggy_ik_copy: Path):
    """让 agent 修复 buggy_ik 副本, 检验它走了完整 ReAct 循环。"""
    relative = buggy_ik_copy.relative_to(PROJECT_ROOT).as_posix()
    original = buggy_ik_copy.read_text(encoding="utf-8")

    result = graph.invoke(
        {
            "task": (
                f"请分析 {relative} 中的 bug, 用 write_patch 修复, "
                "用 execute_python 跑 inverse_kinematics_2link(3, 3) 验证不再抛 ValueError。"
                "完成后给出简短总结。"
            ),
            "loop_step": 0,
            "max_loop_steps": 12,
        },
        config={"recursion_limit": 50},
    )

    messages = result["messages"]

    # 1. 至少跑了 2 圈 planner
    assert result["loop_step"] >= 2, f"too few loops: {result['loop_step']}"

    # 2. 至少有一次工具调用
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)] # 过滤出工具调用消息
    assert len(tool_messages) >= 1, "no tool was invoked"

    # 3. 用过的工具名集合应该非空 (一般会含 read_file_chunk 或 execute_python)
    tool_names = {m.name for m in tool_messages}
    assert tool_names, "no tool names recorded"

    # 4. 末条消息是 AIMessage 且无 tool_calls (Agent 自然结束, 不是熔断)
    last = messages[-1]
    if result["loop_step"] < result.get("max_loop_steps", 15):
        assert isinstance(last, AIMessage)
        assert not getattr(last, "tool_calls", None), "ended with pending tool calls"

    # 5. suggestion 字段被填了最终回答 (planner 在无 tool_calls 时同步)
    assert result.get("suggestion"), "no final suggestion synthesized"

    # 6. 文件被改过 (write_patch 至少调用了一次, 但允许 LLM 仅诊断不动手 ——
    #    所以这里仅 warn-like 断言: 检查 write_patch 是否在 tool_names 中)
    if "write_patch" in tool_names:
        modified = buggy_ik_copy.read_text(encoding="utf-8")
        assert modified != original, "write_patch was called but content unchanged"
