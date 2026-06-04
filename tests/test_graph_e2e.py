"""端到端测试 —— 实跑 graph.invoke, 需要有效 API key。

无 key 时 conftest 自动跳过。
"""

from __future__ import annotations

import pytest

from agent.graph import graph
from agent.tools.file_ops import read_file


def test_read_file_basic():
    """文件读取工具的纯单测, 不依赖网络/LLM。"""
    content = read_file("tests/fixtures/buggy_ik.py")
    assert "inverse_kinematics_2link" in content
    assert "math.acos" in content


def test_read_file_rejects_outside_root():
    with pytest.raises(ValueError):
        read_file("../../../etc/passwd")


@pytest.mark.e2e
def test_linear_flow_finds_file_and_suggests():
    """跑通 planner → reader → suggester, 断言三个关键字段非空。"""
    result = graph.invoke(
        {
            "task": "请分析并修复 tests/fixtures/buggy_ik.py 中的 bug",
            "loop_step": 0,
            "current_file": None,
            "file_content": None,
            "suggestion": None,
        }
    )

    assert result["current_file"] is not None
    assert "buggy_ik" in result["current_file"]
    assert result["file_content"] is not None and len(result["file_content"]) > 0
    assert result["suggestion"] is not None and len(result["suggestion"]) > 50
    # 消息历史应该至少有 3 条 (planner, reader, suggester 各一条)
    assert len(result["messages"]) >= 3
