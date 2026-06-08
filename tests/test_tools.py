"""Phase 2 工具的单元测试 —— 不依赖 LLM, 可离线跑。

每个工具走 ``.invoke()`` 接口 (LangChain @tool 装饰后的标准入口) 来模拟 ToolNode 调用。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import (
    execute_python,
    grep_codebase,
    list_dir,
    read_file_chunk,
    write_patch,
)
from agent.tools._paths import PROJECT_ROOT


# —— read_file_chunk ——


def test_read_file_chunk_basic():
    out = read_file_chunk.invoke({"path": "tests/fixtures/buggy_ik.py"})
    assert "inverse_kinematics_2link" in out
    assert "math.acos" in out
    # 行号前缀格式 (4 位宽度对齐)
    assert "|" in out


def test_read_file_chunk_offset_limit():
    out = read_file_chunk.invoke({"path": "tests/fixtures/buggy_ik.py", "offset": 0, "limit": 5})
    body_lines = [ln for ln in out.splitlines() if "|" in ln]
    assert len(body_lines) <= 5


def test_read_file_chunk_outside_root():
    out = read_file_chunk.invoke({"path": "../../../etc/passwd"})
    assert out.startswith("ERROR")


def test_read_file_chunk_not_found():
    out = read_file_chunk.invoke({"path": "does/not/exist.py"})
    assert out.startswith("ERROR")
    assert "not found" in out.lower()


# —— grep_codebase ——


def test_grep_codebase_finds_known_symbol():
    out = grep_codebase.invoke({"pattern": "inverse_kinematics_2link", "glob": "tests/**/*.py"})
    assert "buggy_ik.py" in out
    # 至少一行带行号
    assert ":" in out


def test_grep_codebase_no_match():
    # 用变量拼接避免测试文件本身被 grep 匹配 (自指 bug 教训!)
    pattern = "zzz" + "_unlikely_" + "absent_token_" + "QQQ"
    out = grep_codebase.invoke({"pattern": pattern, "glob": "**/*.py"})
    assert "<no matches" in out


def test_grep_codebase_invalid_regex():
    out = grep_codebase.invoke({"pattern": "[unclosed"})
    assert out.startswith("ERROR")


# —— list_dir ——


def test_list_dir_root():
    out = list_dir.invoke({"path": "."})
    assert "src/" in out or "src" in out
    assert "tests/" in out or "tests" in out


def test_list_dir_outside_root():
    out = list_dir.invoke({"path": "../../.."})
    assert out.startswith("ERROR")


# —— write_patch ——


@pytest.fixture
def scratch_file(tmp_path: Path):
    """在仓库内放一个临时文件 (write_patch 只允许仓库内路径)。"""
    scratch = PROJECT_ROOT / "tests" / "_scratch_for_write_patch.txt"
    scratch.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    yield scratch
    scratch.unlink(missing_ok=True)


def test_write_patch_success(scratch_file: Path):
    out = write_patch.invoke({
        "path": "tests/_scratch_for_write_patch.txt",
        "old_string": "beta",
        "new_string": "BETA",
    })
    assert out.startswith("OK")
    assert scratch_file.read_text() == "alpha\nBETA\ngamma\n"


def test_write_patch_not_unique(scratch_file: Path):
    scratch_file.write_text("dup\ndup\ndup\n", encoding="utf-8")
    out = write_patch.invoke({
        "path": "tests/_scratch_for_write_patch.txt",
        "old_string": "dup",
        "new_string": "X",
    })
    assert out.startswith("ERROR")
    assert "3" in out  # 报告了出现次数


def test_write_patch_not_found(scratch_file: Path):
    out = write_patch.invoke({
        "path": "tests/_scratch_for_write_patch.txt",
        "old_string": "nonexistent_string",
        "new_string": "X",
    })
    assert out.startswith("ERROR")


def test_write_patch_refuses_outside_root():
    out = write_patch.invoke({
        "path": "../../../tmp/evil.txt",
        "old_string": "x",
        "new_string": "y",
    })
    assert out.startswith("ERROR")


# —— execute_python ——


def test_execute_python_hello():
    out = execute_python.invoke({"code": "print('hello', 1 + 2)"})
    assert "exit_code: 0" in out
    assert "hello 3" in out


def test_execute_python_imports_project_module():
    """子进程能 import 项目代码 (因为 PYTHONPATH 注入了 PROJECT_ROOT/src)。"""
    out = execute_python.invoke({
        "code": "from agent.state import AgentState; print('AgentState fields:', list(AgentState.__annotations__))",
    })
    assert "exit_code: 0" in out
    assert "messages" in out


def test_execute_python_imports_fixture():
    """子进程能 import tests.fixtures (PROJECT_ROOT 也在 PYTHONPATH)。"""
    out = execute_python.invoke({
        "code": (
            "from tests.fixtures.buggy_ik import inverse_kinematics_2link\n"
            "print(inverse_kinematics_2link(1.0, 0.5))"
        ),
    })
    assert "exit_code: 0" in out


def test_execute_python_timeout():
    out = execute_python.invoke({"code": "while True: pass", "timeout": 2})
    assert "timed out" in out.lower()


def test_execute_python_captures_error():
    out = execute_python.invoke({"code": "raise RuntimeError('boom')"})
    assert "exit_code: 1" in out
    assert "RuntimeError" in out
    assert "boom" in out
