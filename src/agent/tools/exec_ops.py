"""代码执行工具 —— 让 LLM 能运行 Python 验证修复。

**安全说明 (Phase 2 重要 caveat)**:
本沙盒只是 ``subprocess + tempdir`` 隔离, **不是** 真正的安全沙盒。
LLM 仍能读 ``import os; os.listdir("/")``, 写你的 home 目录文件等。
Phase 3 起会替换为 E2B Code Interpreter 真沙盒。
当前实现仅适合**学习场景下跑自己仓库内代码**。

实现要点:
- 用 ``sys.executable`` 复用当前 venv (LLM 能用项目依赖如 langchain/pydantic)。
- cwd 设为 tempdir, LLM 的临时文件写入隔离, 不污染仓库。
- ``PROJECT_ROOT`` 加入 PYTHONPATH, 让 ``import agent.xxx`` 工作。
- 输出硬截断, stdout 与 stderr 合并展示。
- 超时硬上限 120 秒, 防 LLM 写死循环卡死整个 graph。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from agent.tools._paths import PROJECT_ROOT

_MAX_TIMEOUT = 120
_MAX_OUTPUT_CHARS = 20_000


def _truncate(text: str, label: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    cut = text[:_MAX_OUTPUT_CHARS]
    omitted = len(text) - _MAX_OUTPUT_CHARS
    return f"{cut}\n... [{label} truncated, {omitted} more chars hidden]"


@tool
def execute_python(code: str, timeout: int = 30) -> str:
    """在隔离子进程中执行 Python 代码, 捕获 stdout/stderr 与退出码。

    工作目录是一次性临时目录; 当前 venv 的依赖可用; ``PROJECT_ROOT`` 在 PYTHONPATH 中,
    所以可以 ``from agent.xxx import ...`` 或 ``import tests.fixtures.buggy_ik``。

    **典型用法**:
    - 复现 bug: ``execute_python("from tests.fixtures.buggy_ik import inverse_kinematics_2link; print(inverse_kinematics_2link(3, 3))")``
    - 验证修复后单元测试: ``execute_python("import subprocess; print(subprocess.run(['pytest', '-q', 'tests/'], capture_output=True, text=True).stdout)")``

    Args:
        code: 要执行的 Python 代码 (字符串)。
        timeout: 超时秒数, 默认 30; 上限 120。
    Returns:
        格式化字符串, 含 returncode 与合并的 stdout/stderr (或 "ERROR: timed out")。
    """
    timeout = min(max(1, int(timeout)), _MAX_TIMEOUT)

    with tempfile.TemporaryDirectory(prefix="agent_exec_") as workdir:
        script_path = Path(workdir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        # PYTHONPATH 加入项目 src/ 让 ``import agent.xxx`` 工作,
        # 加入 PROJECT_ROOT 让 ``import tests.fixtures.xxx`` 工作。
        extra_paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))
        env["PYTHONUTF8"] = "1"  # 子进程内统一 UTF-8
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=workdir,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: execution timed out after {timeout}s (code likely has an infinite loop)"
        except OSError as e:
            return f"ERROR: failed to launch subprocess: {e}"

    parts = [f"exit_code: {proc.returncode}"]
    if proc.stdout:
        parts.append("--- stdout ---")
        parts.append(_truncate(proc.stdout, "stdout"))
    if proc.stderr:
        parts.append("--- stderr ---")
        parts.append(_truncate(proc.stderr, "stderr"))
    if not proc.stdout and not proc.stderr:
        parts.append("(no output)")

    return "\n".join(parts)
