"""代码执行工具 —— Phase 2.5 起支持两套后端, 由 ``EXECUTOR_BACKEND`` 切换。

## 后端对比

| 维度 | local (subprocess + tempdir) | e2b (Code Interpreter) |
|------|-------------------------------|------------------------|
| 隔离强度 | 仅 cwd 隔离, **不安全** | Firecracker microVM, 真隔离 |
| 启动开销 | ~0.5s | ~3-5s (云端 sandbox 冷启动) |
| 依赖 | Python 标准库 | e2b-code-interpreter + 网络 |
| 凭据 | 无 | E2B_API_KEY |
| 项目源码可用性 | PYTHONPATH 注入即可 import | 上传 src/+tests/ 后注入 sys.path |
| 第三方依赖 | 复用当前 venv | 需 E2B 镜像预装或由调用代码安装 |
| 适合场景 | 学习/快速迭代 | 生产/不可信代码 |

## 设计要点

1. **统一入口** ``execute_python`` (@tool) 对 LLM 完全透明; backend 切换不需要改 prompt。
2. **失败时返回 ERROR 字符串而非抛异常** —— ToolNode 收到的就是 LLM 能读懂的反馈。
3. **E2B 后端按需上传** —— 只在 code 显式 import 项目模块时才上传 src/+tests/,
   单跑 ``print(1+1)`` 不走上传, 减少 80% 的冷启动延迟。
4. **每次调用一个全新 sandbox** —— 语义与 local subprocess 一致 (无状态串扰),
   便于 LLM 推理。后续可在 graph 层做 sandbox 复用优化。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from langchain_core.tools import tool

from agent.tools._paths import PROJECT_ROOT

_MAX_TIMEOUT = 120
_MAX_OUTPUT_CHARS = 20_000

# 检测代码是否需要项目文件 (用 import / from 语句中是否引用 agent. / tests. 命名空间)
_PROJECT_IMPORT_RE = re.compile(
    r"^(?:from|import)\s+(agent|tests)(?:[.\s]|$)", re.MULTILINE
)


def _truncate(text: str, label: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    cut = text[:_MAX_OUTPUT_CHARS]
    omitted = len(text) - _MAX_OUTPUT_CHARS
    return f"{cut}\n... [{label} truncated, {omitted} more chars hidden]"


def _format_output(returncode: int, stdout: str, stderr: str) -> str:
    """两后端共用的输出渲染格式。"""
    parts = [f"exit_code: {returncode}"]
    if stdout:
        parts.append("--- stdout ---")
        parts.append(_truncate(stdout, "stdout"))
    if stderr:
        parts.append("--- stderr ---")
        parts.append(_truncate(stderr, "stderr"))
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)


# ─────────────────────────────── local backend ───────────────────────────────


def _execute_local(code: str, timeout: int) -> str:
    """Phase 2 原版: subprocess + tempdir 隔离。

    工作目录是一次性临时目录; 当前 venv 的依赖可用;
    ``PROJECT_ROOT`` 与 ``PROJECT_ROOT/src`` 在 PYTHONPATH 中,
    可以 ``from agent.xxx import ...`` 或 ``import tests.fixtures.xxx``。
    """
    with tempfile.TemporaryDirectory(prefix="agent_exec_") as workdir:
        script_path = Path(workdir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        extra_paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))
        env["PYTHONUTF8"] = "1"
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

    return _format_output(proc.returncode, proc.stdout, proc.stderr)


# ─────────────────────────────── e2b backend ───────────────────────────────


def _collect_project_files() -> list[tuple[str, str]]:
    """收集 src/ 与 tests/ 下的 .py 文件 (path_in_sandbox, content) 对。

    跳过 .venv / __pycache__ / 隐藏目录, 防止 noise 上传。
    """
    pairs: list[tuple[str, str]] = []
    skip = {".venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
            ".ruff_cache", "node_modules", "dist", "build"}
    for sub in ("src", "tests"):
        root = PROJECT_ROOT / sub
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            if any(part in skip or part.startswith(".") for part in f.parts):
                continue
            try:
                content = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = f.relative_to(PROJECT_ROOT).as_posix()
            # E2B sandbox 默认 home 是 /home/user/, 我们映射到 /home/user/<rel>
            pairs.append((f"/home/user/{rel}", content))
    return pairs


def _execute_e2b(code: str, timeout: int) -> str:
    """E2B Code Interpreter 后端: 云端 Firecracker microVM 隔离。

    实现要点:
    - 每次创建新 sandbox, 用完即销毁 (匹配 local 的"无状态"语义)。
    - 只在 code 需要 import 项目模块时才上传 src/+tests/ (减少冷启动延迟)。
    - 在 code 前注入 sys.path, 让 ``import agent.xxx`` / ``import tests.xxx`` 工作。
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        return (
            "ERROR: E2B_API_KEY is not set. "
            "Sign up at https://e2b.dev to get a key, then set E2B_API_KEY in .env. "
            "Or switch back to local by setting EXECUTOR_BACKEND=local."
        )

    # 延迟 import: 用户没装 sandbox group 时只在调用时报清晰错误
    try:
        from e2b_code_interpreter import Sandbox  # type: ignore[import-not-found]
    except ImportError:
        return "ERROR: e2b-code-interpreter not installed. Run: uv sync --extra sandbox"

    # 仅在代码引用项目模块时才付出上传开销
    needs_project = bool(_PROJECT_IMPORT_RE.search(code))

    # sandbox.timeout 是"沙盒生命周期上限", 远大于单次执行的 timeout
    try:
        sbx = Sandbox.create(api_key=api_key, timeout=300)
    except Exception as e:
        return f"ERROR: failed to create E2B sandbox: {type(e).__name__}: {e}"

    try:
        if needs_project:
            # E2B v2 的公开 API 接受 path/data 字典。避免依赖 SDK 内部的
            # ``e2b.sandbox.filesystem.filesystem.WriteEntry`` 导入路径。
            try:
                files = _collect_project_files()
                parent_dirs = sorted({str(PurePosixPath(path).parent) for path, _ in files})
                for parent_dir in parent_dirs:
                    sbx.files.make_dir(parent_dir)
                entries = [
                    {"path": path, "data": content}
                    for path, content in files
                ]
                sbx.files.write_files(entries)
            except Exception as e:
                return f"ERROR: failed to upload project files: {type(e).__name__}: {e}"

        # 注入 sys.path, 让 import 工作。前置 \n 防止用户 code 顶行带 docstring 时被吞。
        prelude = (
            "import sys\n"
            "for _p in ('/home/user', '/home/user/src'):\n"
            "    if _p not in sys.path: sys.path.insert(0, _p)\n"
        )
        wrapped = prelude + code

        try:
            execution = sbx.run_code(wrapped, language="python", timeout=float(timeout))
        except Exception as e:
            return f"ERROR: E2B run_code failed: {type(e).__name__}: {e}"
    finally:
        # 销毁沙盒, 避免资源占用 (即使出错也要清理)
        try:
            sbx.kill()
        except Exception:
            pass

    # E2B Execution 的 logs 是 list[str] (每行一个元素)
    stdout = "".join(execution.logs.stdout)
    stderr = "".join(execution.logs.stderr)
    returncode = 1 if execution.error else 0

    # E2B 把 Python 异常拆成 error.name + error.value + error.traceback;
    # 拼到 stderr 末尾, 让 LLM 看到完整 traceback。
    if execution.error:
        stderr += (
            f"\n[E2B error] {execution.error.name}: {execution.error.value}\n"
            f"{execution.error.traceback}"
        )

    return _format_output(returncode, stdout, stderr)


# ─────────────────────────────── dispatcher (tool entry) ──────────────────────


def _resolve_backend() -> str:
    """读取并归一化 EXECUTOR_BACKEND env var。"""
    return (os.getenv("EXECUTOR_BACKEND") or "local").strip().lower()


@tool
def execute_python(code: str, timeout: int = 30) -> str:
    """在沙盒中执行 Python 代码, 捕获 stdout/stderr 与退出码。

    后端由 ``EXECUTOR_BACKEND`` 环境变量决定:
    - ``local`` (默认) - subprocess + tempdir, 快但不隔离
    - ``e2b`` - E2B Code Interpreter 云沙盒, 真隔离, 需要 E2B_API_KEY

    项目源码可用: local 走 PYTHONPATH, e2b 走文件上传 + sys.path 注入。
    E2B 不会自动安装项目的第三方依赖; 导入的模块若依赖额外包, 需确保沙盒镜像已包含
    这些依赖, 或先在沙盒中安装。

    Args:
        code: 要执行的 Python 代码 (字符串)。
        timeout: 单次执行超时秒数, 默认 30; 上限 120。
    Returns:
        含 ``exit_code:`` 行的格式化字符串, 以及合并的 stdout/stderr。
    """
    timeout = min(max(1, int(timeout)), _MAX_TIMEOUT)

    backend = _resolve_backend()
    if backend == "e2b":
        return _execute_e2b(code, timeout)
    if backend == "local":
        return _execute_local(code, timeout)

    return (
        f"ERROR: unknown EXECUTOR_BACKEND={backend!r}. "
        "Expected one of: local, e2b."
    )
