"""Phase 2.5 双后端测试 —— 验证 local 与 e2b 在相同输入下行为一致。

对照测试设计:
- 同样的 code, local 与 e2b 都跑一遍, 断言输出格式与关键内容一致。
- e2b 用例自动跳过条件: 缺 E2B_API_KEY 或 e2b-code-interpreter 未装。
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agent.tools import execute_python
from agent.tools import exec_ops


def _e2b_available() -> bool:
    if not os.getenv("E2B_API_KEY"):
        return False
    try:
        importlib.import_module("e2b_code_interpreter")
        return True
    except ImportError:
        return False


requires_e2b = pytest.mark.skipif(
    not _e2b_available(),
    reason="E2B_API_KEY missing or e2b-code-interpreter not installed; skipping e2b backend tests.",
)


@pytest.fixture
def backend(request, monkeypatch):
    """参数化 fixture: 把 EXECUTOR_BACKEND 临时设为指定后端, 测试结束自动还原。"""
    backend_name = request.param
    monkeypatch.setenv("EXECUTOR_BACKEND", backend_name)
    return backend_name


# ─── 在两种后端上跑同一组 case ───


@pytest.mark.parametrize("backend", ["local", pytest.param("e2b", marks=[requires_e2b, pytest.mark.e2e])], indirect=True)
def test_print_hello(backend):
    out = execute_python.invoke({"code": "print('hello', 1 + 2)"})
    assert "exit_code: 0" in out
    assert "hello 3" in out


@pytest.mark.parametrize("backend", ["local", pytest.param("e2b", marks=[requires_e2b, pytest.mark.e2e])], indirect=True)
def test_stderr_capture(backend):
    out = execute_python.invoke({"code": "import sys; sys.stderr.write('warned\\n')"})
    assert "warned" in out


@pytest.mark.parametrize("backend", ["local", pytest.param("e2b", marks=[requires_e2b, pytest.mark.e2e])], indirect=True)
def test_exception_traceback(backend):
    out = execute_python.invoke({"code": "raise RuntimeError('boom')"})
    assert "RuntimeError" in out
    assert "boom" in out


@pytest.mark.parametrize("backend", ["local", pytest.param("e2b", marks=[requires_e2b, pytest.mark.e2e])], indirect=True)
def test_project_module_import(backend):
    """两后端都能 import 仅依赖标准库的项目模块。

    此用例验证源码上传与 ``sys.path`` 注入，不假设 E2B 基础镜像已经安装
    项目的第三方依赖。
    """
    out = execute_python.invoke({
        "code": (
            "from tests.fixtures.buggy_ik import inverse_kinematics_2link\n"
            "print('OK', callable(inverse_kinematics_2link))"
        ),
    })
    assert "exit_code: 0" in out
    assert "OK True" in out


# ─── 后端无关的稳健性测试 ───


def test_unknown_backend(monkeypatch):
    monkeypatch.setenv("EXECUTOR_BACKEND", "wormhole")
    out = execute_python.invoke({"code": "print(1)"})
    assert out.startswith("ERROR")
    assert "wormhole" in out


def test_e2b_without_key(monkeypatch):
    """显式选 e2b 后端但缺 key, 应给清晰错误指南。"""
    monkeypatch.setenv("EXECUTOR_BACKEND", "e2b")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    out = execute_python.invoke({"code": "print(1)"})
    assert out.startswith("ERROR")
    assert "E2B_API_KEY" in out
    assert "e2b.dev" in out  # 指向注册页


class _FakeFilesystem:
    def __init__(self):
        self.entries = None
        self.directories = []

    def make_dir(self, path):
        self.directories.append(path)

    def write_files(self, entries):
        self.entries = entries


class _FakeSandbox:
    def __init__(self, execution=None, run_error=None):
        self.execution = execution
        self.run_error = run_error
        self.files = _FakeFilesystem()
        self.run_args = None
        self.killed = False

    def run_code(self, code, **kwargs):
        self.run_args = (code, kwargs)
        if self.run_error:
            raise self.run_error
        return self.execution

    def kill(self):
        self.killed = True


def _install_fake_e2b(monkeypatch, sandbox):
    module = ModuleType("e2b_code_interpreter")

    class Sandbox:
        @classmethod
        def create(cls, **kwargs):
            sandbox.create_kwargs = kwargs
            return sandbox

    module.Sandbox = Sandbox
    monkeypatch.setitem(sys.modules, "e2b_code_interpreter", module)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    monkeypatch.setenv("EXECUTOR_BACKEND", "e2b")


def test_e2b_mock_reports_execution_error_and_kills_sandbox(monkeypatch):
    """云端异常必须包含 traceback, 且成功创建的 sandbox 一定被销毁。"""
    error = SimpleNamespace(
        name="RuntimeError",
        value="boom",
        traceback="Traceback (most recent call last): ... RuntimeError: boom",
    )
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["before failure\n"], stderr=[]),
        error=error,
    )
    sandbox = _FakeSandbox(execution=execution)
    _install_fake_e2b(monkeypatch, sandbox)

    out = execute_python.invoke({"code": "raise RuntimeError('boom')", "timeout": 7})

    assert "exit_code: 1" in out
    assert "[E2B error] RuntimeError: boom" in out
    assert "Traceback" in out
    assert sandbox.killed is True
    assert sandbox.create_kwargs == {"api_key": "test-key", "timeout": 300}
    assert sandbox.run_args[1] == {"language": "python", "timeout": 7.0}


def test_e2b_mock_uploads_project_files_via_public_api(monkeypatch):
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["OK\n"], stderr=[]),
        error=None,
    )
    sandbox = _FakeSandbox(execution=execution)
    _install_fake_e2b(monkeypatch, sandbox)
    monkeypatch.setattr(
        exec_ops,
        "_collect_project_files",
        lambda: [("/home/user/src/agent/example.py", "VALUE = 1\n")],
    )

    out = execute_python.invoke({"code": "from agent.example import VALUE\nprint('OK')"})

    assert "exit_code: 0" in out
    assert sandbox.files.entries == [
        {"path": "/home/user/src/agent/example.py", "data": "VALUE = 1\n"}
    ]
    assert sandbox.files.directories == ["/home/user/src/agent"]
    assert sandbox.killed is True


def test_e2b_mock_kills_sandbox_when_run_code_fails(monkeypatch):
    sandbox = _FakeSandbox(run_error=TimeoutError("request timed out"))
    _install_fake_e2b(monkeypatch, sandbox)

    out = execute_python.invoke({"code": "print('never completes')"})

    assert out.startswith("ERROR: E2B run_code failed: TimeoutError")
    assert sandbox.killed is True
