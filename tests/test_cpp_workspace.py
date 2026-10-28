from __future__ import annotations

import json
import sys

import pytest

from agent.cpp import LspClient, compilation_database, configure_build, semantic_query
from agent.workspace import Workspace, use_workspace


def test_compilation_database_accepts_arguments_and_command(tmp_path):
    database = tmp_path / "compile_commands.json"
    entries = [{"file": "main.cpp", "directory": str(tmp_path), "arguments": ["c++", "main.cpp"]},
               {"file": "other.cpp", "directory": str(tmp_path), "command": "c++ other.cpp"}]
    database.write_text(json.dumps(entries))
    assert compilation_database(database) == entries
    database.write_text(json.dumps([{"file": "bad.cpp", "directory": ".", "arguments": "c++"}]))
    with pytest.raises(ValueError, match="arguments"):
        compilation_database(database)


def test_cpp_requires_explicit_execution_capability(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "CMakeLists.txt").write_text("project(demo)")
    workspace = Workspace.create(root, tmp_path / "state")
    with use_workspace(workspace):
        with pytest.raises(RuntimeError, match="execution"):
            configure_build()
        with pytest.raises(ValueError, match="target"):
            configure_build("all; touch bad")
        (root / "main.cpp").write_text("int a;")
        with pytest.raises(ValueError, match="1-based"):
            semantic_query("main.cpp", "hover", 0, 1)


def test_lsp_protocol_handles_unicode_diagnostics_and_server_errors(tmp_path):
    server = tmp_path / "server.py"
    server.write_text('''import json,sys
def send(value):
    data=json.dumps(value, ensure_ascii=False).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\\r\\n\\r\\n".encode()+data)
    sys.stdout.buffer.flush()
while True:
    line=sys.stdin.buffer.readline()
    if not line: break
    length=int(line.split(b":")[1])
    sys.stdin.buffer.readline()
    message=json.loads(sys.stdin.buffer.read(length))
    if "id" not in message: continue
    send({"method":"textDocument/publishDiagnostics","params":{"uri":"file:///test.cpp","diagnostics":[{"message":"类型错误"}]}})
    if message["method"]=="fail": send({"id":message["id"],"error":{"message":"bad request"}})
    else: send({"id":message["id"],"result":{"contents":"double 值"}})
''', encoding="utf-8")
    client = LspClient([sys.executable, str(server)], timeout=5)
    try:
        assert client.request("hover", {})["contents"] == "double 值"
        assert client.diagnostics["file:///test.cpp"][0]["message"] == "类型错误"
        with pytest.raises(RuntimeError, match="bad request"):
            client.request("fail", {})
    finally:
        client.close()


def test_lsp_timeout_terminates_child():
    client = LspClient([sys.executable, "-c", "import time;time.sleep(30)"], timeout=0.15)
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            client.request("initialize", {})
    finally:
        client.close()
    assert client.process.poll() is not None


def test_lsp_stalled_reader_cannot_block_request_write():
    client = LspClient([sys.executable, "-c", "import time;time.sleep(30)"], timeout=0.2)
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            client.request("large", {"text": "x" * 500000})
    finally:
        client.close()
    assert client.process.poll() is not None
