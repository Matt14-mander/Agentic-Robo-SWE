"""CMake/compilation-database and bounded clangd LSP access for target repos."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import quote

from agent.containers import (
    clean_environment, container_command, remove_container, run_bounded, run_container, stop_process,
)
from agent.workspace import Workspace, current_workspace, exported_source


def require_workspace() -> Workspace:
    workspace = current_workspace()
    if workspace is None:
        raise ValueError("C++ repository tools require an explicit --workspace")
    return workspace


def build_root(workspace: Workspace) -> Path:
    path = workspace.artifacts / "cpp-build"
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Build artifact directory must not be a link")
    if workspace.execution_image and os.name != "nt":
        # Public build scratch only; private oracle/state is never mounted here.
        path.chmod(0o777)
    return path


def compilation_database(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or path.stat().st_size > 10_000_000:
        raise ValueError("Unsafe or oversized compilation database")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) > 20000:
        raise ValueError("Compilation database must be a bounded array")
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("file", "directory")):
            raise ValueError("Compilation entry needs file and directory")
        args = row.get("arguments")
        if args is not None:
            if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args):
                raise ValueError("Compilation arguments must be a non-empty string array")
        elif not isinstance(row.get("command"), str) or not row["command"]:
            raise ValueError("Compilation entry needs arguments or command")
    return rows


def configure_build(target: str = "all", configure_only: bool = False) -> dict[str, object]:
    workspace = require_workspace()
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", target):
        raise ValueError("Invalid CMake target")
    if not workspace.resolve("CMakeLists.txt").is_file():
        raise ValueError("Workspace root has no CMakeLists.txt")
    build = build_root(workspace)
    identity = {"root": str(workspace.root), "image": workspace.execution_image,
                "trusted_local": workspace.trusted_local}
    marker = workspace.artifacts / "cpp-binding.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Build backend changed; select a fresh --artifact-root")
    marker.write_text(json.dumps(identity), encoding="utf-8")

    def execute(root: str, output: str, runner):
        configured = runner(["cmake", "-S", root, "-B", output, "-G", "Ninja",
                             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_BUILD_TYPE=Debug"])
        result = configured
        stage = "configure"
        if configured.returncode == 0 and not configure_only:
            stage = "build"
            result = runner(["cmake", "--build", output, "--target", target, "--parallel", "2"])
        return {"passed": result.returncode == 0, "stage": stage,
                "exit_code": result.returncode, "stdout": result.stdout[:20000],
                "stderr": result.stderr[:20000],
                "compilation_database": str(build / "compile_commands.json")}

    if workspace.execution_image:
        with exported_source(workspace.root) as snapshot:
            return execute("/workspace", "/build", lambda command: run_container(
                workspace.execution_image or "", command,
                [(snapshot, "/workspace", False), (build, "/build", True)], timeout=180))
    if workspace.trusted_local:
        from agent.domain.toolchain import _find_cmake, _find_ninja
        cmake, ninja = _find_cmake(), _find_ninja()
        if not cmake or not ninja:
            raise RuntimeError("CMake and Ninja capabilities are required")

        def native(command):
            command[0] = cmake
            if "-S" in command:
                command.append(f"-DCMAKE_MAKE_PROGRAM={ninja}")
            return run_bounded(command, cwd=workspace.root, timeout=180)

        return execute(str(workspace.root), str(build), native)
    raise RuntimeError("C++ execution needs --execution-image or explicit --trusted-local")


class LspClient:
    """One bounded session per query; stdout is protocol, never compiler logs."""

    def __init__(self, command: list[str], timeout: float = 45):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, env=clean_environment(),
                                        start_new_session=os.name != "nt")
        self.deadline = time.monotonic() + timeout
        self.messages: queue.Queue = queue.Queue(maxsize=256)
        self.diagnostics: dict[str, Any] = {}
        self.sequence = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            assert self.process.stdout is not None
            while True:
                headers = {}
                while True:
                    line = self.process.stdout.readline(8192)
                    if not line:
                        raise RuntimeError("clangd closed its protocol stream")
                    if line in (b"\r\n", b"\n"):
                        break
                    key, value = line.decode("ascii").split(":", 1)
                    headers[key.lower()] = value.strip()
                    if len(headers) > 16:
                        raise RuntimeError("Invalid LSP headers")
                size = int(headers["content-length"])
                if not 0 < size <= 2_000_000:
                    raise RuntimeError("LSP response too large")
                message = json.loads(self.process.stdout.read(size))
                self.messages.put(message, timeout=1)
        except Exception as exc:
            try:
                self.messages.put(exc, timeout=1)
            except queue.Full:
                pass

    def send(self, method: str, params: Any, request: bool = False) -> int:
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if request:
            self.sequence += 1
            message["id"] = self.sequence
        data = json.dumps(message, ensure_ascii=True).encode()
        if len(data) > 2_000_000:
            raise ValueError("LSP request exceeds 2 MB limit")
        finished = threading.Event()
        failures: list[Exception] = []

        def write():
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                failures.append(exc)
            finally:
                finished.set()

        threading.Thread(target=write, daemon=True).start()
        if not finished.wait(max(0, self.deadline - time.monotonic())):
            stop_process(self.process)
            raise RuntimeError("clangd request write timed out")
        if failures:
            raise RuntimeError(f"clangd request failed: {failures[0]}")
        return self.sequence

    def receive(self) -> dict[str, Any]:
        try:
            message = self.messages.get(timeout=max(0, self.deadline - time.monotonic()))
        except queue.Empty as exc:
            raise RuntimeError("clangd query timed out") from exc
        if isinstance(message, Exception):
            raise RuntimeError(str(message)) from message
        if message.get("method") == "textDocument/publishDiagnostics":
            params = message["params"]
            self.diagnostics[params["uri"]] = params.get("diagnostics", [])
        return message

    def request(self, method: str, params: Any) -> Any:
        identity = self.send(method, params, True)
        while True:
            message = self.receive()
            if message.get("id") == identity:
                if "error" in message:
                    raise RuntimeError(f"clangd: {message['error']}")
                return message.get("result")

    def close(self):
        stop_process(self.process)
        if self.process.stdin:
            self.process.stdin.close()
        self.reader.join(timeout=2)
        if self.process.stdout:
            self.process.stdout.close()


@contextmanager
def clangd_session(workspace: Workspace) -> Iterator[tuple[LspClient, str]]:
    build = build_root(workspace)
    marker = workspace.artifacts / "cpp-binding.json"
    expected = {"root": str(workspace.root), "image": workspace.execution_image,
                "trusted_local": workspace.trusted_local}
    if not marker.is_file() or json.loads(marker.read_text()) != expected:
        raise RuntimeError("C++ build binding mismatch; configure this workspace/backend first")
    if not (build / "compile_commands.json").is_file():
        raise RuntimeError("Run cpp_build (configure_only=true) before semantic queries")
    args = ["--background-index=false", "--enable-config=0", "--log=error"]
    if workspace.execution_image:
        name = "robo-clangd-" + uuid.uuid4().hex
        with exported_source(workspace.root) as snapshot:
            command = container_command(workspace.execution_image,
                                        ["clangd", *args, "--compile-commands-dir=/build"],
                                        [(snapshot, "/workspace", False), (build, "/build", False)],
                                        name=name)
            client = LspClient(command)
            try:
                yield client, "file:///workspace"
            finally:
                client.close()
                remove_container(name)
    elif workspace.trusted_local:
        clangd = shutil.which("clangd")
        if not clangd:
            raise RuntimeError("clangd capability missing from PATH")
        client = LspClient([clangd, *args, f"--compile-commands-dir={build}"])
        try:
            yield client, workspace.root.as_uri()
        finally:
            client.close()
    else:
        raise RuntimeError("clangd execution needs --execution-image or --trusted-local")


def semantic_query(path: str, operation: str, line: int = 1, column: int = 1) -> dict[str, Any]:
    workspace = require_workspace()
    target = workspace.resolve(path)
    methods = {"definition": "textDocument/definition", "references": "textDocument/references",
               "hover": "textDocument/hover", "diagnostics": "diagnostics"}
    if operation not in methods:
        raise ValueError("operation must be definition, references, hover or diagnostics")
    def read_source(source: Path) -> str:
        if source.stat().st_size > 1_000_000:
            raise ValueError("clangd input file exceeds 1 MB limit")
        return source.read_text(encoding="utf-8")

    text = read_source(target)
    lines = text.splitlines()
    if not 1 <= line <= len(lines) or not 1 <= column <= len(lines[line - 1]) + 1:
        raise ValueError("line/column are 1-based Unicode character positions within the file")
    rows = compilation_database(build_root(workspace) / "compile_commands.json")
    relative = target.relative_to(workspace.root).as_posix()
    with clangd_session(workspace) as (client, root_uri):
        client.request("initialize", {"processId": None, "rootUri": root_uri,
                                      "capabilities": {"general": {"positionEncodings": ["utf-16"]}}})
        client.send("initialized", {})
        opened: set[str] = set()

        def open_file(rel: str, content: str):
            uri = root_uri + "/" + quote(rel, safe="/")
            if uri not in opened:
                client.send("textDocument/didOpen", {"textDocument": {
                    "uri": uri, "languageId": "cpp", "version": 1, "text": content}})
                client.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
                opened.add(uri)
            return uri

        # Parse a bounded set of translation units, making reference coverage explicit.
        for row in rows[:32]:
            if workspace.execution_image:
                file = PurePosixPath(row["directory"]) / row["file"]
                try:
                    rel = file.relative_to("/workspace").as_posix()
                except ValueError:
                    continue
            else:
                file_path = (Path(row["directory"]) / row["file"]).resolve()
                if not file_path.is_relative_to(workspace.root):
                    continue
                rel = file_path.relative_to(workspace.root).as_posix()
            source = workspace.resolve(rel)
            if source.is_file():
                open_file(rel, read_source(source))
        uri = open_file(relative, text)
        if operation == "diagnostics":
            while uri not in client.diagnostics:
                client.receive()
            result = client.diagnostics[uri]
        else:
            position = {"line": line - 1,
                        "character": len(lines[line - 1][:column - 1].encode("utf-16-le")) // 2}
            params: dict[str, Any] = {"textDocument": {"uri": uri}, "position": position}
            if operation == "references":
                params["context"] = {"includeDeclaration": True}
            result = client.request(methods[operation], params)
        return {"operation": operation, "result": result, "opened_files": len(opened),
                "translation_units": len(rows), "index_scope": "opened files; up to 32 translation units",
                "position_encoding": "utf-16", "result_positions": "LSP zero-based"}
