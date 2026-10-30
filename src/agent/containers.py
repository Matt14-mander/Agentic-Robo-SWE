"""Fail-closed container execution. No host environment, network or implicit pull."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_IMAGE = re.compile(r"^(?:[a-zA-Z0-9][a-zA-Z0-9._/:~-]*@)?sha256:[a-f0-9]{64}$")


def validate_image(image: str) -> None:
    if not isinstance(image, str) or not _IMAGE.fullmatch(image):
        raise ValueError("Container image must be pinned by sha256 digest or local image ID")


def clean_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR")
            if key in os.environ}


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    native_returncode: int | None = None
    termination: str | None = None


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, check=False)
        finally:
            # AppContainer may deny taskkill even for a child. The process handle
            # still permits terminating our own direct child.
            if process.poll() is None:
                process.kill()
    else:
        try:
            getattr(os, "killpg")(process.pid, getattr(signal, "SIGKILL"))
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def run_bounded(
    command: list[str], *, cwd: Path | None = None, stdin: str = "",
    timeout: float = 120, output_limit: int = 2_000_000,
) -> ProcessResult:
    """Bound both wall time and captured bytes, including hostile child output."""
    process = subprocess.Popen(command, cwd=cwd, env=clean_environment(),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=os.name != "nt")
    exceeded = threading.Event()
    buffers = [bytearray(), bytearray()]

    def read(stream, index):
        try:
            while chunk := stream.read(4096):
                room = output_limit - len(buffers[index])
                buffers[index].extend(chunk[:max(0, room)])
                if len(chunk) > room:
                    exceeded.set()
                    break
        finally:
            stream.close()

    def write():
        try:
            assert process.stdin is not None
            process.stdin.write(stdin.encode("utf-8"))
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    readers = [threading.Thread(target=read, args=(stream, i), daemon=True)
               for i, stream in enumerate((process.stdout, process.stderr))]
    for thread in readers:
        thread.start()
    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    deadline = time.monotonic() + timeout
    error = ""
    try:
        while process.poll() is None:
            if exceeded.is_set() or time.monotonic() >= deadline:
                error = "output limit exceeded" if exceeded.is_set() else "execution timed out"
                stop_process(process)
                break
            time.sleep(0.02)
    finally:
        stop_process(process)
        for thread in readers:
            thread.join(timeout=2)
        writer.join(timeout=2)
    if exceeded.is_set():
        error = "output limit exceeded"
    return ProcessResult(124 if error else int(process.returncode),
                         buffers[0].decode("utf-8", errors="replace"),
                         buffers[1].decode("utf-8", errors="replace") + error,
                         native_returncode=int(process.returncode),
                         termination=("output_limit" if exceeded.is_set() else "timeout") if error else None)


def container_command(
    image: str, command: list[str], mounts: list[tuple[Path, str, bool]], *,
    name: str, workdir: str = "/tmp",
) -> list[str]:
    validate_image(image)
    if not command:
        raise ValueError("Container command cannot be empty")
    args = ["docker", "run", "--rm", "--pull=never", "-i", "--name", name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--pids-limit=128", "--memory=1g",
            "--cpus=1", "--user=65534:65534", "--tmpfs=/tmp:rw,exec,size=512m,mode=1777",
            "--workdir", workdir]
    for source, target, writable in mounts:
        source = source.resolve(strict=True)
        if "," in str(source) or not target.startswith("/") or "," in target:
            raise ValueError("Invalid container mount")
        args += ["--mount", f"type=bind,src={source},dst={target}" + ("" if writable else ",readonly")]
    return [*args, "--entrypoint", command[0], image, *command[1:]]


def remove_container(name: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       env=clean_environment(), timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_container(
    image: str, command: list[str], mounts: list[tuple[Path, str, bool]], *,
    stdin: str = "", timeout: float = 120, workdir: str = "/tmp",
) -> ProcessResult:
    name = "robo-swe-" + uuid.uuid4().hex
    args = container_command(image, command, mounts, name=name, workdir=workdir)
    try:
        return run_bounded(args, stdin=stdin, timeout=timeout)
    finally:
        remove_container(name)
