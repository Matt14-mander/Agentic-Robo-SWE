"""Explicit target-repository context, independent of the harness installation."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

HARNESS_ROOT = Path(__file__).resolve().parents[2]
_PRIVATE = {".git", ".agent_state", ".aws", ".ssh", ".docker", ".codex", ".agents", ".venv", "__pycache__"}


@dataclass(frozen=True)
class Workspace:
    root: Path
    artifacts: Path
    trusted_local: bool = False
    execution_image: str | None = None

    @classmethod
    def create(
        cls, root: str | Path, artifacts: str | Path | None = None, *,
        trusted_local: bool = False, execution_image: str | None = None,
    ) -> Workspace:
        target = Path(root).resolve(strict=True)
        if not target.is_dir():
            raise ValueError("Workspace root must be a directory")
        identity = hashlib.sha256(str(target).encode()).hexdigest()[:20]
        output = Path(artifacts or HARNESS_ROOT / ".agent_state" / "workspaces" / identity).resolve()
        if output.is_relative_to(target) or target.is_relative_to(output):
            raise ValueError("Workspace and artifact roots must not overlap")
        if trusted_local and execution_image:
            raise ValueError("Choose trusted local execution OR a container image")
        if execution_image:
            from agent.containers import validate_image
            validate_image(execution_image)
        return cls(target, output, trusted_local, execution_image)

    def binding(self) -> dict[str, object]:
        return {"root": str(self.root), "artifacts": str(self.artifacts),
                "trusted_local": self.trusted_local, "execution_image": self.execution_image}

    def resolve(self, path: str) -> Path:
        raw = Path(path)
        resolved = (self.root / raw).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Path escapes target workspace")
        relative = resolved.relative_to(self.root)
        if any(part in _PRIVATE or part.startswith(".env") for part in relative.parts):
            raise ValueError("Private workspace metadata is not exposed to tools")
        # Do not allow even in-tree aliases: snapshots and patch identity must agree.
        current = self.root
        for part in (raw.relative_to(self.root) if raw.is_absolute() else raw).parts:
            current = current / part
            if current.is_symlink() or _is_junction(current):
                raise ValueError("Workspace symlinks/junctions are not exposed to tools")
        return resolved


def _is_junction(path: Path) -> bool:
    # Path.is_junction is unavailable on Python 3.11; use the Windows reparse
    # attribute as well so the supported minimum version keeps the same boundary.
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


_ACTIVE: ContextVar[Workspace | None] = ContextVar("agent_workspace", default=None)


def current_workspace() -> Workspace | None:
    return _ACTIVE.get()


def check_binding(state) -> None:
    workspace = current_workspace()
    expected = workspace.binding() if workspace else None
    if state.get("workspace_binding") != expected:
        raise ValueError("Session workspace/backend binding mismatch; reopen with original options")
    if workspace and state.get("semantic_cache_enabled"):
        raise ValueError("External repository semantic response caching is not enabled")


@contextmanager
def use_workspace(workspace: Workspace) -> Iterator[Workspace]:
    token = _ACTIVE.set(workspace)
    try:
        yield workspace
    finally:
        _ACTIVE.reset(token)


def source_files(root: Path) -> list[Path]:
    """Enumerate a bounded, regular-file-only candidate tree; never follow links."""
    import os
    files: list[Path] = []
    total = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            path = base / name
            if name in _PRIVATE or name.startswith(".env") or name in {"build", "dist", ".pytest_cache", ".ruff_cache"}:
                dirs.remove(name)
            elif path.is_symlink() or _is_junction(path):
                raise ValueError(f"Candidate directory link is not allowed: {name}")
        for name in names:
            if name.startswith(".env") or name in _PRIVATE:
                continue
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise ValueError(f"Candidate must contain regular files only: {name}")
            total += info.st_size
            files.append(path)
            if len(files) > 20000 or total > 100_000_000:
                raise ValueError("Candidate exceeds 20,000 files / 100 MB snapshot limit")
    return sorted(files)


def source_digest(root: Path) -> str:
    entries = [(p.relative_to(root).as_posix(), bool(p.stat().st_mode & 0o111),
                hashlib.sha256(p.read_bytes()).hexdigest())
               for p in source_files(root)]
    return hashlib.sha256(json.dumps(entries, ensure_ascii=True).encode()).hexdigest()


@contextmanager
def exported_source(root: Path) -> Iterator[Path]:
    """Only sanitized regular files cross the container boundary."""
    import shutil
    with tempfile.TemporaryDirectory(prefix="robo_candidate_") as temporary:
        target = Path(temporary)
        target.chmod(0o755)
        for path in source_files(root):
            destination = target / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            for parent in destination.parents:
                if parent == target:
                    break
                parent.chmod(0o755)
            shutil.copyfile(path, destination)
            destination.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        yield target
