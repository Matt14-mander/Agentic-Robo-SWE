"""Bounded external-repository text retrieval, independent of CMake/clangd execution."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Iterable

from agent.rag.chunking import _make_chunks
from agent.rag.models import CodeChunk
from agent.workspace import Workspace, source_files

CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".tpp", ".inl"}
TEXT_SUFFIXES = CPP_SUFFIXES | {".md", ".markdown", ".cmake"}
MAX_FILE_BYTES = 1_000_000


def workspace_identity(workspace: Workspace) -> str:
    return hashlib.sha256(os.path.normcase(str(workspace.root)).encode("utf-8")).hexdigest()[:20]


def workspace_index_path(workspace: Workspace) -> Path:
    path = (workspace.artifacts / "rag" / workspace_identity(workspace) / "chroma").resolve()
    if not path.is_relative_to(workspace.artifacts) or path.is_relative_to(workspace.root):
        raise ValueError("RAG artifacts must stay outside the workspace and inside its artifacts")
    return path


def supported_source(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name == "CMakeLists.txt"


def source_language(path: str) -> str:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in CPP_SUFFIXES:
        return "c" if suffix == ".c" else "cpp"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".cmake" or target.name == "CMakeLists.txt":
        return "cmake"
    return "python" if suffix == ".py" else "text"


def discover_workspace_files(workspace: Workspace, roots: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for name in roots:
        root = workspace.resolve(name)
        if not root.exists():
            raise ValueError(f"Index source root does not exist: {name}")
        if root.is_file():
            if not supported_source(root):
                raise ValueError(f"Unsupported index source: {name}")
            candidates = [root]
        elif root.is_dir():
            candidates = source_files(root)
        else:
            raise ValueError(f"Index source must be a regular file or directory: {name}")
        for candidate in candidates:
            if supported_source(candidate):
                files.add(workspace.resolve(candidate.relative_to(workspace.root).as_posix()))
    if len(files) > 20000 or sum(path.stat().st_size for path in files) > 100_000_000:
        raise ValueError("Index exceeds 20,000 files / 100 MB limit")
    return sorted(files)


def read_workspace_source(workspace: Workspace, relative: str) -> tuple[Path, str]:
    path = workspace.resolve(relative)
    if not supported_source(path):
        raise ValueError(f"Unsupported index source: {relative}")
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Index source must be a regular file: {relative}")
        if info.st_size > MAX_FILE_BYTES:
            raise ValueError(f"Index source exceeds 1 MB limit: {relative}")
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
    except FileNotFoundError:
        data = b""
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"Index source exceeds 1 MB limit: {relative}")
    source = data.decode("utf-8")
    if "\0" in source:
        raise ValueError(f"Binary index source is not supported: {relative}")
    return path, source


def chunk_workspace_source(workspace: Workspace, path: Path, source: str) -> list[CodeChunk]:
    relative = path.relative_to(workspace.root).as_posix()
    language = source_language(relative)
    kind = "cpp" if language in {"c", "cpp"} else "document" if language == "markdown" else "build"
    return _make_chunks(path=relative, symbol="<file>", kind=kind,
                        lines=source.splitlines(), start_line=1, max_lines=120, overlap=15)


def chunk_workspace(workspace: Workspace, roots: Iterable[str]) -> tuple[list[Path], list[CodeChunk]]:
    files = discover_workspace_files(workspace, roots)
    chunks = []
    for path in files:
        resolved, source = read_workspace_source(workspace, path.relative_to(workspace.root).as_posix())
        chunks.extend(chunk_workspace_source(workspace, resolved, source))
    return files, chunks
