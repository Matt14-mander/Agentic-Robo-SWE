"""Structure-aware Python source chunking and repository discovery."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Iterable

from agent.rag.models import CodeChunk
from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root

_SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "node_modules", "chroma_db", "results", "workspaces",
}


def discover_python_files(roots: Iterable[str] = ("src",)) -> list[Path]:
    files: set[Path] = set()
    for root_name in roots:
        root = resolve_within_root(root_name)
        if root.is_file() and root.suffix == ".py":
            files.add(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if path.is_file() and not any(part in _SKIP_DIRS for part in path.parts):
                files.add(path)
    return sorted(files, key=lambda path: relpath_for_display(path))


def _stable_id(path: str, symbol: str, start_line: int, part: int) -> str:
    raw = f"{path}\0{symbol}\0{start_line}\0{part}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _make_chunks(
    *,
    path: str,
    symbol: str,
    kind: str,
    lines: list[str],
    start_line: int,
    max_lines: int,
    overlap: int,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    step = max(1, max_lines - overlap)
    for part, offset in enumerate(range(0, len(lines), step)):
        selected = lines[offset:offset + max_lines]
        if not selected:
            continue
        content = "\n".join(selected).rstrip() + "\n"
        chunk_start = start_line + offset
        chunk_end = chunk_start + len(selected) - 1
        part_symbol = symbol if part == 0 and len(lines) <= max_lines else f"{symbol}#part{part + 1}"
        chunks.append(CodeChunk(
            id=_stable_id(path, symbol, start_line, part),
            path=path,
            symbol=part_symbol,
            kind=kind,
            start_line=chunk_start,
            end_line=chunk_end,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ))
        if offset + max_lines >= len(lines):
            break
    return chunks


def chunk_python_file(path: Path, max_lines: int = 120, overlap: int = 15) -> list[CodeChunk]:
    """Chunk a Python file by module preamble, functions, classes and methods."""
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Source file is outside project root: {path}") from exc
    if max_lines < 10 or overlap < 0 or overlap >= max_lines:
        raise ValueError("Require max_lines >= 10 and 0 <= overlap < max_lines")

    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    relative = relpath_for_display(resolved)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _make_chunks(
            path=relative,
            symbol="<module>",
            kind="module",
            lines=lines,
            start_line=1,
            max_lines=max_lines,
            overlap=overlap,
        )

    chunks: list[CodeChunk] = []
    first_definition = min(
        (node.lineno for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))),
        default=len(lines) + 1,
    )
    preamble = lines[:first_definition - 1]
    if any(line.strip() for line in preamble):
        chunks.extend(_make_chunks(
            path=relative,
            symbol="<module>",
            kind="module",
            lines=preamble,
            start_line=1,
            max_lines=max_lines,
            overlap=overlap,
        ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            chunks.extend(_make_chunks(
                path=relative,
                symbol=node.name,
                kind="function",
                lines=lines[node.lineno - 1:end],
                start_line=node.lineno,
                max_lines=max_lines,
                overlap=overlap,
            ))
        elif isinstance(node, ast.ClassDef):
            methods = [
                child for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            header_end = methods[0].lineno - 1 if methods else (node.end_lineno or node.lineno)
            chunks.extend(_make_chunks(
                path=relative,
                symbol=node.name,
                kind="class",
                lines=lines[node.lineno - 1:header_end],
                start_line=node.lineno,
                max_lines=max_lines,
                overlap=overlap,
            ))
            for method in methods:
                end = method.end_lineno or method.lineno
                chunks.extend(_make_chunks(
                    path=relative,
                    symbol=f"{node.name}.{method.name}",
                    kind="method",
                    lines=lines[method.lineno - 1:end],
                    start_line=method.lineno,
                    max_lines=max_lines,
                    overlap=overlap,
                ))

    if not chunks and lines:
        chunks.extend(_make_chunks(
            path=relative,
            symbol="<module>",
            kind="module",
            lines=lines,
            start_line=1,
            max_lines=max_lines,
            overlap=overlap,
        ))
    return chunks


def chunk_codebase(roots: Iterable[str] = ("src",)) -> tuple[list[Path], list[CodeChunk]]:
    files = discover_python_files(roots)
    chunks = [chunk for path in files for chunk in chunk_python_file(path)]
    return files, chunks
