"""Structure-aware Python source chunking and repository discovery."""

from __future__ import annotations

import ast
import hashlib
import io
import tokenize
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


def _decorator_lines(text: str) -> list[int]:
    """Locate decorator markers, including @(...), without confusing matrix products."""
    starts: list[int] = []
    statement_start = True
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            continue
        if token.type == tokenize.NEWLINE:
            statement_start = True
            continue
        if statement_start and token.type == tokenize.OP and token.string == "@":
            starts.append(token.start[0])
        statement_start = False
    return starts


def _definition_start(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    decorators: list[int],
) -> int:
    if not node.decorator_list:
        return node.lineno
    # AST expression positions can start below the @ line for parenthesized decorators.
    first_expression = node.decorator_list[0].lineno
    return max(line for line in decorators if line <= first_expression)


def chunk_python_file(
    path: Path,
    max_lines: int = 120,
    overlap: int = 15,
    *,
    source_text: str | None = None,
) -> list[CodeChunk]:
    """Chunk a Python file, optionally using an already-read source snapshot."""
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Source file is outside project root: {path}") from exc
    if max_lines < 10 or overlap < 0 or overlap >= max_lines:
        raise ValueError("Require max_lines >= 10 and 0 <= overlap < max_lines")

    text = resolved.read_text(encoding="utf-8") if source_text is None else source_text
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
    decorators = _decorator_lines(text)

    def emit(start: int, end: int, symbol: str, kind: str) -> None:
        region = lines[start - 1:end]
        if not any(line.strip() for line in region):
            return
        chunks.extend(_make_chunks(
            path=relative,
            symbol=symbol,
            kind=kind,
            lines=region,
            start_line=start,
            max_lines=max_lines,
            overlap=overlap,
        ))

    cursor = 1
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = _definition_start(node, decorators)
        end = node.end_lineno or node.lineno
        # Preserve assignments, imports, control flow and comments between definitions.
        emit(cursor, start - 1, "<module>", "module")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(start, end, node.name, "function")
        else:
            class_cursor = start
            for method in node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                method_start = _definition_start(method, decorators)
                method_end = method.end_lineno or method.lineno
                emit(class_cursor, method_start - 1, node.name, "class")
                emit(method_start, method_end, f"{node.name}.{method.name}", "method")
                class_cursor = method_end + 1
            # Includes class attributes and nested/conditional definitions after methods.
            emit(class_cursor, end, node.name, "class")
        cursor = end + 1
    emit(cursor, len(lines), "<module>", "module")

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
