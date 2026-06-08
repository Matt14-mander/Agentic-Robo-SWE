"""读类工具 —— 给 LLM 探索仓库的能力。

设计参考 Princeton SWE-agent 的 ACI 原则:
- 返回带行号的内容, 让 LLM 能精准引用位置 (后续 write_patch 需要)。
- 对大输出做硬截断, 防止单个工具调用撑爆上下文。
- 出错时返回可读字符串, 不抛异常 (ToolNode 接错误更友好)。
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool

from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root

_MAX_GREP_RESULTS = 50
_MAX_OUTPUT_CHARS = 20_000  # 单工具返回字符上限, 防上下文爆炸


def _truncate(text: str, label: str = "output") -> str: # 超出上限时截断并标注, 让 LLM 知道发生了什么, 而不是莫名其妙地丢失内容。
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    cut = text[:_MAX_OUTPUT_CHARS]
    omitted = len(text) - _MAX_OUTPUT_CHARS
    return f"{cut}\n... [{label} truncated, {omitted} more chars hidden]"


@tool
def read_file_chunk(path: str, offset: int = 0, limit: int = 200) -> str:
    """读取仓库内文本文件的指定行区间, 返回带行号前缀的内容。

    Args:
        path: 相对仓库根目录的路径 (e.g. "src/agent/state.py")。
        offset: 起始行号 (0-based, 默认 0)。
        limit: 最多读取行数 (默认 200; 文件很长时多次调用拼接)。

    Returns:
        每行格式 ``"   42| <content>"`` 的字符串; 失败时返回以 "ERROR:" 开头的描述。
    """
    try:
        p = resolve_within_root(path)
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if not p.is_file():
        return f"ERROR: Not a regular file: {path}"

    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: Cannot decode {path} as UTF-8 (binary file?)"

    lines = text.splitlines()
    total = len(lines)
    if offset >= total and total > 0:
        return f"ERROR: offset {offset} out of range; file has {total} lines."

    end = min(total, offset + limit)
    selected = lines[offset:end]    # 注意 offset 是 0-based, 行号前缀是 1-based, 显示时要加 1
    width = max(4, len(str(end)))   # 行号前缀宽度, 至少 4 位 (支持到 9999 行), 根据实际行数调整宽度以对齐; 过大文件也不至于前缀占太多空间
    body = "\n".join(f"{i + 1:>{width}}| {ln}" for i, ln in enumerate(selected, start=offset))  # 行号前缀右对齐, 后跟 "| " 分隔符, 让 LLM 能清晰区分行号和内容

    header = f"# {relpath_for_display(p)}  (lines {offset + 1}-{end} of {total})\n"
    return _truncate(header + body, label="file")


@tool
def grep_codebase(pattern: str, glob: str = "**/*.py", max_results: int = 30) -> str:
    """正则搜索仓库内文件, 返回 ``path:lineno: <line>`` 形式的匹配清单。

    Args:
        pattern: Python 正则 (会被 ``re.compile`` 编译)。
        glob: 限制搜索的文件通配符, 默认 ``**/*.py``。例: ``tests/**/*.py``。
        max_results: 最大返回匹配数, 默认 30, 上限 50。

    Returns:
        每行一个匹配的字符串; 无匹配返回 "<no matches>"。
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"ERROR: Invalid regex {pattern!r}: {e}"

    max_results = min(max(1, max_results), _MAX_GREP_RESULTS)

    # 跳过常见噪声目录, 减少 LLM 处理量
    skip_dirs = {".venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
                 ".ruff_cache", "node_modules", "dist", "build"}

    matches: list[str] = []
    for file in PROJECT_ROOT.glob(glob): # 注意 glob 可能返回目录, 需要 is_file() 过滤; 也可能返回非 UTF-8 文件, 需要 try-except 过滤
        if not file.is_file():
            continue
        if any(part in skip_dirs for part in file.parts):
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{relpath_for_display(file)}:{lineno}: {line.rstrip()}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"<no matches for {pattern!r} under glob {glob!r}>"

    header = f"# grep {pattern!r}  (glob: {glob}, {len(matches)} match{'es' if len(matches) > 1 else ''})\n"
    return _truncate(header + "\n".join(matches), label="grep")


@tool
def list_dir(path: str = ".", max_depth: int = 1) -> str:
    """列出目录内容 (tree 风格)。

    Args:
        path: 相对仓库根目录的路径, 默认 "." 即根目录。
        max_depth: 递归深度, 默认 1 (仅展示直接子项); 上限 3。

    Returns:
        每行 ``<indent><name>[/]`` 的树形清单。
    """
    try:
        root = resolve_within_root(path)
    except ValueError as e:
        return f"ERROR: {e}"
    if not root.exists():
        return f"ERROR: Path not found: {path}"
    if not root.is_dir():
        return f"ERROR: Not a directory: {path}"

    max_depth = min(max(1, max_depth), 3)   # 防止过深导致输出过大; 3 层已经能看到比较清晰的结构了
    skip = {".venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
            ".ruff_cache", "node_modules", ".langgraph_api"}    # 常见噪声目录, 列出来反而干扰 LLM

    lines: list[str] = [f"# {relpath_for_display(root) or '.'}/"]

    def walk(d: Path, depth: int, indent: str) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            if entry.name in skip or entry.name.startswith("."):
                continue
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{indent}{entry.name}{suffix}")
            if entry.is_dir():
                walk(entry, depth + 1, indent + "  ")

    walk(root, 1, "  ")
    return _truncate("\n".join(lines), label="listing")
