"""文件读取工具 —— Phase 1 基础版,
   预留 ``offset/limit`` 参数, Phase 2 升级为真正的分页 ``read_file_chunk``。
"""

from __future__ import annotations

from pathlib import Path

# 路径越界保护: 仅允许在仓库根目录下读取
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    """读取仓库内文本文件。

    Args:
        path: 相对或绝对路径。绝对路径必须位于仓库根目录内。
        offset: 起始行号 (0-based, Phase 2 启用)。
        limit: 最大读取行数 (Phase 2 启用)。

    Returns:
        文件全文 (Phase 1) 或指定区间 (Phase 2)。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: 路径越出仓库根目录。
    """
    p = Path(path)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()

    try:
        p.relative_to(_PROJECT_ROOT)
    except ValueError as e:
        raise ValueError(
            f"Refusing to read outside project root: {p} not under {_PROJECT_ROOT}"
        ) from e

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8")

    if offset == 0 and limit is None:
        return text

    lines = text.splitlines(keepends=True)
    end = len(lines) if limit is None else offset + limit
    return "".join(lines[offset:end])
