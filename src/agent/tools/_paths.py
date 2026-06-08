"""Internal path helpers shared by tools. Not exposed as @tool."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_within_root(path: str) -> Path:
    """Resolve a path and ensure it stays inside ``PROJECT_ROOT``.

    Raises ``ValueError`` if the resolved path escapes the project root.
    """
    p = Path(path)
    p = (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        p.relative_to(PROJECT_ROOT)
    except ValueError as e:
        raise ValueError(
            f"Refusing to access path outside project root: {p} not under {PROJECT_ROOT}"
        ) from e
    return p


def relpath_for_display(p: Path) -> str:
    """Return path relative to project root with forward slashes (for LLM-friendly display)."""
    return p.relative_to(PROJECT_ROOT).as_posix()
