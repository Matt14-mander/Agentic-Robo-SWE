"""SQLite-backed graph lifecycle for local resumable Agent sessions."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agent.tools._paths import PROJECT_ROOT, resolve_within_root
from agent.workspace import current_workspace

DEFAULT_CHECKPOINT_DB = PROJECT_ROOT / ".agent_state" / "checkpoints.sqlite"


@contextmanager
def open_persistent_graph(
    database: str | Path | None = None,
) -> Iterator[Any]:
    """Compile the Agent with SqliteSaver and close the connection on exit."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite checkpoint support is not installed. Run: uv sync --extra checkpoint"
        ) from exc

    workspace = current_workspace()
    path = Path(database) if database is not None else (
        Path("checkpoints.sqlite") if workspace else DEFAULT_CHECKPOINT_DB)
    if workspace:
        path = (workspace.artifacts / path).resolve()
        if not path.is_relative_to(workspace.artifacts):
            raise ValueError("Checkpoint must be inside the workspace artifact root")
    elif not path.is_absolute():
        path = resolve_within_root(str(path))
    else:
        path = resolve_within_root(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        from agent.graph import build_graph

        yield build_graph(checkpointer=SqliteSaver(connection))
    finally:
        connection.close()
