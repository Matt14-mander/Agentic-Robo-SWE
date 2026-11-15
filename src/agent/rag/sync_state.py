"""Pending sync intent, persisted independently of the indexed chunks."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class SyncJournal:
    """Atomic intent file for retrying an interrupted, single-writer sync.

    In-memory indexes can omit the path. Persistent indexes must supply a unique
    path per collection so a newly opened index can discover incomplete writes.
    """

    def __init__(self, source_root: Path, path: Path | None = None) -> None:
        self.source_root = str(source_root.resolve())
        self.path = path
        self._pending: tuple[tuple[str, ...], bool] | None = None

    def load(self) -> tuple[tuple[str, ...], bool] | None:
        if self.path is None:
            return self._pending
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            value = json.loads(text)
            if (
                not isinstance(value, dict)
                or value.get("version") != 1
                or value.get("source_root") != self.source_root
                or not isinstance(value.get("roots"), list)
                or not all(isinstance(root, str) for root in value["roots"])
                or not isinstance(value.get("rebuild"), bool)
            ):
                raise ValueError("invalid schema or source binding")
            return tuple(value["roots"]), value["rebuild"]
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid pending RAG sync journal; explicitly sync to recover") from exc

    def begin(self, roots: tuple[str, ...], rebuild: bool) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({
                "version": 1, "source_root": self.source_root,
                "roots": roots, "rebuild": rebuild,
            })
            # The intent must reach disk before any destructive collection writes.
            fd, temporary = tempfile.mkstemp(prefix=self.path.name, dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        self._pending = roots, rebuild

    def clear(self) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        self._pending = None
