"""Cooperating Chroma writers share a lock and a cheap invalidation generation."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, cast
from weakref import WeakValueDictionary, finalize

_WRITER = uuid.uuid4().hex  # Process identity; PID reuse must not hide another writer.
_STORES: WeakValueDictionary[str, PersistentStore] = WeakValueDictionary()
_STORES_LOCK = RLock()


class PersistentStore:
    """Own one public Chroma client per directory so foreign writes can reopen it.

    Chroma's in-process ANN reader can lag a different process's writes even when
    SQLite metadata is current. Reopen the owned client on a foreign generation;
    never clear Chroma's global cache or reset/delete a collection.
    """

    def __init__(self, directory: Path) -> None:
        import chromadb
        from filelock import FileLock
        self.directory = directory
        self.guard = FileLock(str(directory / "rag-store.lock"), timeout=60)
        # The factory's ClientAPI annotation omits the concrete public close().
        self.client = cast(Any, chromadb.PersistentClient(path=str(directory)))
        self._release = finalize(self, self.client.close)
        self.collections: dict[str, Any] = {}
        self.foreign_tokens: dict[str, str] = {}

    def collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = self.client.get_collection(name)
        return self.collections[name]

    def refresh_reader(self, name: str, token: str) -> None:
        if token == "legacy" or token.startswith(_WRITER + ":"):
            return
        if self.foreign_tokens.get(name) == token:
            return
        import chromadb
        self._release()
        self.client = cast(Any, chromadb.PersistentClient(path=str(self.directory)))
        self._release = finalize(self, self.client.close)
        self.collections.clear()
        self.foreign_tokens[name] = token


def open_collection(directory: Path, name: str, metadata: dict) -> TrackedCollection:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(str(directory))
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = PersistentStore(directory)
            _STORES[key] = store
    with store.guard:
        collection = store.client.get_or_create_collection(name=name, metadata=metadata)
        store.collections[name] = collection
        return TrackedCollection(collection, directory, store=store)


class TrackedCollection:
    def __init__(self, collection: Any, directory: Path, *, store: PersistentStore | None = None) -> None:
        from filelock import FileLock
        self._raw = collection if store is None else None
        self._store = store
        self._name = collection.name if store else ""
        name = f"rag-{collection.id}"
        self.generation = directory / f"{name}.generation"
        self.guard = store.guard if store else FileLock(str(directory / f"{name}.lock"), timeout=60)

    @property
    def raw(self):
        return self._store.collection(self._name) if self._store else self._raw

    def refresh_reader(self, token: str) -> None:
        if self._store:
            self._store.refresh_reader(self._name, token)

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def revision_token(self) -> str:
        try:
            return self.generation.read_text(encoding="ascii")
        except FileNotFoundError:
            return "legacy"

    def _invalidate(self) -> None:
        fd, temporary = tempfile.mkstemp(dir=self.generation.parent, prefix="rag-generation-")
        try:
            with os.fdopen(fd, "w", encoding="ascii") as stream:
                stream.write(_WRITER + ":" + uuid.uuid4().hex)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.generation)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def upsert(self, **kwargs):
        with self.guard:
            self.refresh_reader(self.revision_token())
            self._invalidate()  # Also invalidates readers if the process dies during the write.
            try:
                return self.raw.upsert(**kwargs)
            finally:
                self._invalidate()

    def delete(self, **kwargs):
        with self.guard:
            self.refresh_reader(self.revision_token())
            self._invalidate()
            try:
                return self.raw.delete(**kwargs)
            finally:
                self._invalidate()
