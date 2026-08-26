"""Revision-aware, process-local LRU cache for deterministic retrieval results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from threading import RLock

from agent.rag.models import SearchResult


@dataclass(frozen=True)
class RetrievalCacheKey:
    revision: str
    query: str
    top_k: int
    path_prefix: str | None
    strategy: str


@dataclass(frozen=True)
class RetrievalCacheStats:
    hits: int
    misses: int
    bypasses: int
    invalidations: int
    evictions: int
    entries: int
    max_entries: int
    hit_rate: float
    estimated_saved_ms: float
    revision: str | None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class _CacheEntry:
    results: tuple[SearchResult, ...]
    compute_ms: float


class RetrievalCache:
    """Thread-safe LRU cache cleared whenever the source-index revision changes."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[RetrievalCacheKey, _CacheEntry] = OrderedDict()
        self._revision: str | None = None
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._invalidations = 0
        self._evictions = 0
        self._estimated_saved_ms = 0.0
        self._lock = RLock()

    @staticmethod
    def make_key(
        *,
        revision: str,
        query: str,
        top_k: int,
        path_prefix: str | None,
        strategy: str,
    ) -> RetrievalCacheKey:
        return RetrievalCacheKey(
            revision=revision,
            query=" ".join(query.lower().split()),
            top_k=top_k,
            path_prefix=path_prefix.replace("\\", "/").strip() if path_prefix else None,
            strategy=strategy,
        )

    def prepare_revision(self, revision: str) -> None:
        with self._lock:
            if self._revision is not None and revision != self._revision:
                self._entries.clear()
                self._invalidations += 1
            self._revision = revision

    def get(self, key: RetrievalCacheKey) -> tuple[bool, list[SearchResult]]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return False, []
            self._entries.move_to_end(key)
            self._hits += 1
            self._estimated_saved_ms += entry.compute_ms
            return True, list(entry.results)

    def put(
        self,
        key: RetrievalCacheKey,
        results: list[SearchResult],
        *,
        compute_ms: float,
    ) -> None:
        with self._lock:
            self._entries[key] = _CacheEntry(tuple(results), max(0.0, compute_ms))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    def record_bypass(self) -> None:
        with self._lock:
            self._bypasses += 1

    def stats(self) -> RetrievalCacheStats:
        with self._lock:
            attempts = self._hits + self._misses
            return RetrievalCacheStats(
                hits=self._hits,
                misses=self._misses,
                bypasses=self._bypasses,
                invalidations=self._invalidations,
                evictions=self._evictions,
                entries=len(self._entries),
                max_entries=self.max_entries,
                hit_rate=round(self._hits / attempts, 4) if attempts else 0.0,
                estimated_saved_ms=round(self._estimated_saved_ms, 3),
                revision=self._revision,
            )
