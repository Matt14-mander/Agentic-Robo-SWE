"""Workspace-scoped semantic cache for final read-only LLM responses."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from agent.rag.embedding import HashingEmbedder
from agent.tools._paths import PROJECT_ROOT, resolve_within_root

DEFAULT_SEMANTIC_CACHE_DB = PROJECT_ROOT / ".agent_state" / "semantic_cache.sqlite"
_FINGERPRINT_SUFFIXES = {".py", ".toml", ".json", ".yaml", ".yml", ".md"}
_SKIP_DIRS = {
    ".agent_state", ".claude", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "benchmarks/results", "benchmarks/rag_results", "benchmarks/workspaces",
    "build", "chroma_db", "dist", "node_modules",
}


@dataclass(frozen=True)
class SemanticCacheScope:
    workspace: str
    model: str
    prompt: str
    tools: str


@dataclass(frozen=True)
class SemanticCacheMatch:
    response: str
    similarity: float
    exact: bool


@dataclass(frozen=True)
class SemanticCacheStats:
    hits: int
    misses: int
    bypasses: int
    stores: int
    entries: int
    hit_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def workspace_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """Hash relevant repository content while excluding generated mutable state."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in _FINGERPRINT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        if any(relative == item or relative.startswith(f"{item}/") for item in _SKIP_DIRS):
            continue
        files.append(path)
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(128 * 1024), b""):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest()[:20]


def semantic_cache_scope(
    *,
    model_identity: str,
    system_prompt: str,
    tools: Iterable[Any],
) -> SemanticCacheScope:
    schemas = []
    for tool in tools:
        schema = getattr(tool, "args_schema", None)
        schema_payload = schema.model_json_schema() if schema is not None else {}
        schemas.append({"name": getattr(tool, "name", type(tool).__name__), "schema": schema_payload})
    return SemanticCacheScope(
        workspace=workspace_fingerprint(),
        model=_digest_text(model_identity),
        prompt=_digest_text(system_prompt),
        tools=_digest_text(json.dumps(schemas, sort_keys=True, ensure_ascii=False)),
    )


class SemanticResponseCache:
    """Persistent semantic lookup with strict scope partitioning and bounded TTL."""

    def __init__(
        self,
        database: str | Path,
        *,
        threshold: float = 0.97,
        ttl_seconds: float = 7 * 24 * 60 * 60,
        max_entries: int = 512,
        embedder: HashingEmbedder | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if ttl_seconds <= 0 or max_entries < 1:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.embedder = embedder or HashingEmbedder()
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.stores = 0
        self._setup()

    def _setup(self) -> None:
        with self.lock:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS semantic_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    tools TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    query TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_access REAL NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS semantic_scope_idx
                ON semantic_responses(workspace, model, prompt, tools, expires_at);
                """
            )
            self.connection.commit()

    @staticmethod
    def _normalized_query(query: str) -> str:
        return " ".join(query.lower().split())

    def lookup(
        self,
        query: str,
        scope: SemanticCacheScope,
        *,
        threshold: float | None = None,
    ) -> SemanticCacheMatch | None:
        normalized = self._normalized_query(query)
        cutoff = self.threshold if threshold is None else threshold
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        now = time.time()
        vector = self.embedder.embed([query])[0]
        with self.lock:
            self.connection.execute("DELETE FROM semantic_responses WHERE expires_at <= ?", (now,))
            rows = self.connection.execute(
                """
                SELECT id, normalized_query, embedding, response
                FROM semantic_responses
                WHERE workspace = ? AND model = ? AND prompt = ? AND tools = ?
                  AND expires_at > ?
                """,
                (scope.workspace, scope.model, scope.prompt, scope.tools, now),
            ).fetchall()
            best: tuple[int, str, float, bool] | None = None
            for row_id, stored_query, encoded_vector, response in rows:
                exact = stored_query == normalized
                similarity = 1.0 if exact else _dot(vector, json.loads(encoded_vector))
                if similarity >= cutoff and (best is None or similarity > best[2]):
                    best = (int(row_id), str(response), similarity, exact)
            if best is None:
                self.misses += 1
                self.connection.commit()
                return None
            self.connection.execute(
                "UPDATE semantic_responses SET hits = hits + 1, last_access = ? WHERE id = ?",
                (now, best[0]),
            )
            self.connection.commit()
            self.hits += 1
            return SemanticCacheMatch(best[1], round(best[2], 6), best[3])

    def store(self, query: str, response: str, scope: SemanticCacheScope) -> None:
        if not query.strip() or not response.strip():
            return
        now = time.time()
        normalized = self._normalized_query(query)
        embedding = json.dumps(self.embedder.embed([query])[0], separators=(",", ":"))
        with self.lock:
            self.connection.execute(
                """
                DELETE FROM semantic_responses
                WHERE workspace = ? AND model = ? AND prompt = ? AND tools = ?
                  AND normalized_query = ?
                """,
                (scope.workspace, scope.model, scope.prompt, scope.tools, normalized),
            )
            self.connection.execute(
                """
                INSERT INTO semantic_responses (
                    workspace, model, prompt, tools, normalized_query, query, embedding,
                    response, created_at, expires_at, last_access
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope.workspace, scope.model, scope.prompt, scope.tools, normalized, query,
                    embedding, response, now, now + self.ttl_seconds, now,
                ),
            )
            self.connection.execute(
                """
                DELETE FROM semantic_responses WHERE id IN (
                    SELECT id FROM semantic_responses ORDER BY last_access DESC LIMIT -1 OFFSET ?
                )
                """,
                (self.max_entries,),
            )
            self.connection.commit()
            self.stores += 1

    def record_bypass(self) -> None:
        self.bypasses += 1

    def stats(self) -> SemanticCacheStats:
        with self.lock:
            entries = int(self.connection.execute(
                "SELECT COUNT(*) FROM semantic_responses WHERE expires_at > ?", (time.time(),)
            ).fetchone()[0])
        attempts = self.hits + self.misses
        return SemanticCacheStats(
            hits=self.hits,
            misses=self.misses,
            bypasses=self.bypasses,
            stores=self.stores,
            entries=entries,
            hit_rate=round(self.hits / attempts, 4) if attempts else 0.0,
        )

    def close(self) -> None:
        with self.lock:
            self.connection.close()


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@lru_cache(maxsize=1)
def get_semantic_cache() -> SemanticResponseCache:
    configured = os.getenv("SEMANTIC_CACHE_DB", str(DEFAULT_SEMANTIC_CACHE_DB))
    path = resolve_within_root(configured)
    return SemanticResponseCache(path)
