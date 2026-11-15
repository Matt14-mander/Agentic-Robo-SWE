"""Data contracts shared by the Phase 3 indexer and retrieval tool."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class CodeChunk:
    id: str
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    parent_id: str = ""
    parent_hash: str = ""
    start_column: int = 1
    end_column: int = 1

    @property
    def revision_hash(self) -> str:
        if not self.parent_id:
            return self.content_hash
        return hashlib.sha256(
            f"{self.content_hash}:{self.parent_hash}:{self.start_column}:{self.end_column}".encode()
        ).hexdigest()

    def metadata(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_hash": self.revision_hash,
            "parent_id": self.parent_id,
            "start_column": self.start_column,
            "end_column": self.end_column,
        }


@dataclass(frozen=True)
class ParentContext:
    id: str
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True)
class SearchResult:
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str
    score: float
    parent_id: str = ""
    start_column: int = 1
    end_column: int = 1
    parent_context: ParentContext | None = None


@dataclass(frozen=True)
class IndexStats:
    files: int
    chunks: int
    added_or_updated: int
    deleted: int
    unchanged: int
