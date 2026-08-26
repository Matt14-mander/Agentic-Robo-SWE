"""Data contracts shared by the Phase 3 indexer and retrieval tool."""

from __future__ import annotations

from dataclasses import dataclass


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

    def metadata(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class SearchResult:
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str
    score: float


@dataclass(frozen=True)
class IndexStats:
    files: int
    chunks: int
    added_or_updated: int
    deleted: int
    unchanged: int
