"""Incremental Chroma index for structure-aware source chunks."""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from agent.rag.chunking import chunk_codebase, chunk_python_file
from agent.rag.cache import RetrievalCache, RetrievalCacheStats
from agent.rag.embedding import Embedder, EmbeddingConfig, HashingEmbedder, create_embedder
from agent.rag.freshness import SourceDetector
from agent.rag.inverted import InvertedIndex
from agent.rag.storage import TrackedCollection, open_collection
from agent.rag.models import CodeChunk, IndexStats, ParentContext, SearchResult
from agent.rag.settings import SearchStrategy
from agent.tools._paths import PROJECT_ROOT, resolve_within_root
from agent.workspace import Workspace, current_workspace
from agent.rag.workspace_source import (
    chunk_workspace, chunk_workspace_source, read_workspace_source, workspace_index_path,
)

DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION = "agentic_robo_swe_code"


@dataclass(frozen=True)
class _Candidate:
    chunk_id: str
    result: SearchResult


class CodeIndex:
    """Backend-neutral index logic operating on a Chroma-compatible collection."""

    def __init__(
        self,
        collection: Any,
        embedder: Embedder | None = None,
        cache: RetrievalCache | None = None,
        *,
        workspace: Workspace | None = None,
        strict_freshness: bool = False,
    ) -> None:
        self.collection = collection
        self.workspace = workspace
        self.embedder = embedder or HashingEmbedder()
        self.cache = cache or RetrievalCache()
        self._revision: str | None = None
        # (source hash, indexed chunk revision), scoped to the paths already indexed.
        self._source_snapshots: dict[str, tuple[str, str]] = {}
        self._parents: dict[str, CodeChunk] = {}
        self._parents_by_path: dict[str, set[str]] = {}
        self._by_path: dict[str, dict[str, str]] = {}
        self._lexical = InvertedIndex()
        self._detector = SourceDetector(workspace.root if workspace else PROJECT_ROOT,
                                       self._resolve_source, self._read_source, strict=strict_freshness)
        self._lock = RLock()
        self._guard = collection.guard if isinstance(collection, TrackedCollection) else nullcontext()
        self._loaded = False
        self._observed_token: str | None = None
        self._collection_reads = 0

    def sync(
        self,
        roots: Iterable[str] | None = None,
        *,
        rebuild: bool = False,
    ) -> IndexStats:
        with self._lock, self._guard:
            self._loaded = False
            return self._sync(roots, rebuild=rebuild)

    def _sync(self, roots: Iterable[str] | None, *, rebuild: bool) -> IndexStats:
        self._check_workspace()
        roots = tuple(roots) if roots is not None else ((".",) if self.workspace else ("src",))
        files, chunks = (
            chunk_workspace(self.workspace, roots) if self.workspace else chunk_codebase(roots)
        )
        parents = {chunk.id: chunk for chunk in chunks}
        chunks = self._embedding_chunks(chunks)
        existing_payload = self.collection.get(include=["metadatas"])
        existing_ids = [str(item) for item in existing_payload.get("ids") or []]
        existing_metadatas = existing_payload.get("metadatas") or []
        existing_hashes = {
            chunk_id: str(metadata.get("content_hash", ""))
            for chunk_id, metadata in zip(existing_ids, existing_metadatas)
            if isinstance(metadata, dict)
        }
        if rebuild and existing_ids:
            self.collection.delete(ids=existing_ids)
            existing_hashes = {}

        current_ids = {chunk.id for chunk in chunks}
        stale_ids = sorted(set(existing_hashes) - current_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        changed = [
            chunk for chunk in chunks
            if existing_hashes.get(chunk.id) != chunk.revision_hash
        ]
        for batch in _batches(changed, 128):
            documents = [chunk.content for chunk in batch]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=documents,
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=self.embedder.embed(documents),
            )

        self._parents = parents
        self._parents_by_path = {}
        for key, parent in parents.items():
            self._parents_by_path.setdefault(parent.path, set()).add(key)
        self._set_revision(_revision_for_chunks(chunks))

        return IndexStats(
            files=len(files),
            chunks=len(chunks),
            added_or_updated=len(changed),
            deleted=len(stale_ids) + (len(existing_ids) if rebuild else 0),
            unchanged=len(chunks) - len(changed),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        path_prefix: str | None = None,
        strategy: SearchStrategy = "hybrid",
        use_cache: bool = True,
    ) -> list[SearchResult]:
        with self._lock, self._guard:
            for _ in range(3):
                results = self._search(query, top_k=top_k, path_prefix=path_prefix,
                                       strategy=strategy, use_cache=use_cache)
                if self._verify_results(results):
                    return results
            raise RuntimeError("Sources changed repeatedly during retrieval; retry the query")

    def _search(self, query: str, *, top_k: int, path_prefix: str | None,
                strategy: SearchStrategy, use_cache: bool) -> list[SearchResult]:
        self._check_workspace()
        if not query.strip():
            raise ValueError("query must not be empty")
        if strategy not in {"vector", "lexical", "hybrid"}:
            raise ValueError(f"Unsupported search strategy: {strategy}")
        if self.workspace and path_prefix:
            prefix = self.workspace.resolve(path_prefix).relative_to(self.workspace.root).as_posix()
            path_prefix = None if prefix == "." else prefix
        top_k = min(max(1, top_k), 20)
        revision = self._ensure_revision()
        key = self.cache.make_key(
            revision=f"{self.embedder.identity}:{revision}",
            query=query,
            top_k=top_k,
            path_prefix=path_prefix,
            strategy=strategy,
        )
        if use_cache:
            found, cached = self.cache.get(key)
            if found:
                return cached
        else:
            self.cache.record_bypass()

        started = time.perf_counter()
        results = self._search_uncached(
            query,
            top_k=top_k,
            path_prefix=path_prefix,
            strategy=strategy,
        )
        results = [self._with_parent(result) for result in results]
        if use_cache:
            self.cache.put(
                key,
                results,
                compute_ms=(time.perf_counter() - started) * 1000,
            )
        return results

    def _search_uncached(
        self,
        query: str,
        *,
        top_k: int,
        path_prefix: str | None,
        strategy: SearchStrategy,
    ) -> list[SearchResult]:
        count = int(self.collection.count())
        if count == 0:
            return []

        if strategy == "lexical":
            return [candidate.result for candidate in self._lexical_candidates(query, path_prefix)[:top_k]]

        vector_ranked = self._vector_candidates(query, top_k, path_prefix)
        if strategy == "vector":
            return [candidate.result for candidate in vector_ranked[:top_k]]

        lexical_ranked = self._lexical_candidates(query, path_prefix)
        return self._fuse_rankings(vector_ranked, lexical_ranked, top_k)

    def cache_stats(self) -> RetrievalCacheStats:
        return self.cache.stats()

    def performance_stats(self) -> dict[str, int]:
        return {"source_metadata_checks": self._detector.metadata_checks,
                "source_content_reads": self._detector.content_reads,
                "source_bytes_read": self._detector.bytes_read,
                "collection_full_reads": self._collection_reads,
                "lexical_documents_tokenized": self._lexical.documents_tokenized,
                "last_lexical_candidates": self._lexical.last_candidates}

    def retrieval_manifest(self) -> dict:
        return {"embedding": self.embedder.describe(),
                "source_revision": _revision_for_chunks(self._parents.values()),
                "index_revision": self._revision, "parent_count": len(self._parents),
                "chunk_count": int(self.collection.count())}

    def _embedding_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        chunker = getattr(self.embedder, "chunker", None)
        return [child for chunk in chunks for child in chunker.split(chunk)] if chunker else chunks

    def _with_parent(self, result: SearchResult) -> SearchResult:
        if not result.parent_id:
            return result
        parent = self._parents[result.parent_id]
        return replace(result, parent_context=ParentContext(
            parent.id, parent.start_line, parent.end_line, parent.content,
        ))

    def _check_workspace(self) -> None:
        active = current_workspace()
        if active is not None and active != self.workspace:
            raise ValueError("RAG workspace binding mismatch")

    def _resolve_source(self, relative: str) -> Path:
        if self.workspace:
            return self.workspace.resolve(relative)
        path = (PROJECT_ROOT / relative).resolve()
        path.relative_to(PROJECT_ROOT)
        return path

    def _read_source(self, relative: str) -> tuple[Path, str]:
        if self.workspace:
            return read_workspace_source(self.workspace, relative)
        path = self._resolve_source(relative)
        try:
            return path, path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return path, ""

    def _load_collection(self) -> None:
        self._collection_reads += 1
        payload = self.collection.get(include=["documents", "metadatas"])
        rows = {}
        by_path: dict[str, dict[str, str]] = {}
        for key, document, metadata in zip(payload.get("ids") or [],
                                           payload.get("documents") or [],
                                           payload.get("metadatas") or []):
            if not isinstance(metadata, dict) or not metadata.get("path"):
                raise ValueError("Indexed chunk has no source path; rebuild the code index")
            relative = str(metadata["path"])
            self._resolve_source(relative)  # Validate stored paths before reading any source.
            key = str(key)
            rows[key] = (str(document), dict(metadata))
            by_path.setdefault(relative, {})[key] = str(metadata.get("content_hash", ""))
        changed_paths = set(self._by_path) ^ set(by_path)
        for key, row in rows.items():
            if self._lexical.rows.get(key) != row:
                if key in self._lexical.rows:
                    changed_paths.add(str(self._lexical.rows[key][1]["path"]))
                changed_paths.add(str(row[1]["path"]))
                self._lexical.upsert(key, *row)
        for key in set(self._lexical.rows) - rows.keys():
            changed_paths.add(str(self._lexical.rows[key][1]["path"]))
            self._lexical.remove(key)
        for relative in changed_paths:
            self._source_snapshots.pop(relative, None)
        for relative in set(self._parents_by_path) - by_path.keys():
            for key in self._parents_by_path.pop(relative):
                self._parents.pop(key, None)
        self._by_path = by_path
        if not self._loaded:
            self._source_snapshots.clear()
            self._detector.stamps.clear()
        self._loaded = True

    def _ensure_revision(self) -> str:
        tracked = isinstance(self.collection, TrackedCollection)
        token = self.collection.revision_token() if tracked else None
        if tracked:
            self.collection.refresh_reader(token)
        reload = not self._loaded or not tracked or token != self._observed_token
        if reload:
            self._load_collection()
        dirty = self._detector.changed(self._by_path)
        dirty.update(set(self._by_path) - self._source_snapshots.keys())
        for relative in sorted(dirty):
            self._refresh_path(relative)
        if reload or dirty or self._revision is None:
            revision = _revision_for_pairs((key, value) for rows in self._by_path.values()
                                           for key, value in rows.items())
            if revision != self._revision:
                self._set_revision(revision)
        if tracked:
            self._observed_token = self.collection.revision_token()
        assert self._revision is not None
        return self._revision

    def _refresh_path(self, relative: str) -> None:
        existing = self._by_path[relative]
        path, source, stamp = self._detector.read_stable(relative)
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        snapshot = (source_hash, _revision_for_pairs(existing.items()))
        if self._source_snapshots.get(relative) == snapshot:
            self._detector.accept(relative, stamp)
            return
        parents = (chunk_workspace_source(self.workspace, path, source) if self.workspace
                   else chunk_python_file(path, source_text=source))
        chunks = self._embedding_chunks(parents)
        current = {chunk.id: chunk.revision_hash for chunk in chunks}
        changed = [chunk for chunk in chunks if existing.get(chunk.id) != chunk.revision_hash]
        stale = sorted(set(existing) - current.keys())
        # Publish derived state only after every backend operation succeeds. An
        # exception leaves this path dirty so the next lookup retries before cache.
        for batch in _batches(changed, 128):
            documents = [chunk.content for chunk in batch]
            self.collection.upsert(ids=[chunk.id for chunk in batch], documents=documents,
                                   metadatas=[chunk.metadata() for chunk in batch],
                                   embeddings=self.embedder.embed(documents))
        if stale:
            self.collection.delete(ids=stale)
        for key in stale:
            self._lexical.remove(key)
        for chunk in changed:
            self._lexical.upsert(chunk.id, chunk.content, chunk.metadata())
        for key in self._parents_by_path.pop(relative, set()):
            self._parents.pop(key, None)
        self._parents.update((chunk.id, chunk) for chunk in parents)
        self._parents_by_path[relative] = {chunk.id for chunk in parents}
        if current:
            self._by_path[relative] = current
            self._source_snapshots[relative] = (source_hash, _revision_for_pairs(current.items()))
            self._detector.accept(relative, stamp)
        else:
            self._by_path.pop(relative, None)
            self._source_snapshots.pop(relative, None)
            self._detector.stamps.pop(relative, None)

    def _verify_results(self, results: list[SearchResult]) -> bool:
        valid = True
        for relative in {result.path for result in results}:
            _, source, _ = self._detector.read_stable(relative)
            if hashlib.sha256(source.encode("utf-8")).hexdigest() != self._source_snapshots[relative][0]:
                self._detector.dirty.add(relative)
                valid = False
        return valid

    def _set_revision(self, revision: str) -> None:
        self._revision = revision
        self.cache.prepare_revision(f"{self.embedder.identity}:{revision}")

    def _vector_candidates(
        self,
        query: str,
        top_k: int,
        path_prefix: str | None,
    ) -> list[_Candidate]:
        count = int(self.collection.count())
        candidates = count if path_prefix else min(count, max(top_k * 8, 32))
        query_embeddings = self.embedder.embed([query])
        while True:
            payload = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=candidates,
                include=["documents", "metadatas", "distances"],
            )
            identities = {
                str(metadata.get("parent_id") or chunk_id)
                for chunk_id, metadata in zip(
                    _first_result_list(payload.get("ids")),
                    _first_result_list(payload.get("metadatas")),
                ) if isinstance(metadata, dict)
            }
            if len(identities) >= top_k or candidates == count:
                break
            candidates = min(count, candidates * 2)
        ids = _first_result_list(payload.get("ids"))
        documents = _first_result_list(payload.get("documents"))
        metadatas = _first_result_list(payload.get("metadatas"))
        distances = _first_result_list(payload.get("distances"))
        results: list[_Candidate] = []
        normalized_prefix = path_prefix.replace("\\", "/") if path_prefix else None
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            if not isinstance(metadata, dict):
                continue
            path = str(metadata.get("path", ""))
            if normalized_prefix and not path.startswith(normalized_prefix):
                continue
            results.append(_Candidate(
                chunk_id=str(chunk_id),
                result=_make_result(document, metadata, 1.0 - float(distance)),
            ))
        return _collapse_parents(results)

    def _lexical_candidates(self, query: str, path_prefix: str | None) -> list[_Candidate]:
        return _collapse_parents([
            _Candidate(key, _make_result(*self._lexical.rows[key], score))
            for key, score in self._lexical.search(query, path_prefix)
        ])

    @staticmethod
    def _fuse_rankings(
        vector_ranked: list[_Candidate],
        lexical_ranked: list[_Candidate],
        top_k: int,
    ) -> list[SearchResult]:
        """Weighted reciprocal-rank fusion; scores are normalized to [0, 1]."""
        weights = tuple(
            (ranking, weight)
            for ranking, weight in ((vector_ranked, 1.0), (lexical_ranked, 1.25))
            if ranking
        )
        fused: dict[str, float] = {}
        candidates: dict[str, _Candidate] = {}
        for ranking, weight in weights:
            for rank, candidate in enumerate(ranking, start=1):
                # Preserve the vector-matched child when lexical evidence names
                # a different child of the same parent.
                candidates.setdefault(candidate.chunk_id, candidate)
                fused[candidate.chunk_id] = fused.get(candidate.chunk_id, 0.0) + weight / (60 + rank)
        maximum = sum(weight / 61 for _, weight in weights)
        ordered_ids = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]
        return [
            _with_score(candidates[chunk_id].result, fused[chunk_id] / maximum)
            for chunk_id in ordered_ids
        ]


def _batches(items: list[CodeChunk], size: int) -> Iterable[list[CodeChunk]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _revision_for_chunks(chunks: Iterable[CodeChunk]) -> str:
    return _revision_for_pairs((chunk.id, chunk.revision_hash) for chunk in chunks)


def _collapse_parents(candidates: list[_Candidate]) -> list[_Candidate]:
    seen: set[str] = set()
    results = []
    for candidate in candidates:
        identity = candidate.result.parent_id or candidate.chunk_id
        if identity not in seen:
            seen.add(identity)
            results.append(_Candidate(identity, candidate.result))
    return results


def _revision_for_pairs(pairs: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for chunk_id, content_hash in sorted(pairs):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _first_result_list(value: Any) -> list[Any]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return []


def _make_result(document: Any, metadata: dict[str, Any], score: float) -> SearchResult:
    return SearchResult(
        path=str(metadata.get("path", "")),
        symbol=str(metadata.get("symbol", "<unknown>")),
        kind=str(metadata.get("kind", "code")),
        start_line=int(metadata.get("start_line", 1)),
        end_line=int(metadata.get("end_line", 1)),
        content=str(document),
        score=round(score, 6),
        parent_id=str(metadata.get("parent_id", "")),
        start_column=int(metadata.get("start_column", 1)),
        end_column=int(metadata.get("end_column", 1)),
    )


def _with_score(result: SearchResult, score: float) -> SearchResult:
    return replace(result, score=round(score, 6))


def open_code_index(
    persist_directory: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    *,
    workspace: Workspace | None = None,
    embedding: EmbeddingConfig | None = None,
) -> CodeIndex:
    """Open the persistent Chroma collection with a clear optional-dependency error."""
    try:
        import chromadb  # type: ignore[import-not-found,import-untyped]  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed. Run: uv sync --extra rag"
        ) from exc

    workspace = workspace or current_workspace()
    if workspace:
        if persist_directory is not None or collection_name != DEFAULT_COLLECTION:
            raise ValueError("External RAG storage is bound to workspace artifacts; no persist-dir override")
        path = workspace_index_path(workspace)
    else:
        path = Path(persist_directory) if persist_directory is not None else DEFAULT_CHROMA_DIR
    if not path.is_absolute():
        path = resolve_within_root(str(path))
    embedder = create_embedder(embedding)
    legacy_hash = embedder.identity == HashingEmbedder().identity
    if not legacy_hash:
        collection_name = f"{collection_name}_{embedder.identity}"
    metadata = {"hnsw:space": "cosine"}
    if workspace:
        metadata["source_root"] = str(workspace.root)
        metadata["source_schema"] = "workspace-text-v1"
    collection = open_collection(path, collection_name,
                                 {**metadata, "embedding_identity": embedder.identity})
    if workspace and any((collection.metadata or {}).get(key) != value for key, value in metadata.items()):
        raise ValueError("RAG collection workspace binding or schema mismatch")
    stored_identity = (collection.metadata or {}).get("embedding_identity")
    if stored_identity != embedder.identity and not (stored_identity is None and legacy_hash):
        raise ValueError("RAG embedding identity mismatch; choose a fresh collection")
    freshness = os.getenv("RAG_FRESHNESS", "incremental")
    if freshness not in {"incremental", "strict"}:
        raise ValueError("RAG_FRESHNESS must be incremental or strict")
    return CodeIndex(collection, embedder=embedder, workspace=workspace,
                     strict_freshness=freshness == "strict")
