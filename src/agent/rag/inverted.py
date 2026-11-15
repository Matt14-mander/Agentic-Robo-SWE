"""Incremental BM25-style postings, retaining the existing ranking formula."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from agent.rag.embedding import tokenize


class InvertedIndex:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any]]] = {}
        self.terms: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        self.symbols: dict[str, set[str]] = {}
        self.paths: dict[str, set[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.document_postings: dict[str, set[str]] = defaultdict(set)
        self.path_ids: dict[str, set[str]] = defaultdict(set)
        self.total_length = 0
        self.documents_tokenized = 0
        self.last_candidates = 0
        self._scopes: dict[str, tuple[set[str], int]] = {}

    def remove(self, chunk_id: str) -> None:
        if chunk_id not in self.rows:
            return
        _, metadata = self.rows.pop(chunk_id)
        frequencies = self.terms.pop(chunk_id)
        self.total_length -= self.lengths.pop(chunk_id)
        for term in set(frequencies) | self.symbols.pop(chunk_id) | self.paths.pop(chunk_id):
            self.postings[term].discard(chunk_id)
            if not self.postings[term]:
                del self.postings[term]
        for term in frequencies:
            self.document_postings[term].discard(chunk_id)
            if not self.document_postings[term]:
                del self.document_postings[term]
        path = str(metadata.get("path", ""))
        self.path_ids[path].discard(chunk_id)
        if not self.path_ids[path]:
            del self.path_ids[path]
        self._scopes.clear()

    def upsert(self, chunk_id: str, document: str, metadata: dict[str, Any]) -> None:
        if self.rows.get(chunk_id) == (document, metadata):
            return
        self.remove(chunk_id)
        self.rows[chunk_id] = (document, dict(metadata))
        frequencies = Counter(tokenize(document))
        self.documents_tokenized += 1
        self.terms[chunk_id] = frequencies
        self.lengths[chunk_id] = frequencies.total()
        self.symbols[chunk_id] = set(tokenize(str(metadata.get("symbol", ""))))
        self.paths[chunk_id] = set(tokenize(str(metadata.get("path", ""))))
        self.total_length += self.lengths[chunk_id]
        self.path_ids[str(metadata.get("path", ""))].add(chunk_id)
        for term in set(frequencies) | self.symbols[chunk_id] | self.paths[chunk_id]:
            self.postings[term].add(chunk_id)
        for term in frequencies:
            self.document_postings[term].add(chunk_id)
        self._scopes.clear()

    def search(self, query: str, prefix: str | None) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        scope = None
        count, total_length = len(self.rows), self.total_length
        if prefix:
            prefix = prefix.replace("\\", "/")
            if prefix not in self._scopes:
                if len(self._scopes) >= 64:
                    self._scopes.pop(next(iter(self._scopes)))
                ids = set().union(*(ids for path, ids in self.path_ids.items() if path.startswith(prefix)))
                self._scopes[prefix] = (ids, sum(self.lengths[key] for key in ids))
            scope, total_length = self._scopes[prefix]
            count = len(scope)
        candidates = set().union(*(self.postings.get(term, set()) for term in terms))
        if scope is not None:
            candidates &= scope
        self.last_candidates = len(candidates)
        if not terms or not count:
            return []
        average_length = total_length / count
        df = {term: len(self.document_postings.get(term, set())) if scope is None else
              len(self.document_postings.get(term, set()) & scope) for term in terms}
        ranked = []
        term_set = set(terms)
        for key in candidates:
            frequencies = self.terms[key]
            score = 0.0
            for term in terms:
                frequency = frequencies[term]
                if frequency:
                    idf = math.log(1.0 + (count - df[term] + 0.5) / (df[term] + 0.5))
                    ratio = self.lengths[key] / average_length if average_length else 1.0
                    score += idf * (frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * ratio)))
            score += 2.5 * len(term_set & self.symbols[key])
            score += 0.6 * len(term_set & self.paths[key])
            if score > 0:
                ranked.append((key, score))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        maximum = ranked[0][1] if ranked else 1.0
        return [(key, score / maximum) for key, score in ranked]
