"""Token-bounded children of existing source chunks, retaining exact source slices."""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from typing import Any

from agent.rag.models import CodeChunk

TOKEN_CHUNK_SCHEMA = "token-children-v1"


class TokenChunker:
    def __init__(self, tokenizer: Any, max_tokens: int, overlap_tokens: int = 32):
        # A separate tokenizer must have neither padding nor truncation enabled.
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.budget = max_tokens - tokenizer.num_special_tokens_to_add(False)
        if self.budget < 1 or overlap_tokens < 0:
            raise ValueError("Token budget must leave room for source text")
        self.overlap = min(overlap_tokens, self.budget // 8)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def split(self, parent: CodeChunk) -> list[CodeChunk]:
        text = parent.content
        encoded = self.tokenizer.encode(text)
        if len(encoded.ids) <= self.max_tokens:
            return [parent]
        offsets = [(start, end) for start, end in encoded.offsets if end > start]
        ends = [end for _, end in offsets]
        children = []
        start = 0
        while start < len(text):
            first = bisect_right(ends, start)
            last = min(first + self.budget, len(offsets))
            end = len(text) if last == len(offsets) else offsets[last - 1][1]
            # Prefer a complete line, but split within a long line when necessary.
            newline = text.rfind("\n", start, end)
            if newline >= start and end < len(text):
                end = newline + 1
            # Substring boundaries can change tokenization. Re-encode the actual
            # slice, including special tokens; approximate counts are insufficient.
            if self.count(text[start:end]) > self.max_tokens:
                low, high = start, end
                while high - low > 1:
                    middle = (low + high) // 2
                    if self.count(text[start:middle]) <= self.max_tokens:
                        low = middle
                    else:
                        high = middle
                end = low
            if end <= start:
                raise ValueError("Token budget cannot encode one source character")
            content = text[start:end]
            start_line = parent.start_line + text.count("\n", 0, start)
            end_line = parent.start_line + text.count("\n", 0, end - 1)
            children.append(CodeChunk(
                id=hashlib.sha256(f"{TOKEN_CHUNK_SCHEMA}:{parent.id}:{start}:{end}".encode()).hexdigest()[:32],
                path=parent.path, symbol=parent.symbol, kind=parent.kind,
                start_line=start_line, end_line=end_line, content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                parent_id=parent.id, parent_hash=parent.content_hash,
                start_column=start - text.rfind("\n", 0, start),
                end_column=end - 1 - text.rfind("\n", 0, end - 1),
            ))
            if end == len(text):
                break
            overlap_index = max(first, bisect_right(ends, end) - self.overlap)
            overlap_start = offsets[overlap_index][0] if overlap_index < len(offsets) else end
            start = overlap_start if start < overlap_start < end else end
        return children
