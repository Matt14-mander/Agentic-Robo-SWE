"""Validated retrieval defaults, independent of model and index initialization."""

from __future__ import annotations

import os
from typing import Literal, cast

from dotenv import load_dotenv

from agent.tools._paths import PROJECT_ROOT

SearchStrategy = Literal["lexical", "vector", "hybrid"]
STRATEGIES = ("lexical", "vector", "hybrid")


def resolve_strategy(explicit: str | None = None, session: str | None = None) -> SearchStrategy:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    value = explicit if explicit is not None else session
    if value is None:
        value = os.getenv("RAG_STRATEGY", "hybrid")
    if value not in STRATEGIES:
        raise ValueError(f"Invalid RAG strategy {value!r}; expected lexical, vector or hybrid")
    return cast(SearchStrategy, value)
