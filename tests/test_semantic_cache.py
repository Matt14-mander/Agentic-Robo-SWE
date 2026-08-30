"""Phase 4.1 workspace-scoped semantic response cache tests."""

from __future__ import annotations

import importlib

from langchain_core.messages import AIMessage

from agent.semantic_cache import (
    SemanticCacheMatch,
    SemanticCacheScope,
    SemanticResponseCache,
    workspace_fingerprint,
)


_SCOPE = SemanticCacheScope(
    workspace="workspace-a",
    model="model-a",
    prompt="prompt-a",
    tools="tools-a",
)


def test_semantic_cache_matches_similar_query_and_isolates_workspace(tmp_path):
    cache = SemanticResponseCache(tmp_path / "semantic.sqlite", threshold=0.75)
    try:
        cache.store("explain official validator gate", "cached explanation", _SCOPE)

        exact = cache.lookup(" explain  official validator gate ", _SCOPE)
        similar = cache.lookup("explain the official validator gate", _SCOPE)
        isolated = cache.lookup(
            "explain official validator gate",
            SemanticCacheScope("workspace-b", "model-a", "prompt-a", "tools-a"),
        )

        assert exact == SemanticCacheMatch("cached explanation", 1.0, True)
        assert similar is not None and similar.response == "cached explanation"
        assert not similar.exact
        assert isolated is None
        assert cache.stats().hits == 2
        assert cache.stats().misses == 1
    finally:
        cache.close()


def test_semantic_cache_is_lru_bounded(tmp_path):
    cache = SemanticResponseCache(tmp_path / "bounded.sqlite", threshold=1.0, max_entries=1)
    try:
        cache.store("first query", "first", _SCOPE)
        cache.store("second query", "second", _SCOPE)

        assert cache.lookup("first query", _SCOPE) is None
        assert cache.lookup("second query", _SCOPE) is not None
        assert cache.stats().entries == 1
    finally:
        cache.close()


def test_default_threshold_accepts_punctuation_only_near_duplicate(tmp_path):
    cache = SemanticResponseCache(tmp_path / "conservative.sqlite")
    try:
        cache.store("explain official validator gate", "answer", _SCOPE)
        match = cache.lookup("explain official validator gate?", _SCOPE)

        assert match is not None
        assert match.similarity == 1.0
        assert not match.exact
    finally:
        cache.close()


def test_workspace_fingerprint_changes_with_relevant_content(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    path = source / "controller.py"
    path.write_text("gain = 1.0\n", encoding="utf-8")
    first = workspace_fingerprint(tmp_path)

    path.write_text("gain = 2.0\n", encoding="utf-8")
    second = workspace_fingerprint(tmp_path)

    generated = tmp_path / ".agent_state"
    generated.mkdir()
    (generated / "ignored.json").write_text("changed", encoding="utf-8")
    third = workspace_fingerprint(tmp_path)

    assert first != second
    assert second == third


def test_read_only_planner_returns_cache_hit_without_calling_llm(monkeypatch):
    planner_module = importlib.import_module("agent.nodes.planner")

    class FakeCache:
        def lookup(self, query, scope):
            return SemanticCacheMatch("cached read-only answer", 0.991, False)

    monkeypatch.setattr(planner_module, "get_semantic_cache", lambda: FakeCache())
    monkeypatch.setattr(planner_module, "semantic_cache_scope", lambda **kwargs: _SCOPE)
    monkeypatch.setattr(
        planner_module,
        "get_chat_model",
        lambda temperature=0.0: (_ for _ in ()).throw(AssertionError("LLM must not run")),
    )

    update = planner_module.planner({
        "task": "explain validator gate",
        "loop_step": 0,
        "read_only_mode": True,
        "semantic_cache_enabled": True,
    })

    assert update["suggestion"] == "cached read-only answer"
    assert update["semantic_cache_hit"] is True
    assert update["semantic_cache_similarity"] == 0.991


def test_read_only_planner_stores_only_final_text(monkeypatch):
    planner_module = importlib.import_module("agent.nodes.planner")
    stored = []
    bound_names = []

    class FakeCache:
        def lookup(self, query, scope):
            return None

        def store(self, query, response, scope):
            stored.append((query, response, scope))

    class FakeModel:
        def bind_tools(self, tools):
            bound_names.extend(tool.name for tool in tools)
            return self

        def invoke(self, messages):
            return AIMessage(content="fresh read-only answer")

    monkeypatch.setattr(planner_module, "get_semantic_cache", lambda: FakeCache())
    monkeypatch.setattr(planner_module, "semantic_cache_scope", lambda **kwargs: _SCOPE)
    monkeypatch.setattr(planner_module, "get_chat_model", lambda temperature=0.0: FakeModel())

    update = planner_module.planner({
        "task": "explain validator gate",
        "loop_step": 0,
        "read_only_mode": True,
        "semantic_cache_enabled": True,
    })

    assert bound_names == [
        "list_dir", "search_code_knowledge", "read_file_chunk", "grep_codebase"
    ]
    assert stored == [("explain validator gate", "fresh read-only answer", _SCOPE)]
    assert update["semantic_cache_hit"] is False


def test_read_only_planner_never_stores_tool_calls(monkeypatch):
    planner_module = importlib.import_module("agent.nodes.planner")

    class FakeCache:
        def lookup(self, query, scope):
            return None

        def store(self, query, response, scope):
            raise AssertionError("tool calls must never enter the response cache")

    class FakeModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "read_file_chunk",
                    "args": {"path": "README.md"},
                    "id": "read-1",
                    "type": "tool_call",
                }],
            )

    monkeypatch.setattr(planner_module, "get_semantic_cache", lambda: FakeCache())
    monkeypatch.setattr(planner_module, "semantic_cache_scope", lambda **kwargs: _SCOPE)
    monkeypatch.setattr(planner_module, "get_chat_model", lambda temperature=0.0: FakeModel())

    update = planner_module.planner({
        "task": "inspect README",
        "loop_step": 0,
        "read_only_mode": True,
        "semantic_cache_enabled": True,
    })

    assert update.get("suggestion") is None
    assert update["messages"][0].tool_calls[0]["name"] == "read_file_chunk"
