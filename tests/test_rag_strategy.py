"""Agent-visible strategy selection, session isolation and checkpoint defaults."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.rag import settings
from agent.rag.models import SearchResult
from agent.tools import rag_ops
from scripts import run_agent


@pytest.fixture(autouse=True)
def clean_strategy_env(monkeypatch):
    monkeypatch.delenv("RAG_STRATEGY", raising=False)
    monkeypatch.setattr(settings, "load_dotenv", lambda *a, **k: None)


@pytest.fixture
def recording_index(monkeypatch):
    calls = []

    class Index:
        def search(self, query, **kwargs):
            calls.append((query, kwargs))
            return [SearchResult("source.cpp", "<file>", "cpp", 1, 1, "// evidence", 1.0)]

    monkeypatch.setattr(rag_ops, "get_code_index", lambda: Index())
    return calls


@pytest.mark.parametrize("strategy", settings.STRATEGIES)
def test_tool_explicit_strategy_overrides_session_and_environment(recording_index, monkeypatch, strategy):
    monkeypatch.setenv("RAG_STRATEGY", "lexical")
    result = rag_ops.search_code_knowledge.invoke(
        {"query": "control", "strategy": strategy, "top_k": 2, "path_prefix": "src"},
        config={"configurable": {"rag_strategy": "vector"}},
    )
    assert recording_index == [("control", {"strategy": strategy, "top_k": 2, "path_prefix": "src"})]
    assert result.startswith(f"# {strategy.capitalize()} code search results")


@pytest.mark.parametrize(("session", "env", "expected"), [
    (None, None, "hybrid"), (None, "lexical", "lexical"), ("vector", "lexical", "vector"),
])
def test_omitted_tool_strategy_uses_default(recording_index, monkeypatch, session, env, expected):
    if env:
        monkeypatch.setenv("RAG_STRATEGY", env)
    rag_ops.search_code_knowledge.invoke({"query": "q"},
                                        config={"configurable": {"rag_strategy": session}})
    assert recording_index[0][1]["strategy"] == expected


def test_tool_schema_exposes_choices_but_not_internal_config(recording_index):
    properties = rag_ops.search_code_knowledge.tool_call_schema.model_json_schema()["properties"]
    assert "config" not in properties
    assert properties["strategy"]["anyOf"][0]["enum"] == list(settings.STRATEGIES)
    with pytest.raises(ValueError):
        rag_ops.search_code_knowledge.invoke({"query": "q", "strategy": "unknown"})
    assert not recording_index


def test_invalid_env_fails_before_index_open(monkeypatch):
    monkeypatch.setenv("RAG_STRATEGY", "unknown")
    monkeypatch.setattr(rag_ops, "get_code_index", lambda: pytest.fail("must validate first"))
    assert "Invalid RAG strategy" in rag_ops.search_code_knowledge.invoke({"query": "q"})


def test_retrieval_cache_is_partitioned_by_strategy():
    from agent.rag.index import CodeIndex
    from tests.test_rag import MemoryCollection
    index = CodeIndex(MemoryCollection())
    index.sync(["src/agent/rag/cache.py"])
    expected = {strategy: index.search("RetrievalCache", strategy=strategy)
                for strategy in settings.STRATEGIES}
    assert index.cache_stats().misses == 3 and index.cache_stats().hits == 0
    for strategy in settings.STRATEGIES:
        assert index.search("RetrievalCache", strategy=strategy) == expected[strategy]
    assert index.cache_stats().hits == 3 and index.cache_stats().entries == 3


def tool_state(strategy, query="q", explicit=None):
    args = {"query": query}
    if explicit:
        args["strategy"] = explicit
    return {"rag_strategy": strategy, "read_only_mode": True, "messages": [AIMessage(
        content="", tool_calls=[{"id": query, "type": "tool_call", "name": "search_code_knowledge",
                                 "args": args}],
    )]}


def search_graph(monkeypatch, explicit=None):
    graph_module = importlib.import_module("agent.graph")

    def planner(state):
        if state.get("loop_step", 0):
            return {"messages": [AIMessage(content="done")], "loop_step": 2}
        return {"messages": tool_state(state["rag_strategy"], state["task"], explicit)["messages"],
                "loop_step": 1}

    monkeypatch.setattr(graph_module, "planner", planner)
    return graph_module.build_graph()


def test_graph_parallel_sessions_do_not_leak_strategy(recording_index, monkeypatch):
    from agent.tools import all_registered_tools
    # Initialize the existing lazy registry before exercising strategy isolation.
    all_registered_tools()
    graph = search_graph(monkeypatch)
    config = {"configurable": {"rag_strategy": "hybrid", "thread_id": "ignored"}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(graph.invoke, {"rag_strategy": s, "task": s, "read_only_mode": True},
                               config) for s in ("lexical", "vector")]
        for future in futures:
            assert any("code search results" in m.content for m in future.result()["messages"])
    assert {query: args["strategy"] for query, args in recording_index} == {
        "lexical": "lexical", "vector": "vector",
    }
    assert config["configurable"]["rag_strategy"] == "hybrid"


def test_graph_tool_override_remains_available(recording_index, monkeypatch):
    search_graph(monkeypatch, explicit="vector").invoke({"rag_strategy": "lexical", "task": "q"})
    assert recording_index[0][1]["strategy"] == "vector"


def test_planner_freezes_default_and_prompt_partitions_response_cache(monkeypatch):
    planner = importlib.import_module("agent.nodes.planner")
    monkeypatch.setenv("RAG_STRATEGY", "vector")
    llm = SimpleNamespace(bind_tools=lambda tools: SimpleNamespace(invoke=lambda messages:
                          AIMessage(content="done")))
    monkeypatch.setattr(planner, "get_chat_model", lambda **kwargs: llm)
    result = planner.planner({"task": "inspect", "loop_step": 0})
    assert result["rag_strategy"] == "vector"
    vector_prompt = planner.build_system_prompt({"rag_strategy": "vector"})
    lexical_prompt = planner.build_system_prompt({"rag_strategy": "lexical"})
    assert vector_prompt != lexical_prompt and "Default RAG strategy: vector" in vector_prompt


def test_checkpoint_retains_strategy_across_graph_reopen(recording_index, monkeypatch):
    graph_module = importlib.import_module("agent.graph")

    def planner(state):
        if state.get("loop_step", 0):
            return {"messages": [AIMessage(content="done")], "loop_step": 2}
        return {"messages": tool_state(state["rag_strategy"])["messages"], "loop_step": 1}

    monkeypatch.setattr(graph_module, "planner", planner)
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "rag-saved"}}
    graph_module.build_graph(checkpointer=saver).invoke({"rag_strategy": "lexical", "task": "inspect"}, config)
    monkeypatch.setenv("RAG_STRATEGY", "vector")
    snapshot = graph_module.build_graph(checkpointer=saver).get_state(config)
    assert snapshot.values["rag_strategy"] == recording_index[0][1]["strategy"] == "lexical"


def test_updating_strategy_at_approval_preserves_pending_resume(recording_index, monkeypatch):
    from langgraph.types import Command
    graph_module = importlib.import_module("agent.graph")

    def planner(state):
        step = state.get("loop_step", 0)
        if step == 0:
            return {"messages": [AIMessage(content="", tool_calls=[{
                "id": "patch", "type": "tool_call", "name": "write_patch",
                "args": {"path": "README.md", "old_string": "unused", "new_string": "unused"},
            }])], "loop_step": 1}
        if step == 1:
            return {"messages": tool_state(state["rag_strategy"])["messages"], "loop_step": 2}
        return {"messages": [AIMessage(content="done")], "loop_step": 3}

    monkeypatch.setattr(graph_module, "planner", planner)
    graph = graph_module.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "approval-strategy"}}
    graph.invoke({"task": "inspect", "rag_strategy": "hybrid", "hitl_enabled": True}, config)
    assert graph.get_state(config).interrupts
    graph.update_state(config, {"rag_strategy": "lexical"})
    graph.invoke(Command(resume={"approved": False}), config)
    assert recording_index[0][1]["strategy"] == "lexical"
    assert not graph.get_state(config).interrupts


@pytest.mark.parametrize(("mode", "override", "expected"), [
    ("new", None, "vector"), ("new", "lexical", "lexical"),
    ("continue", None, "hybrid"), ("continue", "lexical", "lexical"),
    ("resume", None, "hybrid"), ("resume", "vector", "vector"),
])
def test_cli_persists_new_default_and_respects_resume_override(monkeypatch, mode, override, expected):
    monkeypatch.setenv("RAG_STRATEGY", "vector")
    values = {} if mode == "new" else {"rag_strategy": "hybrid", "task": "inspect"}
    snapshot = SimpleNamespace(values=values, next=("planner",),
                               interrupts=(SimpleNamespace(value={}),) if mode == "resume" else ())
    updates, invoked = [], []

    class Graph:
        def get_state(self, config):
            return snapshot

        def update_state(self, config, update):
            updates.append(update)
            values.update(update)

    @contextmanager
    def opened(path):
        yield Graph()

    def run(agent, graph_input, config, **kwargs):
        invoked.append(graph_input)
        if mode == "new":
            values.update(graph_input)
        snapshot.interrupts = ()
        return {}

    monkeypatch.setattr(run_agent, "open_persistent_graph", opened)
    monkeypatch.setattr(run_agent, "_run", run)
    argv = ["inspect"] if mode == "new" else ["--thread-id", "saved"]
    if mode == "continue":
        argv += ["--continue"]
    if mode == "resume":
        argv += ["--resume", "approve"]
    if override:
        argv += ["--rag-strategy", override]
    assert run_agent._run_session(run_agent._build_parser().parse_args(argv)) == 0
    assert values["rag_strategy"] == expected and len(invoked) == 1
    assert len(updates) == int(mode != "new" and override is not None)
