"""External C++ retrieval boundaries, freshness, CLI and tool integration (no LLM)."""

from __future__ import annotations

import importlib
import json
import os
import shutil

import pytest

from agent.rag import CodeIndex, open_code_index
from agent.rag.workspace_source import workspace_index_path
from agent.tools import rag_ops, resolve_tools, write_patch
from agent.workspace import HARNESS_ROOT, Workspace, use_workspace
from tests.test_rag import MemoryCollection


def make_repo(tmp_path, name="repo"):
    root = tmp_path / name
    (root / "src").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "src/controller.cpp").write_text(
        '#include "controller.hpp"\n// torque saturation limiter\n'
        "double clamp_torque(double torque) { return torque; }\n", encoding="utf-8",
    )
    (root / "include/controller.hpp").write_text(
        "#pragma once\ndouble clamp_torque(double torque);\n", encoding="utf-8",
    )
    (root / "README.md").write_text("# Controller\nJoint calibration guide.\n", encoding="utf-8")
    (root / "CMakeLists.txt").write_text("project(controller)\n", encoding="utf-8")
    return Workspace.create(root, tmp_path / "shared-artifacts")


@pytest.fixture
def workspace_index(tmp_path):
    workspace = make_repo(tmp_path)
    index = CodeIndex(MemoryCollection(), workspace=workspace)
    index.sync()
    return workspace, index


@pytest.mark.parametrize("strategy", ["vector", "lexical", "hybrid"])
def test_external_cpp_header_document_search(workspace_index, strategy):
    workspace, index = workspace_index
    with use_workspace(workspace):
        result = index.search("torque saturation limiter", strategy=strategy)[0]
        assert result.path == "src/controller.cpp" and result.kind == "cpp"
        assert result.symbol == "<file>" and result.start_line == 1
        assert index.search("clamp_torque", path_prefix="include", strategy=strategy)[0].path.endswith(".hpp")
        assert index.search("calibration guide", strategy=strategy)[0].path == "README.md"
    assert index.sync().unchanged == 4


def test_cpp_windows_cover_macros_templates_and_comments(workspace_index):
    workspace, index = workspace_index
    source = "#define LIMIT 42\n/* multi line comment\n*/\n"
    source += "template<class T> T clamp(T value) {\n"
    source += "".join(f"    // preserved line {i}\n" for i in range(250))
    source += "    return value;\n}\n"
    (workspace.root / "src/controller.cpp").write_text(source, encoding="utf-8")
    index.sync(["src"])
    chunks = index.search("preserved", top_k=20, strategy="lexical")
    covered = set()
    for chunk in chunks:
        assert chunk.end_line - chunk.start_line + 1 <= 120
        expected = "\n".join(source.splitlines()[chunk.start_line - 1:chunk.end_line]).rstrip() + "\n"
        assert chunk.content == expected
        covered.update(range(chunk.start_line, chunk.end_line + 1))
    assert covered == set(range(1, len(source.splitlines()) + 1))


def test_cpp_patch_external_edit_and_delete_refresh_cached_index(workspace_index):
    workspace, index = workspace_index
    with use_workspace(workspace):
        index.search("torque")
        result = write_patch.invoke({"path": "src/controller.cpp", "old_string": "return torque;",
                                     "new_string": "return torque * 0.5;"})
        assert result.startswith("OK:")
        assert "torque * 0.5" in index.search("torque", path_prefix="src")[0].content
        header = workspace.root / "include/controller.hpp"
        header.write_text("// new_header_marker\n", encoding="utf-8")
        assert "new_header_marker" in index.search("new_header_marker", strategy="lexical")[0].content
        header.unlink()
        assert index.search("new_header_marker", path_prefix="include") == []
        assert index.search("torque", path_prefix="src")


def test_scoped_sync_never_silently_expands_to_whole_repository(workspace_index):
    workspace, index = workspace_index
    index.sync(["include"])
    assert {result.path for result in index.search("torque", top_k=20)} == {"include/controller.hpp"}
    with pytest.raises(ValueError, match="does not exist"):
        index.sync(["misspelled-source-dir"])
    assert index.collection.count() == 1


def test_private_build_and_unsupported_files_are_not_indexed(workspace_index):
    workspace, index = workspace_index
    for folder in (".git", ".env-private", ".agent_state", "build", "dist"):
        directory = workspace.root / folder
        directory.mkdir()
        (directory / "secret.cpp").write_text("hidden_marker", encoding="utf-8")
    (workspace.root / "config.json").write_text('{"hidden_marker": true}', encoding="utf-8")
    assert index.sync().files == 4
    assert index.search("hidden_marker", strategy="lexical") == []
    for prefix in ("../outside", ".git", ".env-private"):
        with pytest.raises(ValueError):
            index.search("torque", path_prefix=prefix)
        with pytest.raises(ValueError):
            index.sync([prefix])


def test_refresh_checks_stored_paths_and_rejects_links(workspace_index, tmp_path):
    workspace, index = workspace_index
    row = next(iter(index.collection.data.values()))
    original = row[1]["path"]
    row[1]["path"] = "../outside.cpp"
    with pytest.raises(ValueError, match="escapes"):
        index.search("torque")
    row[1]["path"] = original
    outside = tmp_path / "outside.cpp"
    outside.write_text("hidden_marker", encoding="utf-8")
    try:
        (workspace.root / "alias.cpp").symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit symlinks")
    with pytest.raises(ValueError):
        index.sync()


@pytest.mark.parametrize("content", [b"x" * 1_000_001, b"bad\0binary", b"\xff\xfe"],
                         ids=["oversized", "binary", "invalid-utf8"])
def test_invalid_changed_source_fails_without_serving_cached_results(workspace_index, content):
    workspace, index = workspace_index
    index.search("torque")
    (workspace.root / "src/controller.cpp").write_bytes(content)
    with pytest.raises(ValueError):
        index.search("torque")


def test_tool_cache_and_storage_are_bound_to_workspace(tmp_path, monkeypatch):
    first, second = make_repo(tmp_path, "first"), make_repo(tmp_path, "second")
    (second.root / "src/controller.cpp").write_text("// second_repo_marker\n", encoding="utf-8")
    assert workspace_index_path(first) != workspace_index_path(second)
    opened = []

    def open_index(*, workspace, embedding):
        opened.append(workspace)
        index = CodeIndex(MemoryCollection(), workspace=workspace)
        index.sync()
        return index

    monkeypatch.setattr(rag_ops, "open_code_index", open_index)
    rag_ops._get_code_index.cache_clear()
    try:
        with use_workspace(first):
            first_index = rag_ops.get_code_index()
            output = rag_ops.search_code_knowledge.invoke({"query": "saturation"})
            assert "```cpp" in output and "torque saturation" in output
            assert "second_repo_marker" not in output
        with use_workspace(second):
            output = rag_ops.search_code_knowledge.invoke({"query": "second_repo_marker"})
            assert "second_repo_marker" in output and "torque saturation" not in output
            with pytest.raises(ValueError, match="binding"):
                first_index.search("torque")
            with pytest.raises(ValueError, match="binding"):
                CodeIndex(MemoryCollection()).sync()
        with use_workspace(first):
            assert rag_ops.get_code_index() is first_index
        assert opened == [first, second]
    finally:
        rag_ops._get_code_index.cache_clear()


def test_read_only_tool_dispatch_and_prompt_use_workspace_rag(workspace_index, monkeypatch):
    from agent.nodes.planner import build_system_prompt
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    workspace, index = workspace_index
    monkeypatch.setattr(rag_ops, "get_code_index", lambda: index)
    graph_module = importlib.import_module("agent.graph")

    def fake_planner(state):
        if state.get("loop_step", 0):
            return {"messages": [AIMessage(content="done")], "suggestion": "done", "loop_step": 2}
        return {"messages": [AIMessage(content="", tool_calls=[{
            "name": "search_code_knowledge", "args": {"query": "saturation"},
            "id": "search", "type": "tool_call"}])], "loop_step": 1}

    monkeypatch.setattr(graph_module, "planner", fake_planner)
    with use_workspace(workspace):
        names = {tool.name for tool in resolve_tools(mode="read_only", cpp=True)}
        assert "search_code_knowledge" in names
        assert not {"cpp_build", "cpp_semantic_query", "execute_python", "write_patch"} & names
        assert "cpp_semantic_query" in {tool.name for tool in resolve_tools(cpp=True)}
        state = {"workspace_binding": workspace.binding(), "read_only_mode": True, "task": "inspect"}
        result = graph_module.build_graph(checkpointer=InMemorySaver()).invoke(
            state, {"configurable": {"thread_id": "rag-read-only"}}
        )
        assert any(isinstance(message, ToolMessage) and "src/controller.cpp" in message.content
                   for message in result["messages"])
        prompt = build_system_prompt(state)
        assert "separately built index" in prompt and "symbol line/column" in prompt


def test_empty_workspace_index_gives_workspace_setup_hint(workspace_index, monkeypatch):
    workspace, _ = workspace_index
    monkeypatch.setattr(rag_ops, "get_code_index", lambda: CodeIndex(MemoryCollection(), workspace=workspace))
    with use_workspace(workspace):
        assert "--workspace" in rag_ops.search_code_knowledge.invoke({"query": "torque"})


def test_external_index_cli(workspace_index, monkeypatch, capsys):
    workspace, _ = workspace_index
    cli = importlib.import_module("scripts.index_codebase")
    monkeypatch.setattr(cli, "open_code_index", lambda directory, *, workspace, embedding:
                        CodeIndex(MemoryCollection(), workspace=workspace))
    monkeypatch.setattr("sys.argv", ["index_codebase", "--workspace", str(workspace.root),
                                    "--artifact-root", str(workspace.artifacts),
                                    "--root", "src", "--query", "saturation"])
    assert cli.main() == 0
    assert "src/controller.cpp" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["index_codebase", "--workspace", str(workspace.root),
                                    "--persist-dir", "chroma_db"])
    assert cli.main() == 2
    assert "cannot override" in capsys.readouterr().err


def test_real_chroma_workspace_isolation_and_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING", "hash")
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    pytest.importorskip("chromadb")
    first, second = make_repo(tmp_path, "one"), make_repo(tmp_path, "two")
    for workspace in (first, second):
        with use_workspace(workspace):
            index = open_code_index()
            assert index.sync().files == 4
            assert workspace_index_path(workspace).is_relative_to(workspace.artifacts)
            with pytest.raises(ValueError, match="artifacts"):
                open_code_index(tmp_path / "override")
    (first.root / "src/controller.cpp").write_text("// changed_first_marker\n", encoding="utf-8")
    with use_workspace(first):
        reopened = open_code_index()
        assert reopened.search("changed_first_marker", strategy="lexical")
    with use_workspace(second):
        assert open_code_index().search("changed_first_marker", strategy="lexical") == []
    reopened.collection.modify(metadata={"source_root": "wrong-repository"})
    with use_workspace(first), pytest.raises(ValueError, match="binding"):
        open_code_index()


@pytest.mark.cpp
def test_retrieval_handoff_to_real_clangd(tmp_path):
    image = os.getenv("ROBO_TEST_RUNNER_IMAGE")
    if not image and not all(shutil.which(name) for name in ("clangd", "cmake", "ninja", "g++")):
        pytest.skip("Requires native clangd/CMake/Ninja/g++ or ROBO_TEST_RUNNER_IMAGE")
    from agent.cpp import configure_build, semantic_query

    root = tmp_path / "native"
    shutil.copytree(HARNESS_ROOT / "examples/external_cpp", root)
    workspace = Workspace.create(root, tmp_path / "native-state", trusted_local=not bool(image),
                                 execution_image=image)
    with use_workspace(workspace):
        index = CodeIndex(MemoryCollection(), workspace=workspace)
        index.sync()
        result = index.search("joint_error", path_prefix="main.cpp", strategy="lexical")[0]
        lines = (workspace.root / result.path).read_text().splitlines()
        line = next(i for i, text in enumerate(lines, 1) if "joint_error(target" in text)
        column = lines[line - 1].index("joint_error") + 1
        assert configure_build(configure_only=True)["passed"]
        definition = semantic_query(result.path, "definition", line, column)
        assert "controller.cpp" in json.dumps(definition["result"])
