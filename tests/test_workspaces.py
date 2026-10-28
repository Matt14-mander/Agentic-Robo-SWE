from __future__ import annotations

import importlib
import json
import os

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.persistence import open_persistent_graph
from agent.tools import execute_python, grep_codebase, list_dir, read_file_chunk, resolve_tools, write_patch
from agent.workspace import Workspace, check_binding, current_workspace, exported_source, use_workspace


def make_workspace(tmp_path, name="repo", **kwargs):
    root = tmp_path / name
    root.mkdir()
    (root / "main.cpp").write_text("int value = 1;\n", encoding="utf-8")
    return Workspace.create(root, tmp_path / (name + "-state"), **kwargs)


def test_workspaces_do_not_share_paths_or_mutations(tmp_path):
    first, second = make_workspace(tmp_path), make_workspace(tmp_path, "second")
    for workspace in (first, second):
        with use_workspace(workspace):
            assert "value = 1" in read_file_chunk.invoke({"path": "main.cpp"})
            assert "main.cpp:1:" in grep_codebase.invoke({"pattern": "value", "glob": "**/*.cpp"})
            assert "main.cpp" in list_dir.invoke({})
    with use_workspace(first):
        assert write_patch.invoke({"path": "main.cpp", "old_string": "1", "new_string": "2"}).startswith("OK")
        assert read_file_chunk.invoke({"path": str(second.root / "main.cpp")}).startswith("ERROR")
    assert "value = 1" in (second.root / "main.cpp").read_text()
    assert current_workspace() is None


def test_private_paths_and_untrusted_execution_rejected(tmp_path):
    workspace = make_workspace(tmp_path)
    (workspace.root / ".env").write_text("private-token")
    (workspace.root / ".env-private").mkdir()
    (workspace.root / ".env-private" / "token").write_text("private-token")
    with use_workspace(workspace):
        assert read_file_chunk.invoke({"path": ".env"}).startswith("ERROR")
        assert read_file_chunk.invoke({"path": "../repo-state/checkpoints.sqlite"}).startswith("ERROR")
        assert execute_python.invoke({"code": "print(1)"}).startswith("ERROR")
        assert grep_codebase.invoke({"pattern": "private-token", "glob": "**/*"}).startswith("<no matches")
    with exported_source(workspace.root) as exported:
        assert not (exported / ".env").exists()
        assert not (exported / ".env-private").exists()
        assert (exported / "main.cpp").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable permission contract")
def test_snapshot_preserves_executable_semantics_and_hash(tmp_path):
    from agent.workspace import source_digest
    workspace = make_workspace(tmp_path)
    script = workspace.root / "scripts" / "check.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    with exported_source(workspace.root) as exported:
        assert source_digest(exported) == source_digest(workspace.root)
        assert (exported / "scripts" / "check.sh").stat().st_mode & 0o111


def test_link_escape_and_overlapping_artifacts_rejected(tmp_path):
    workspace = make_workspace(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        Workspace.create(workspace.root, workspace.root / "state")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    try:
        (workspace.root / "alias").symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit creating symlinks")
    with pytest.raises(ValueError):
        workspace.resolve("alias")
    with pytest.raises(ValueError, match="regular"):
        with exported_source(workspace.root):
            pass


def test_checkpoint_is_outside_target_and_binding_is_checked(tmp_path):
    first, second = make_workspace(tmp_path), make_workspace(tmp_path, "second")
    config = {"configurable": {"thread_id": "binding-test"}}
    with use_workspace(first):
        with open_persistent_graph() as default_graph:
            assert default_graph is not None
        assert (first.artifacts / "checkpoints.sqlite").exists()
        with open_persistent_graph("session.sqlite") as graph:
            graph.update_state(config, {"task": "inspect", "workspace_binding": first.binding()})
        with open_persistent_graph("session.sqlite") as graph:
            values = graph.get_state(config).values
            check_binding(values)
        with pytest.raises(ValueError, match="artifact"):
            with open_persistent_graph(first.root / "bad.sqlite"):
                pass
    with use_workspace(second), pytest.raises(ValueError, match="binding"):
        check_binding(values)
    assert (first.artifacts / "session.sqlite").exists()
    assert not list(first.root.glob("*.sqlite"))


def test_graph_tool_threads_keep_workspace_context(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    module = importlib.import_module("agent.graph")

    def fake_planner(state):
        if state.get("loop_step", 0):
            return {"messages": [AIMessage(content="done")], "suggestion": "done", "loop_step": 2}
        return {"messages": [AIMessage(content="", tool_calls=[{
            "name": "write_patch", "args": {"path": "main.cpp", "old_string": "1", "new_string": "3"},
            "id": "change", "type": "tool_call"}])], "loop_step": 1}

    monkeypatch.setattr(module, "planner", fake_planner)
    with use_workspace(workspace):
        graph = module.build_graph(checkpointer=InMemorySaver())
        graph.invoke({"task": "change", "workspace_binding": workspace.binding()},
                     {"configurable": {"thread_id": "tool-thread"}})
    assert "value = 3" in (workspace.root / "main.cpp").read_text()


def test_dispatch_rejects_tools_not_in_session_schema(tmp_path):
    from agent.graph import execute_tools
    workspace = make_workspace(tmp_path)
    with use_workspace(workspace):
        assert "cpp_build" in {t.name for t in resolve_tools(cpp=True)}
        assert "cpp_build" not in {t.name for t in resolve_tools(mode="read_only", cpp=True)}
        state = {"workspace_binding": workspace.binding(), "read_only_mode": True,
                 "messages": [AIMessage(content="", tool_calls=[{
                     "name": "execute_python", "args": {"code": "print('bad')"},
                     "id": "unauthorized", "type": "tool_call"}])]}
        result = execute_tools(state, {})
        assert "capabilities" in result["messages"][0].content


def test_trusted_local_exec_uses_target_cwd_without_credentials(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, trusted_local=True)
    monkeypatch.setenv("ROBO_SECRET_SENTINEL", "must-not-inherit")
    with use_workspace(workspace):
        output = execute_python.invoke({"code": (
            "import os,json; from pathlib import Path; "
            "print(json.dumps([Path('main.cpp').exists(), os.getenv('ROBO_SECRET_SENTINEL')]))")})
    assert json.dumps([True, None]) in output
