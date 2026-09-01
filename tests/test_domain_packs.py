"""Phase 6.0 Domain Pack contracts, selection and reference C++ integration."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent.benchmark import domain_case_metadata, load_cases, prepare_workspace
from agent.domain import DomainManifest, DomainPack, DomainRegistry, get_domain_registry
from agent.domain.loader import load_domain_manifest
from agent.domain.selection import append_domain_tools, domain_prompt_fragment
from agent.domain.toolchain import configure_and_build, inspect_toolchain
from agent.tools import FOCUSED_BENCHMARK_TOOLS, resolve_tools
from agent.domain_packs.cpp_reference.tools import inspect_cpp_toolchain
from benchmarks.validators import validate
from scripts.inspect_domain_packs import _payload


@tool
def _example_tool(value: int) -> int:
    """Return the supplied test value."""
    return value


def _pack(pack_id: str = "test_pack", tools=()) -> DomainPack:
    return DomainPack(
        manifest=DomainManifest(
            schema_version=1,
            id=pack_id,
            version="0.1.0",
            description="test pack",
        ),
        tools=tuple(tools),
        validators={f"{pack_id}.check": lambda workspace: {
            "passed": True,
            "details": [workspace],
            "error": None,
        }},
        prompt_fragment="Use the test workflow.",
    )


def test_manifest_loader_is_strict(tmp_path):
    manifest = tmp_path / "pack.toml"
    manifest.write_text(
        'schema_version = 1\nid = "demo_pack"\nversion = "1.2.3"\n'
        'description = "demo"\nrequired_executables = ["cmake"]\n',
        encoding="utf-8",
    )

    loaded = load_domain_manifest(manifest)

    assert loaded.id == "demo_pack"
    assert loaded.required_executables == ("cmake",)


def test_registry_rejects_duplicate_pack_tool_and_non_namespaced_validator():
    registry = DomainRegistry()
    registry.register(_pack("first", (_example_tool,)))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_pack("first"))
    with pytest.raises(ValueError, match="tool name conflict"):
        registry.register(_pack("second", (_example_tool,)))

    invalid = replace(_pack("third"), validators={"not_namespaced": lambda _: {}})
    with pytest.raises(ValueError, match="must start"):
        DomainRegistry().register(invalid)


def test_builtin_cpp_reference_pack_is_discoverable():
    pack = get_domain_registry().get("cpp_reference")

    assert pack.version == "0.1.0"
    assert {tool.name for tool in pack.tools} == {
        "inspect_cpp_toolchain",
        "build_cpp_project",
        "run_cpp_target",
    }
    assert "cpp_reference.output_equivalence" in pack.validators
    assert "successful compile is intermediate evidence" in pack.prompt_fragment
    listed = _payload()["packs"]
    assert any(item["id"] == "cpp_reference" for item in listed)


def test_domain_tools_and_prompt_are_selected_only_when_pack_is_active():
    plain = resolve_tools(mode="focused")
    domain = resolve_tools(mode="focused", domain_packs=("cpp_reference",))

    assert [tool.name for tool in plain] == [tool.name for tool in FOCUSED_BENCHMARK_TOOLS]
    assert [tool.name for tool in domain[-3:]] == [
        "inspect_cpp_toolchain",
        "build_cpp_project",
        "run_cpp_target",
    ]
    assert domain_prompt_fragment(()) == ""
    assert "Cpp Reference Pack" in domain_prompt_fragment(("cpp_reference",))


def test_domain_tool_cannot_shadow_a_core_tool(monkeypatch):
    conflicting = _pack("conflict", (FOCUSED_BENCHMARK_TOOLS[0],))
    selection_module = importlib.import_module("agent.domain.selection")
    monkeypatch.setattr(selection_module, "active_domain_packs", lambda _: (conflicting,))
    with pytest.raises(ValueError, match="conflicts with core"):
        append_domain_tools(FOCUSED_BENCHMARK_TOOLS, ("conflict",))


def test_focused_planner_binds_pack_tools_and_prompt(monkeypatch):
    planner_module = importlib.import_module("agent.nodes.planner")
    captured: dict[str, object] = {}

    class FakeModel:
        def bind_tools(self, tools):
            captured["tools"] = [tool.name for tool in tools]
            return self

        def invoke(self, messages):
            captured["system"] = str(messages[0].content)
            return AIMessage(content="done")

    monkeypatch.setattr(planner_module, "get_chat_model", lambda temperature=0.0: FakeModel())
    planner_module.planner({
        "task": "fix cpp source",
        "benchmark_mode": True,
        "benchmark_strategy": "focused",
        "domain_packs": ("cpp_reference",),
    })

    assert "build_cpp_project" in captured["tools"]
    assert "Cpp Reference Pack" in captured["system"]


def test_cpp_benchmark_manifest_v2_loads_domain_metadata():
    case = load_cases("benchmarks/cpp_cases.json")[0]

    assert case.domain_pack == "cpp_reference"
    assert case.required_capabilities == ("cmake", "cxx_compiler")
    assert case.validator_timeout == 180
    assert case.execution_environment == "local"
    version, capabilities, fingerprint = domain_case_metadata(case)
    assert version == "0.1.0"
    assert set(capabilities) == {"cmake", "cxx_compiler", "execution_environment"}
    assert fingerprint


def test_toolchain_probe_is_deterministic_and_build_paths_are_restricted():
    assert inspect_toolchain() == inspect_toolchain()
    with pytest.raises(ValueError, match="Invalid CMake target"):
        configure_and_build(
            "src/agent/domain_packs/cpp_reference/harness",
            "bad target",
        )
    with pytest.raises(ValueError, match="outside project root"):
        configure_and_build("../../outside", "target")


def test_cpp_pack_tools_refuse_non_local_executor(monkeypatch):
    monkeypatch.setenv("EXECUTOR_BACKEND", "e2b")

    output = inspect_cpp_toolchain.invoke({})

    assert "local-only" in output
    assert "e2b" in output


@pytest.mark.cpp
def test_cpp_reference_fixture_fails_then_minimal_fix_passes():
    toolchain = inspect_toolchain()
    if not toolchain.available:
        pytest.skip(f"missing C++ toolchain: {toolchain.missing}")
    case = load_cases("benchmarks/cpp_cases.json")[0]
    workspace = prepare_workspace(case)
    try:
        baseline = validate(case.validator, case.workspace)
        assert not baseline["passed"]
        assert baseline["error"] is None

        text = workspace.read_text(encoding="utf-8")
        workspace.write_text(
            text.replace("state[1] + control", "state[1] - control"),
            encoding="utf-8",
        )
        fixed = validate(case.validator, case.workspace)
        assert fixed["passed"]
        assert fixed["error"] is None
        cached = validate(case.validator, case.workspace)
        assert cached["passed"]
        assert "build_cache_hit=True" in cached["details"][0]
    finally:
        workspace.unlink(missing_ok=True)
