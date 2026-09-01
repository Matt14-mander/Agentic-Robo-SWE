"""Phase 6.1 static contracts and optional real CppADCodeGen integration."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from agent.benchmark import load_cases, prepare_workspace
from agent.domain import get_domain_registry
from agent.domain_packs.robotics_autodiff_codegen.compatibility import inspect_source
from agent.domain_packs.robotics_autodiff_codegen.dependencies import inspect_dependencies
from benchmarks.validators import validate
from scripts.bootstrap_autodiff_codegen import _matching_c_compiler, bootstrap


def test_autodiff_pack_is_discoverable_and_dependency_probe_is_offline():
    pack = get_domain_registry().get("robotics_autodiff_codegen")

    assert pack.version == "0.1.0"
    assert {tool.name for tool in pack.tools} == {
        "inspect_ad_compatibility",
        "inspect_autodiff_codegen_environment",
        "validate_autodiff_codegen_model",
    }
    assert "robotics_autodiff_codegen.output_and_jacobian" in pack.validators
    first = inspect_dependencies()
    second = inspect_dependencies()
    assert first == second
    assert first["cppad_commit"] == "67bdbf1bbf89cb0c490e7bdf6eac6fe92508f072"


def test_autodiff_manifest_has_three_distinct_failure_modes():
    cases = load_cases("benchmarks/autodiff_codegen_cases.json")

    assert [case.id for case in cases] == [
        "autodiff_hardcoded_scalar",
        "autodiff_scalar_propagation",
        "autodiff_jacobian_input_order",
    ]
    assert all(case.domain_pack == "robotics_autodiff_codegen" for case in cases)
    assert all(case.validator_timeout == 300 for case in cases)


def test_ad_compatibility_checker_reports_scalar_branch_and_black_box(tmp_path):
    source = Path("benchmarks/workspaces/ad_compatibility_probe.hpp")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "double loss = x[0];\nif (x[0] > 0) loss += std::erf(x[1]);\n",
        encoding="utf-8",
    )
    try:
        result = inspect_source(source)
    finally:
        source.unlink(missing_ok=True)

    rules = {item["rule"] for item in result["findings"]}
    assert not result["passed"]
    assert {"hardcoded_scalar", "data_dependent_branch", "unsupported_black_box"} <= rules


def test_faulty_scalar_fixtures_fail_static_gate_and_order_fixture_reaches_numeric_gate():
    cases = load_cases("benchmarks/autodiff_codegen_cases.json")
    first = inspect_source(cases[0].template)
    second = inspect_source(cases[1].template)
    third = inspect_source(cases[2].template)

    assert not first["passed"]
    assert not second["passed"]
    assert third["passed"]


def test_lock_file_matches_pack_constants():
    lock = json.loads(Path(
        "src/agent/domain_packs/robotics_autodiff_codegen/dependencies.lock.json"
    ).read_text(encoding="utf-8"))

    assert lock["cppad"]["tag"] == "20240000.7"
    assert lock["cppad_codegen"]["tag"] == "v2.5.0"


def test_bootstrap_matches_sibling_c_compiler(tmp_path):
    cxx = tmp_path / "g++.exe"
    cxx.touch()
    c = tmp_path / "gcc.exe"
    c.touch()

    assert _matching_c_compiler(cxx) == str(c)


def test_bootstrap_refuses_non_linux_before_writing(monkeypatch):
    bootstrap_module = importlib.import_module("scripts.bootstrap_autodiff_codegen")
    monkeypatch.setattr(bootstrap_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        bootstrap_module.shutil,
        "rmtree",
        lambda _: pytest.fail("non-Linux bootstrap must not delete its cache"),
    )

    with pytest.raises(RuntimeError, match="only on Linux"):
        bootstrap(force=True)


@pytest.mark.autodiff_codegen
@pytest.mark.cpp
def test_all_autodiff_fixtures_fail_then_minimal_fixes_pass_real_codegen():
    dependencies = inspect_dependencies()
    if not dependencies["available"]:
        pytest.skip(f"missing autodiff capabilities: {dependencies['missing']}")
    replacements = {
        "autodiff_hardcoded_scalar": (
            "inline std::array<double, 2> robot_codegen_model(const std::array<double, 2>& x)",
            "template <class Scalar>\nstd::array<Scalar, 2> robot_codegen_model(const std::array<Scalar, 2>& x)",
        ),
        "autodiff_scalar_propagation": ("double coupling", "Scalar coupling"),
        "autodiff_jacobian_input_order": (
            "x[1] * x[1] + Scalar(0.5) * x[0], x[1] * x[0] + x[0] * x[0]",
            "x[0] * x[0] + Scalar(0.5) * x[1], x[0] * x[1] + x[1] * x[1]",
        ),
    }
    for case in load_cases("benchmarks/autodiff_codegen_cases.json"):
        workspace = prepare_workspace(case)
        try:
            baseline = validate(case.validator, case.workspace)
            assert not baseline["passed"], case.id
            old, new = replacements[case.id]
            workspace.write_text(
                workspace.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
            )
            fixed = validate(case.validator, case.workspace)
            assert fixed["passed"], (case.id, fixed)
        finally:
            workspace.unlink(missing_ok=True)
