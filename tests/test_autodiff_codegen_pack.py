"""Phase 6.1 static contracts and optional real CppADCodeGen integration."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent.benchmark import load_cases, prepare_workspace
from agent.domain import get_domain_registry
from agent.domain.toolchain import BuildResult
from agent.domain_packs.robotics_autodiff_codegen.compatibility import inspect_source
from agent.domain_packs.robotics_autodiff_codegen.dense_validation import (
    DEFAULT_DENSE_SPEC,
    deterministic_samples,
    load_dense_report,
)
from agent.domain_packs.robotics_autodiff_codegen.dependencies import inspect_dependencies
from benchmarks.validators import validate
from scripts.bootstrap_autodiff_codegen import _matching_c_compiler, bootstrap


def test_autodiff_pack_is_discoverable_and_dependency_probe_is_offline():
    pack = get_domain_registry().get("robotics_autodiff_codegen")

    assert pack.version == "0.2.0"
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


def test_autodiff_manifest_has_dense_correctness_failure_modes():
    cases = load_cases("benchmarks/autodiff_codegen_cases.json")

    assert [case.id for case in cases] == [
        "autodiff_hardcoded_scalar",
        "autodiff_scalar_propagation",
        "autodiff_jacobian_input_order",
        "autodiff_random_blind_spot",
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


def test_dense_sample_contract_is_deterministic_and_includes_boundaries():
    first = deterministic_samples()
    second = deterministic_samples()

    assert first == second
    assert DEFAULT_DENSE_SPEC.sample_count == 40
    assert [sample["kind"] for sample in first].count("regression") == 3
    assert [sample["kind"] for sample in first].count("boundary") == 5
    assert [sample["kind"] for sample in first].count("random") == 32
    assert first[8]["input"] == pytest.approx([1.5518862895933725, 0.5394656006764178])


def test_random_blind_spot_is_invisible_to_legacy_points_but_not_dense_samples():
    def ghost(x: float) -> float:
        return 1e-4 * (x - 0.25) * (x - 1.2) * (x + 0.9) * x * (x - 2.0) * (x + 2.0)

    samples = deterministic_samples()
    assert all(ghost(sample["input"][0]) == 0.0 for sample in samples[:8])
    assert any(abs(ghost(sample["input"][0])) > 1e-8 for sample in samples[8:])


def test_dense_report_rejects_changed_seed_or_samples(tmp_path):
    report = {
        "schema_version": 1,
        "passed": True,
        "seed": DEFAULT_DENSE_SPEC.seed,
        "random_samples": DEFAULT_DENSE_SPEC.random_samples,
        "sample_count": DEFAULT_DENSE_SPEC.sample_count,
        "thresholds": {
            "output_atol": DEFAULT_DENSE_SPEC.output_atol,
            "output_rtol": DEFAULT_DENSE_SPEC.output_rtol,
            "jacobian_atol": DEFAULT_DENSE_SPEC.jacobian_atol,
            "jacobian_rtol": DEFAULT_DENSE_SPEC.jacobian_rtol,
        },
        "metrics": {"failures": 0, "non_finite": 0, "dimension_failures": 0},
        "samples": deterministic_samples(),
    }
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_dense_report(path)["passed"] is True

    report["samples"][8]["input"][0] = 999.0
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="official generator"):
        load_dense_report(path)


def test_static_failure_also_writes_reproducible_diagnostic_artifact():
    case = load_cases("benchmarks/autodiff_codegen_cases.json")[0]
    workspace = prepare_workspace(case)
    try:
        result = validate(case.validator, case.workspace)
    finally:
        workspace.unlink(missing_ok=True)

    assert not result["passed"]
    assert result["stage"] == "compatibility"
    validation_path = Path(result["artifacts"]["validation"])
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["validation_spec"]["sample_count"] == 40
    assert payload["source_sha256"]


@pytest.mark.parametrize("passed", [True, False])
def test_dense_validator_persists_success_and_element_level_failure(
    monkeypatch, passed, tmp_path
):
    validator_module = importlib.import_module(
        "agent.domain_packs.robotics_autodiff_codegen.validators"
    )
    case = load_cases("benchmarks/autodiff_codegen_cases.json")[2]
    workspace = prepare_workspace(case)
    build_dir = Path(".agent_state/domain_artifacts/cmake/dense-unit-test")
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = tmp_path / ("pass" if passed else "fail")
    monkeypatch.setattr(validator_module, "relpath_for_display", lambda path: str(path))
    worst: dict[str, Any] | None = None
    if not passed:
        worst = {
            "quantity": "jacobian",
            "comparison": "codegen_vs_finite_difference",
            "sample_index": 8,
            "sample_kind": "random",
            "input": deterministic_samples()[8]["input"],
            "row": 1,
            "column": 0,
            "expected": 0.5,
            "actual": 0.0,
            "absolute_error": 0.5,
            "normalized_error": 500000.0,
        }
    report = {
        "schema_version": 1,
        "passed": passed,
        "seed": DEFAULT_DENSE_SPEC.seed,
        "random_samples": DEFAULT_DENSE_SPEC.random_samples,
        "sample_count": DEFAULT_DENSE_SPEC.sample_count,
        "thresholds": {
            "output_atol": DEFAULT_DENSE_SPEC.output_atol,
            "output_rtol": DEFAULT_DENSE_SPEC.output_rtol,
            "jacobian_atol": DEFAULT_DENSE_SPEC.jacobian_atol,
            "jacobian_rtol": DEFAULT_DENSE_SPEC.jacobian_rtol,
        },
        "metrics": {
            "failures": 0 if passed else 1,
            "non_finite": 0,
            "dimension_failures": 0,
            "max_output_absolute_error": 0.0,
            "max_jacobian_absolute_error": 0.0 if passed else 0.5,
            "max_jacobian_frobenius_relative_error": 0.0 if passed else 0.5,
        },
        "worst_failure": worst,
        "samples": deterministic_samples(),
    }
    monkeypatch.setattr(validator_module, "_artifact_directory", lambda *_: artifact_dir)
    monkeypatch.setattr(validator_module, "inspect_dependencies", lambda: {
        "available": True,
        "missing": [],
        "fingerprint": "dependency-test",
        "bootstrap_command": "unused",
    })
    monkeypatch.setattr(validator_module, "configure_and_build", lambda *_, **__: BuildResult(
        passed=True,
        source_dir="harness",
        build_dir=str(build_dir),
        target="autodiff_codegen_validator",
        fingerprint="build-test",
        cache_hit=False,
        configure_seconds=0.1,
        build_seconds=0.1,
        stdout="",
        stderr="",
    ))

    def fake_run(*_, **__):
        assert not (build_dir / "dense-validation.json").exists()
        (build_dir / "dense-validation.json").write_text(json.dumps(report), encoding="utf-8")
        return {
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "stdout": json.dumps(report),
            "stderr": "",
            "duration_seconds": 0.1,
            "executable": "fake",
        }

    monkeypatch.setattr(validator_module, "run_built_executable", fake_run)
    try:
        result = validator_module.validate_autodiff_codegen(case.workspace)
    finally:
        workspace.unlink(missing_ok=True)

    assert result["passed"] is passed
    assert result["stage"] == ("complete" if passed else "dense_correctness")
    assert (artifact_dir / "validation.json").is_file()
    assert (artifact_dir / "dense-validation.json").is_file()
    assert (artifact_dir / "samples.json").is_file()
    assert (artifact_dir / "failing-input.json").is_file() is (not passed)
    if not passed:
        assert any("coupled_velocity_term/joint_position" in item for item in result["details"])


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
        "autodiff_random_blind_spot": (
            "+ ghost, x[0] * x[1]",
            ", x[0] * x[1]",
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
