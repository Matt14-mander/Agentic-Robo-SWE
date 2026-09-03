"""Sparse structure adversarial tests and optional real CodeGen task integration."""

from __future__ import annotations

import copy
import json

import pytest

from agent.benchmark import load_cases, prepare_workspace
from agent.domain import get_domain_registry
from agent.domain_packs.robotics_autodiff_codegen.dense_validation import deterministic_samples
from agent.domain_packs.robotics_autodiff_codegen.dependencies import (
    INSTALL_PREFIX,
    inspect_dependencies,
)
from agent.domain.toolchain import configure_and_build, inspect_toolchain, run_built_executable
from agent.domain_packs.robotics_autodiff_codegen.sparse_validation import (
    MODEL_PATTERNS,
    evaluate_sparse_report,
    reconstruct_coo,
)
from benchmarks.validators import validate


def test_sparse_manifest_selects_registered_triangular_gate():
    cases = load_cases("benchmarks/sparse_codegen_cases.json")
    pack = get_domain_registry().get("robotics_autodiff_codegen")
    assert [case.id for case in cases] == [
        "sparse_unexpected_dependency", "sparse_missing_dependency",
    ]
    for case in cases:
        assert case.validator == "robotics_autodiff_codegen.sparse_output_and_jacobian"
        assert case.validator in pack.validators
        assert case.domain_pack == pack.id
        assert case.execution_environment == "local"


def sparse_observations(contract="coupled"):
    pattern = MODEL_PATTERNS[contract]
    observations = []
    for sample in deterministic_samples():
        q, v = sample["input"]
        dense = [2 * q, 0.0 if contract == "triangular" else 0.5, v, q + 2 * v]
        coo = {
            "rows": [row for row, _ in pattern],
            "columns": [column for _, column in pattern],
            "values": [dense[row * 2 + column] for row, column in pattern],
        }
        observations.append({
            "index": sample["index"], "cppad": copy.deepcopy(coo),
            "codegen": copy.deepcopy(coo), "cppad_dense": dense[:],
            "codegen_dense": dense[:], "finite_difference": dense[:],
        })
    return observations


@pytest.mark.parametrize("contract", ["coupled", "triangular"])
def test_sparse_valid_and_permuted_coo_are_equivalent(contract):
    observations = sparse_observations(contract)
    for record in observations:
        for values in record["codegen"].values():
            values.reverse()
    report = evaluate_sparse_report({"sparse_samples": observations}, contract=contract)
    assert report["passed"]
    assert report["sample_count"] == 40
    assert report["max_absolute_error"] == 0


@pytest.mark.parametrize("coo,kind", [
    ({"rows": [0], "columns": [], "values": [1]}, "length_mismatch"),
    ({"rows": [2], "columns": [0], "values": [1]}, "out_of_bounds"),
    ({"rows": [-1], "columns": [0], "values": [1]}, "out_of_bounds"),
    ({"rows": [True], "columns": [0], "values": [1]}, "invalid_index"),
    ({"rows": [0, 0], "columns": [0, 0], "values": [1, 1]}, "duplicate_coordinate"),
    ({"rows": [0], "columns": [0], "values": [float("nan")]}, "non_finite_value"),
    ({"rows": [0], "columns": [0], "values": [None]}, "non_finite_value"),
    ({"rows": [], "columns": [], "values": []}, "missing_coordinate"),
    ({"rows": [0, 0], "columns": [0, 1], "values": [1, 0]}, "unexpected_coordinate"),
])
def test_coo_rejects_malformed_structure(coo, kind):
    _, failures = reconstruct_coo(coo, shape=(2, 2), expected_pattern=((0, 0),))
    assert kind in {item["kind"] for item in failures}


def test_numeric_zero_does_not_remove_structural_dependency():
    observations = sparse_observations("triangular")
    origin = observations[3]
    assert origin["codegen"]["values"] == [0, 0, 0]
    origin["codegen"] = {"rows": [], "columns": [], "values": []}
    result = evaluate_sparse_report({"sparse_samples": observations}, contract="triangular")
    assert not result["passed"]
    assert result["structure_failures"] == 3


def test_values_detached_from_coordinates_fail_even_with_correct_dense_results():
    observations = sparse_observations("triangular")
    observations[8]["codegen"]["values"].reverse()
    result = evaluate_sparse_report({"sparse_samples": observations}, contract="triangular")
    assert not result["passed"]
    assert result["structure_failures"] == 0
    assert result["worst_failure"]["sample_index"] == 8


def test_overflowing_failure_metrics_remain_serializable():
    observations = sparse_observations()
    observations[8]["codegen"]["values"][0] = 1e308
    report = evaluate_sparse_report({"sparse_samples": observations})
    assert not report["passed"]
    assert report["max_normalized_error"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutation", ["missing", "repeated_index", "non_finite_dense"])
def test_no_silent_dense_only_fallback(mutation):
    observations = sparse_observations()
    if mutation == "missing":
        observations.pop()
    elif mutation == "repeated_index":
        observations[-1]["index"] = 0
    else:
        observations[0]["cppad_dense"][0] = float("inf")
        assert not evaluate_sparse_report({"sparse_samples": observations})["passed"]
        return
    with pytest.raises(ValueError):
        evaluate_sparse_report({"sparse_samples": observations})


@pytest.mark.cpp
def test_native_cppad_sparse_preserves_structural_zero_contract():
    if not (INSTALL_PREFIX / "include/cppad/configure.hpp").is_file():
        pytest.skip("CppAD headers not bootstrapped; independent of CodeGen runtime")
    if not inspect_toolchain().available:
        pytest.skip("C++ toolchain unavailable")
    build = configure_and_build(
        "src/agent/domain_packs/robotics_autodiff_codegen/harness", "cppad_sparse_smoke",
        cmake_definitions={"AUTODIFF_PREFIX": INSTALL_PREFIX.as_posix(), "WORKSPACE_HEADER": "unused"},
    )
    assert build.passed and build.build_dir, build
    result = run_built_executable(build.build_dir, "cppad_sparse_smoke")
    assert result["passed"], result
    assert "structural nnz=3" in result["stdout"]


@pytest.mark.cpp
@pytest.mark.autodiff_codegen
def test_sparse_model_faults_fail_and_minimal_source_repairs_pass():
    capabilities = inspect_dependencies()
    if not capabilities["available"]:
        pytest.skip(str(capabilities["missing"]))
    fixes = {
        "sparse_unexpected_dependency": (" + Scalar(0.01) * x[1]", ""),
        "sparse_missing_dependency": (", x[1] * x[1]", ", x[0] * x[1] + x[1] * x[1]"),
    }
    for case in load_cases("benchmarks/sparse_codegen_cases.json"):
        workspace = prepare_workspace(case)
        try:
            baseline = validate(case.validator, case.workspace)
            assert not baseline["passed"], baseline
            assert baseline["sparse_metrics"]["structure_failures"] > 0, baseline
            old, new = fixes[case.id]
            workspace.write_text(workspace.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
            fixed = validate(case.validator, case.workspace)
            assert fixed["passed"], fixed
            assert fixed["sparse_metrics"]["expected_nnz"] == 3
        finally:
            workspace.unlink(missing_ok=True)
