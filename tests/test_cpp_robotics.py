"""Controlled professional-pattern C++ tasks: provenance, mutation gates and real builds."""
import importlib
import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from agent.benchmark import load_cases
from agent.tools._paths import PROJECT_ROOT, relpath_for_display
from agent.domain.registry import get_domain_registry
from agent.domain.toolchain import BuildResult, inspect_toolchain
from agent.domain_packs.cpp_robotics.spec import CASE_NAMES, expected, samples
from agent.domain_packs.cpp_robotics.validators import PACK_ROOT, judge_observations, validate_robotics_cpp


def output_text(scenario):
    values = [expected(scenario, x) for x in samples(scenario)]
    return "\n".join(str(len(row)) + " " + " ".join(map(repr, row)) for row in values)


def test_manifest_and_registered_judges_are_complete_and_public():
    cases = load_cases("benchmarks/cpp_robotics_cases.json")
    assert len(cases) == 8
    assert len({case.category for case in cases}) == 8
    assert all(case.suite == "m2_cpp_robotics_v1" for case in cases)
    assert all(case.provenance == "synthetic_controlled" for case in cases)
    pack = get_domain_registry().get("cpp_robotics")
    assert set(pack.validators) == {case.validator for case in cases}
    assert pack.manifest.local_only
    assert not pack.tools  # Uses the same official validation route, not extra exploratory tools.


def test_professional_suite_composes_existing_and_new_cases_without_duplication():
    cases = load_cases("benchmarks/cpp_professional_cases.json")
    assert len(cases) == len({case.id for case in cases}) == 17
    assert all(case.provenance == "synthetic_controlled" for case in cases)
    assert {case.domain_pack for case in cases} == {
        "cpp_robotics", "cpp_reference", "robotics_autodiff_codegen", "pinocchio_rnea"}


@pytest.mark.parametrize("fault", ["cycle", "duplicate", "outside", "type"])
def test_manifest_includes_reject_cycles_duplicate_ids_and_external_paths(fault):
    path = PROJECT_ROOT / ".agent_state/test-manifests" / (uuid.uuid4().hex + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    includes = {"cycle": [relpath_for_display(path)],
                "duplicate": ["benchmarks/cpp_cases.json", "benchmarks/cpp_cases.json"],
                "outside": [str(PROJECT_ROOT.parent / "outside.json")], "type": "invalid"}[fault]
    path.write_text(json.dumps({"schema_version": 2, "includes": includes}), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            load_cases(path)
    finally:
        path.unlink()


def test_invalid_baseline_is_not_counted_as_success_after_later_pass(monkeypatch):
    import agent.benchmark as module
    case = replace(load_cases()[0], workspace="benchmarks/workspaces/_invalid_baseline_test.py")
    results = iter([module.ValidationResult(False, [], 0., "environment failed"),
                    module.ValidationResult(True, [], 0.)])
    monkeypatch.setattr(module, "validate_case", lambda *_: next(results))
    result = module.run_case(case, invoke=lambda *_args, **_kwargs: {})
    assert not result.success
    assert result.baseline_error == "environment failed"
    assert result.attempts == 0
    monkeypatch.setattr(module, "validate_case", lambda *_: module.ValidationResult(False, [], 0., "environment failed"))
    report, _ = module.run_suite([case], invoke=lambda *_args, **_kwargs: {},
                                output_dir=f".agent_state/test-reports/{uuid.uuid4().hex}")
    assert report["invalid_baselines"] == 1
    assert report["passed"] == 0
    assert report["grouped"]["category"][case.category]["runs_with_errors"] == 1
    assert report["official_validator_compliance_rate"] == 0.


@pytest.mark.parametrize("scenario", CASE_NAMES)
def test_deterministic_samples_and_independent_numeric_judge(scenario):
    assert samples(scenario) == samples(scenario)
    assert len(samples(scenario)) == 28
    gate = judge_observations(scenario, output_text(scenario))
    assert gate["passed"] and gate["sample_count"] == 28
    assert gate["max_abs_error"] == 0.


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "nan", "infinity", "value"])
def test_malformed_or_incorrect_native_evidence_cannot_pass(fault):
    rows = output_text("scalar_adl").splitlines()
    if fault == "missing":
        rows.pop()
    elif fault == "extra":
        rows.append(rows[0])
    elif fault == "shape":
        rows[0] = "1 0"
    else:
        fields = rows[0].split()
        fields[2] = {"nan": "nan", "infinity": "inf", "value": "999"}[fault]
        rows[0] = " ".join(fields)
    if fault in {"missing", "extra", "shape"}:
        with pytest.raises(ValueError):
            judge_observations("scalar_adl", "\n".join(rows))
    else:
        gate = judge_observations("scalar_adl", "\n".join(rows))
        assert not gate["passed"]
        json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("fault", [None, "build", "runtime", "output", "value"])
def test_validator_preserves_source_identity_and_diagnostics(monkeypatch, tmp_path, fault):
    module = importlib.import_module("agent.domain_packs.cpp_robotics.validators")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    class Toolchain:
        available = True
        def to_dict(self):
            return {"fingerprint": "test"}
    monkeypatch.setattr(module, "inspect_toolchain", Toolchain)
    def build(source, target, **options):
        assert options["cmake_definitions"]["ROBOTICS_CASE"] == "0"
        assert len(options["cmake_definitions"]["WORKSPACE_SHA256"]) == 64
        return BuildResult(fault != "build", source, "test-build", target, "test", False,
                           0., 0., "", "", "failed" if fault == "build" else None)
    def run(*args, **options):
        if fault != "output":
            text = output_text("scalar_adl") if fault != "value" else "3 999 999 999\n" * 28
            Path(options["args"][1]).write_text(text, encoding="utf-8")
        return {"passed": fault != "runtime", "stderr": "test failure"}
    monkeypatch.setattr(module, "configure_and_build", build)
    monkeypatch.setattr(module, "run_built_executable", run)
    result = validate_robotics_cpp("benchmarks/fixtures/cpp_robotics/scalar_adl.hpp", scenario="scalar_adl")
    assert result["passed"] is (fault is None)
    if fault == "value":
        assert result["error"] is None  # A genuine correctness failure, not an environment failure.
    elif fault:
        assert result["error"]
    evidence = json.loads(Path(result["artifacts"]["validation"]).read_text(encoding="utf-8"))
    assert len(evidence["source_sha256"]) == 64
    assert len(evidence["oracle_sha256"]) == 64


@pytest.mark.cpp
@pytest.mark.parametrize("scenario", CASE_NAMES)
def test_native_broken_fixture_fails_and_reference_passes(scenario):
    if not inspect_toolchain().available:
        pytest.skip("CMake/C++17 toolchain missing")
    broken = validate_robotics_cpp(f"benchmarks/fixtures/cpp_robotics/{scenario}.hpp", scenario=scenario)
    repaired = validate_robotics_cpp(str(PACK_ROOT / "harness/reference.hpp"), scenario=scenario)
    assert broken["error"] is None, broken
    assert broken["passed"] is False, broken
    assert repaired["error"] is None, repaired
    assert repaired["passed"] is True, repaired
