"""Judge mutations and pinned native integration for controlled replacement."""
import copy
import importlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from agent.domain.toolchain import BuildResult
from agent.domain_packs.crocoddyl_mpc.dependencies import inspect_dependencies
from agent.domain_packs.crocoddyl_mpc.robot_report import plant_step
from agent.domain_packs.crocoddyl_mpc.spec import TRIAL_BACKENDS, trial_contract, trial_samples
from agent.domain_packs.crocoddyl_mpc.trial_report import (
    acceleration, numerical_jacobian, expected_cost, evaluate_trial_correctness, evaluate_trial_profile,
)
from agent.domain_packs.crocoddyl_mpc.validators import validate_codegen_trial
from tests.test_crocoddyl_robot_mpc import observations


def trial_observations(repeats=10):
    samples = []
    for input_ in trial_samples():
        jac = numerical_jacobian(input_)
        action = {"xnext": plant_step(input_[:4], input_[4:]),
            "Fx": [(1. if r == c else 0.) + (.02 if r < 2 and c == r+2 else 0.)
                   + (.0004 if r < 2 else .02)*jac[(r % 2)*6+c]
                   for r in range(4) for c in range(4)],
            "Fu": [(.0004 if r < 2 else .02)*jac[(r % 2)*6+c+4]
                   for r in range(4) for c in range(2)],
            **expected_cost(input_, terminal=False)}
        terminal = {"xnext": input_[:4],
                    "Fx": [1. if r == c else 0. for r in range(4) for c in range(4)],
                    "Fu": [0.] * 8, **expected_cost(input_, terminal=True)}
        obs = {"value": acceleration(input_), "jacobian": jac,
               "action": action, "terminal": terminal}
        samples.append({"input": input_, "backends": {name: copy.deepcopy(obs) for name in TRIAL_BACKENDS}})
    base = observations()
    return {"schema_version": 3, "contract": trial_contract(repeats), "samples": samples,
        "runs": {name: [copy.deepcopy(base["runs"]["shifted_warm"][i % 10])
                        for i in range(repeats)] for name in TRIAL_BACKENDS},
        "generation_compile_load_seconds": 2.,
        "calc_diff_timings_ns": {name: [2000. if name != "codegen" else 1000.] * repeats for name in TRIAL_BACKENDS},
        "mpc_timings_ns": {name: [20000. if name != "codegen" else 15000.] * repeats for name in TRIAL_BACKENDS}}


def test_three_way_correctness_and_compile_amortization():
    raw = trial_observations()
    assert evaluate_trial_correctness(raw)["passed"]
    profile = evaluate_trial_profile(raw, correctness_passed=True)
    assert profile["recommendation"] == "candidate_for_repeated_process_trial"
    assert profile["codegen_end_to_end_speedup"] == pytest.approx(4/3)
    assert profile["break_even_32_step_rollouts"] == 400000
    assert profile["break_even_control_steps"] == 12800000


@pytest.mark.parametrize("fault", ["value", "jacobian", "hessian", "terminal", "plant", "cost", "convergence"])
def test_every_backend_and_repetition_is_gated(fault):
    raw = trial_observations()
    obs = raw["samples"][13]["backends"]["cppad"]
    if fault == "value":
        obs["value"][1] += .1
    elif fault == "jacobian":
        obs["jacobian"][0] += .1
    elif fault == "hessian":
        obs["action"]["Lxx"][0] += .1
    elif fault == "terminal":
        obs["terminal"]["cost"] += .1
    elif fault == "plant":
        raw["runs"]["codegen"][9]["states"][20][0] += .1
    elif fault == "cost":
        raw["runs"]["codegen"][9]["costs"][20] = 10
    else:
        raw["runs"]["codegen"][9]["all_solved"] = False
    gate = evaluate_trial_correctness(raw)
    assert not gate["passed"]
    assert evaluate_trial_profile(raw, correctness_passed=gate["passed"])["recommendation"] == "blocked_by_correctness"


@pytest.mark.parametrize("quantity", ["jacobian", "Lxx", "cost"])
def test_shared_wrong_baseline_cannot_make_all_backends_pass(quantity):
    raw = trial_observations()
    for obs in raw["samples"][7]["backends"].values():
        if quantity == "jacobian":
            obs["jacobian"][0] += .1
        elif quantity == "cost":
            obs["action"]["cost"] += .1
        else:
            obs["action"]["Lxx"][0] += .1
    assert not evaluate_trial_correctness(raw)["passed"]


@pytest.mark.parametrize("fault", ["contract", "sample", "order", "nan", "shape", "backend", "repeats"])
def test_missing_or_nonfinite_evidence_rejected(fault):
    raw = trial_observations()
    if fault == "contract":
        raw["contract"]["input_order"].reverse()
    elif fault == "sample":
        raw["samples"].pop()
    elif fault == "order":
        raw["samples"][0]["input"].reverse()
    elif fault == "nan":
        raw["samples"][0]["backends"]["codegen"]["jacobian"][0] = float("nan")
    elif fault == "shape":
        raw["samples"][0]["backends"]["codegen"]["action"]["Fx"].pop()
    elif fault == "backend":
        raw["samples"][0]["backends"].pop("native")
    else:
        raw["runs"]["cppad"].pop()
    with pytest.raises(ValueError):
        evaluate_trial_correctness(raw)


def test_no_benefit_noise_and_unpaired_wins_do_not_authorize_adoption():
    raw = trial_observations()
    raw["mpc_timings_ns"]["codegen"] = [25000.] * 10
    profile = evaluate_trial_profile(raw, correctness_passed=True)
    assert profile["recommendation"] == "keep_native_no_measured_end_to_end_benefit"
    assert profile["break_even_32_step_rollouts"] is None
    raw["mpc_timings_ns"]["codegen"][0] = 1e8
    assert evaluate_trial_profile(raw, correctness_passed=True)["recommendation"] == "inconclusive_noisy_profile"
    raw = trial_observations()
    raw["mpc_timings_ns"]["codegen"] = [15000.] * 6 + [21000.] * 4
    assert evaluate_trial_profile(raw, correctness_passed=True)["recommendation"] == "keep_native_no_measured_end_to_end_benefit"
    raw["calc_diff_timings_ns"]["native"].pop()
    with pytest.raises(ValueError):
        evaluate_trial_profile(raw, correctness_passed=True)


@pytest.mark.parametrize("artifact_fault", [None, "library", "sources"])
@pytest.mark.parametrize("repeats", [10, 50])
def test_trial_validator_uses_unique_library_and_serial_build(monkeypatch, tmp_path, artifact_fault, repeats):
    module = importlib.import_module("agent.domain_packs.crocoddyl_mpc.validators")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    monkeypatch.setattr(module, "inspect_dependencies", lambda: {"available": True})
    def build(source, target, **options):
        assert target == "crocoddyl_codegen_trial"
        assert options["build_jobs"] == 1
        assert options["cmake_definitions"]["AUTODIFF_PREFIX"]
        return BuildResult(True, "source", "build", target, "test", False, 0, 0, "", "")
    monkeypatch.setattr(module, "configure_and_build", build)
    def run(build_dir, target, *, args, timeout):
        assert json.loads(Path(args[1]).read_text()) == trial_contract(repeats)
        assert args[3] == str(repeats)
        assert timeout == 120
        Path(args[0]).write_text(json.dumps(trial_observations(repeats)), encoding="utf-8")
        if artifact_fault != "library":
            Path(args[2] + ".so").write_bytes(b"test-only-generated-library")
        sources = Path(args[2] + "_sources")
        sources.mkdir()
        if artifact_fault != "sources":
            (sources / "model.c").write_text("/* test-only-generated-source */", encoding="utf-8")
        return {"passed": True}
    monkeypatch.setattr(module, "run_built_executable", run)
    first, second = (module.validate_crocoddyl_mpc(codegen=True, repeats=repeats) for _ in range(2))
    if artifact_fault:
        assert not first["passed"] and first["stage"] == "report"
        return
    assert first["passed"] and second["passed"]
    assert first["artifacts"]["directory"] != second["artifacts"]["directory"]
    saved = json.loads(Path(first["artifacts"]["validation"]).read_text())
    assert len(saved["generated_library"]["sha256"]) == 64
    assert len(saved["generated_sources"][0]["sha256"]) == 64


@pytest.mark.parametrize("repeats", [10, 11, 50, 100, 200])
def test_configurable_repeats_preserve_correctness_and_eighty_percent_gate(repeats):
    raw = trial_observations(repeats)
    assert evaluate_trial_correctness(raw, repeats=repeats)["passed"]
    minimum = (4 * repeats + 4) // 5
    raw["mpc_timings_ns"]["codegen"] = [15000.] * (minimum - 1) + [20000.] * (repeats - minimum + 1)
    profile = evaluate_trial_profile(raw, correctness_passed=True, repeats=repeats)
    assert profile["gates"]["minimum_paired_wins"]["minimum"] == minimum
    assert profile["rejection_reasons"] == ["minimum_paired_wins"]
    raw["mpc_timings_ns"]["codegen"][minimum - 1] = 15000.
    assert not evaluate_trial_profile(raw, correctness_passed=True, repeats=repeats)["rejection_reasons"]


@pytest.mark.parametrize("repeats", [9, 201, True, 10.5])
def test_invalid_repeat_budget_rejected(repeats):
    with pytest.raises(ValueError):
        trial_contract(repeats)


def test_tail_regression_is_explained_without_relaxing_gate():
    raw = trial_observations()
    raw["mpc_timings_ns"]["native"] = [10_385_982.] * 10
    raw["mpc_timings_ns"]["codegen"] = [7_832_892.] * 9 + [10_409_256.]
    profile = evaluate_trial_profile(raw, correctness_passed=True)
    assert profile["rejection_reasons"] == ["p95_non_regression"]
    assert profile["gates"]["p95_non_regression"]["delta_ns"] == 23_274
    assert profile["p95_order_statistic"] == 10
    assert profile["tail_sample_warning"]
    assert profile["default_backend"] == "native"
    assert profile["production_adoption"] is False


def test_requested_repeat_budget_cannot_be_changed_by_raw_report():
    raw = trial_observations(50)
    with pytest.raises(ValueError, match="contract"):
        evaluate_trial_correctness(raw)
    with pytest.raises(ValueError, match="contract"):
        evaluate_trial_profile(raw, correctness_passed=True)
    raw["runs"]["codegen"].pop()
    with pytest.raises(ValueError, match="repetitions"):
        evaluate_trial_correctness(raw, repeats=50)


@pytest.mark.cpp
@pytest.mark.crocoddyl
def test_native_controlled_codegen_action_replacement():
    if not inspect_dependencies()["available"]:
        pytest.skip("pinned Crocoddyl/Pinocchio/CppADCodeGen Linux stack unavailable")
    result = validate_codegen_trial()
    assert result["passed"], result
    assert result["correctness"]["sample_count"] == 25
    # Correctness is mandatory; speedup is an observation, never a required CI claim.
    assert result["profile"]["codegen_end_to_end_speedup"] > 0


@pytest.mark.parametrize("arguments", [
    ["--repeats", "50"], ["--robot", "--repeats", "10"],
    ["--codegen", "--repeats", "9"], ["--codegen", "--repeats", "201"],
    ["--codegen", "--processes", "2"],
])
def test_cli_rejects_invalid_measurement_configuration(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["run_crocoddyl_mpc.py", *arguments])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_crocoddyl_mpc.py"),
                       run_name="__main__")
    assert exit_.value.code == 2


def test_cli_forwards_measurement_budget(monkeypatch, capsys):
    module = importlib.import_module("agent.domain_packs.crocoddyl_mpc.process_trial")
    def trial(**options):
        assert options["processes"] == 3
        assert options["repeats"] == 50
        return {"passed": True, "profile": {"recommendation": "keep_native_inconsistent_process_benefit"}}
    monkeypatch.setattr(module, "validate_process_trial", trial)
    monkeypatch.setattr(sys, "argv", ["run_crocoddyl_mpc.py", "--codegen", "--processes", "3", "--repeats", "50"])
    with pytest.raises(SystemExit) as exit_:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_crocoddyl_mpc.py"),
                       run_name="__main__")
    assert exit_.value.code == 0  # Negative performance evidence is not a correctness failure.
    assert json.loads(capsys.readouterr().out)["passed"]
