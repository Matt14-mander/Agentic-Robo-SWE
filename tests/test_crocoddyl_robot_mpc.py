"""Independent robot MPC acceptance gates and native integration."""

import copy
import importlib
import json
import math
from pathlib import Path

import pytest

from agent.domain.toolchain import BuildResult
from agent.domain_packs.crocoddyl_mpc.dependencies import inspect_dependencies
from agent.domain_packs.crocoddyl_mpc.robot_report import (
    evaluate_robot_correctness, evaluate_robot_profile, plant_step,
)
from agent.domain_packs.crocoddyl_mpc.spec import (
    MPC_REPEATS, MPC_STEPS, MPC_TIMING_KEYS, robot_contract,
)
from agent.domain_packs.crocoddyl_mpc.validators import validate_robot_mpc


def observations():
    """Synthetic PD trace checks the judge; it does not establish native MPC success."""
    states = [robot_contract()["initial_state"]]
    controls = []
    for _ in range(MPC_STEPS):
        q1, q2, v1, v2 = states[-1]
        a1, a2 = -100*q1-20*v1, -100*q2-20*v2
        m22 = .131
        m12 = m22 + .28 * math.cos(q2)
        m11 = 1.332 + .56 * math.cos(q2)
        g2 = -2.7468 * math.cos(q1+q2)
        g1 = -13.734 * math.cos(q1) + g2
        controls.append([
            m11*a1 + m12*a2 - .28*math.sin(q2)*(2*v1*v2+v2*v2) + g1,
            m12*a1 + m22*a2 + .28*math.sin(q2)*v1*v1 + g2,
        ])
        states.append(plant_step(states[-1], controls[-1]))
    run = {"states": states, "controls": controls, "costs": [1.] * MPC_STEPS,
           "initial_costs": [2.] * MPC_STEPS, "rollout_errors": [0.] * MPC_STEPS,
           "solve_ns": [1000.] * MPC_STEPS, "iterations": [2] * MPC_STEPS,
           "all_solved": True}
    return {"schema_version": 2, "contract": robot_contract(),
            "derivative_max_abs_error": 1e-7,
            "runs": {name: [copy.deepcopy(run) for _ in range(MPC_REPEATS)]
                     for name in ("cold", "shifted_warm")},
            "stage_timings_ns": {name: [1000.] * MPC_REPEATS for name in MPC_TIMING_KEYS},
            "mpc_timings_ns": {"cold": [20000.] * MPC_REPEATS,
                               "shifted_warm": [15000.] * MPC_REPEATS}}


def test_independent_physics_and_mpc_profile():
    raw = observations()
    assert evaluate_robot_correctness(raw)["passed"]
    report = evaluate_robot_profile(raw, correctness_passed=True)
    assert report["warm_start_recommendation"] == "use_shifted_warm_start"
    assert report["codegen_end_to_end_speedup"] is None
    # At zero state, the gravity compensation must be a fixed point of the plant.
    assert plant_step([0.] * 4, [-16.4808, -2.7468]) == pytest.approx([0.] * 4, abs=1e-14)


@pytest.mark.parametrize("fault", ["plant", "cost", "limits", "convergence", "derivative"])
def test_bad_evidence_in_any_repetition_blocks_performance(fault):
    raw = observations()
    run = raw["runs"]["shifted_warm"][3]
    if fault == "plant":
        run["states"][12][0] += .1
    elif fault == "cost":
        run["costs"][7] = 10
    elif fault == "limits":
        run["controls"][10][0] = 101
    elif fault == "convergence":
        run["all_solved"] = False
    else:
        raw["derivative_max_abs_error"] = .1
    result = evaluate_robot_correctness(raw)
    assert not result["passed"]
    assert evaluate_robot_profile(raw, correctness_passed=result["passed"])["recommendation"] == (
        "blocked_by_correctness"
    )


@pytest.mark.parametrize("fault", ["contract", "repeats", "states", "nan", "iterations"])
def test_malformed_robot_evidence_is_rejected(fault):
    raw = observations()
    if fault == "contract":
        raw["contract"]["dt"] = .1
    elif fault == "repeats":
        raw["runs"]["cold"].pop()
    elif fault == "states":
        raw["runs"]["cold"][0]["states"].pop()
    elif fault == "nan":
        raw["runs"]["cold"][0]["states"][1][0] = float("nan")
    else:
        raw["runs"]["cold"][0]["iterations"][0] = True
    with pytest.raises(ValueError):
        evaluate_robot_correctness(raw)


def test_noisy_or_incomplete_latency_does_not_establish_benefit():
    raw = observations()
    raw["mpc_timings_ns"]["shifted_warm"][0] = 1000000.
    assert evaluate_robot_profile(raw, correctness_passed=True)["recommendation"] == (
        "inconclusive_noisy_profile"
    )
    raw["stage_timings_ns"]["calc"].pop()
    with pytest.raises(ValueError):
        evaluate_robot_profile(raw, correctness_passed=True)


def test_robot_validator_contract_artifacts_and_progress(monkeypatch, tmp_path):
    module = importlib.import_module("agent.domain_packs.crocoddyl_mpc.validators")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "relpath_for_display", str)
    monkeypatch.setattr(module, "inspect_dependencies", lambda: {"available": True})
    def build(source, target, **options):
        assert target == "crocoddyl_robot_mpc"
        assert options["cmake_definitions"]["ROBOT_URDF"].endswith("two_link.urdf")
        return BuildResult(True, "source", "build", target, "test", False, 0, 0, "", "")
    monkeypatch.setattr(module, "configure_and_build", build)
    def run(build_dir, target, *, args, timeout):
        assert json.loads(Path(args[1]).read_text()) == robot_contract()
        Path(args[0]).write_text(json.dumps(observations()), encoding="utf-8")
        return {"passed": True}
    monkeypatch.setattr(module, "run_built_executable", run)
    messages = []
    result = module.validate_crocoddyl_mpc(robot=True, progress=messages.append)
    assert result["passed"]
    assert len(messages) == 3
    assert len(list(tmp_path.glob("*/profile.json"))) == 1


@pytest.mark.cpp
@pytest.mark.crocoddyl
def test_real_pinocchio_crocoddyl_mpc():
    if not inspect_dependencies()["available"]:
        pytest.skip("pinned Crocoddyl/Pinocchio Linux stack unavailable")
    result = validate_robot_mpc()
    assert result["passed"], result
    assert result["correctness"]["max_plant_abs_error"] < 1e-8
