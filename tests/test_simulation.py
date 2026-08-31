"""Phase 5 deterministic MuJoCo closed-loop benchmark tests."""

from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("mujoco")

from agent.benchmark import load_cases, prepare_workspace
from agent.simulation import (
    TrajectoryPerturbation,
    diagnose_trial,
    generate_robustness_scenarios,
    render_diagnostic_html,
    run_robustness_suite,
    simulate_joint_tracking,
    simulate_two_joint_trajectory,
)
from agent.tools._paths import PROJECT_ROOT
from benchmarks.validators import validate
from scripts.run_robustness import _write_report


pytestmark = pytest.mark.simulation


def _safe_pd(target: float, position: float, velocity: float) -> float:
    command = 20.0 * (target - position) - 5.0 * velocity
    return max(-8.0, min(8.0, command))


def _safe_trajectory_controller(
    time_seconds,
    target_positions,
    target_velocities,
    positions,
    velocities,
):
    del time_seconds
    commands = []
    for joint, limit in enumerate((10.0, 8.0)):
        command = (
            (28.0, 22.0)[joint] * (target_positions[joint] - positions[joint])
            + (7.0, 5.5)[joint] * (target_velocities[joint] - velocities[joint])
        )
        commands.append(max(-limit, min(limit, command)))
    return commands


def _unsafe_trajectory_controller(
    time_seconds,
    target_positions,
    target_velocities,
    positions,
    velocities,
):
    del time_seconds
    commands = []
    for joint in range(2):
        target_joint = 1 - joint
        commands.append(
            (28.0, 22.0)[joint]
            * (target_positions[target_joint] - positions[joint])
            + (7.0, 5.5)[joint]
            * (target_velocities[target_joint] - velocities[joint])
        )
    return commands


def test_safe_pd_tracks_positive_and_negative_targets_deterministically():
    scenarios = ((1.0, 0.0), (-0.75, 0.5))
    first = [
        simulate_joint_tracking(_safe_pd, target=target, initial_position=initial)
        for target, initial in scenarios
    ]
    second = [
        simulate_joint_tracking(_safe_pd, target=target, initial_position=initial)
        for target, initial in scenarios
    ]

    for left, right in zip(first, second, strict=True):
        assert left == right
        assert left.finite
        assert left.final_error < 0.001
        assert left.tail_rmse < 0.001
        assert abs(left.final_velocity) < 0.001
        assert left.max_abs_command <= 8.0
        assert left.settling_time is not None and left.settling_time < 1.0


def test_buggy_simulation_fixture_fails_official_validator():
    result = validate(
        "sim_joint_pd_tracking",
        "benchmarks/fixtures/sim_joint_pd_tracking.py",
    )

    assert not result["passed"]
    assert result["error"] is None
    assert any("tracking error" in detail for detail in result["details"])
    assert any("torque_limit" in detail for detail in result["details"])


def test_minimal_controller_fix_passes_official_validator():
    source = PROJECT_ROOT / "benchmarks" / "fixtures" / "sim_joint_pd_tracking.py"
    workspace = PROJECT_ROOT / "benchmarks" / "workspaces" / "_test_sim_joint_pd.py"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, workspace)
    try:
        text = workspace.read_text(encoding="utf-8")
        text = text.replace("error = position - target", "error = target - position")
        text = text.replace(
            "return self.kp * error - self.kd * velocity",
            "command = self.kp * error - self.kd * velocity\n"
            "        return max(-self.torque_limit, min(self.torque_limit, command))",
        )
        workspace.write_text(text, encoding="utf-8")
        result = validate(
            "sim_joint_pd_tracking",
            "benchmarks/workspaces/_test_sim_joint_pd.py",
        )
    finally:
        workspace.unlink(missing_ok=True)

    assert result["passed"]
    assert result["error"] is None
    assert len(result["details"]) == 2
    assert all("final_error=" in detail for detail in result["details"])
    assert all("max_command=8.0000" in detail for detail in result["details"])


def test_simulation_manifest_prepares_isolated_workspace():
    case = load_cases("benchmarks/sim_cases.json")[0]
    workspace = prepare_workspace(case)
    try:
        assert case.id == "sim_joint_pd_tracking"
        assert workspace.read_bytes() == (
            PROJECT_ROOT / "benchmarks" / "fixtures" / "sim_joint_pd_tracking.py"
        ).read_bytes()
    finally:
        workspace.unlink(missing_ok=True)


def test_simulation_rejects_invalid_duration():
    with pytest.raises(ValueError, match="must be positive"):
        simulate_joint_tracking(_safe_pd, target=1.0, duration_seconds=0.0)


def test_safe_two_joint_controller_tracks_trajectory_without_constraint_violations():
    metrics = simulate_two_joint_trajectory(
        _safe_trajectory_controller,
        initial_positions=(-1.00, 0.50),
        goal_positions=(-0.30, -1.00),
    )

    assert metrics.finite
    assert metrics.final_max_error < 0.04
    assert metrics.tracking_rmse < 0.09
    assert metrics.tail_rmse < 0.035
    assert metrics.max_abs_commands[0] <= 10.0
    assert metrics.max_abs_commands[1] <= 8.0
    assert metrics.max_joint_limit_violation == 0.0
    assert metrics.collision_steps == 0


def test_buggy_two_joint_fixture_fails_official_validator():
    result = validate(
        "sim_two_joint_trajectory",
        "benchmarks/fixtures/sim_two_joint_trajectory.py",
    )

    assert not result["passed"]
    assert result["error"] is None
    assert any("tracking" in detail.lower() for detail in result["details"])
    assert any("torque limit" in detail.lower() for detail in result["details"])


def test_minimal_two_joint_fix_passes_official_validator():
    source = PROJECT_ROOT / "benchmarks" / "fixtures" / "sim_two_joint_trajectory.py"
    workspace = PROJECT_ROOT / "benchmarks" / "workspaces" / "_test_sim_two_joint.py"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, workspace)
    try:
        text = workspace.read_text(encoding="utf-8")
        text = text.replace("target_joint = 1 - joint", "target_joint = joint")
        text = text.replace(
            "return commands[0], commands[1]",
            "return tuple(\n"
            "            max(-limit, min(limit, command))\n"
            "            for command, limit in zip(commands, self.torque_limits)\n"
            "        )",
        )
        workspace.write_text(text, encoding="utf-8")
        result = validate(
            "sim_two_joint_trajectory",
            "benchmarks/workspaces/_test_sim_two_joint.py",
        )
    finally:
        workspace.unlink(missing_ok=True)

    assert result["passed"]
    assert result["error"] is None
    assert len(result["details"]) == 2
    assert all("tracking_rmse=" in detail for detail in result["details"])
    assert all("collisions=0" in detail for detail in result["details"])


def test_simulation_manifest_contains_all_phase_five_cases():
    cases = load_cases("benchmarks/sim_cases.json")
    assert [case.id for case in cases] == [
        "sim_joint_pd_tracking",
        "sim_two_joint_trajectory",
        "sim_two_joint_robustness",
    ]


def test_two_joint_simulation_rejects_invalid_controller_shape():
    def one_command(*_args):
        return (0.0,)

    with pytest.raises(ValueError, match="exactly two"):
        simulate_two_joint_trajectory(
            one_command,
            initial_positions=(0.0, 0.0),
            goal_positions=(0.1, 0.1),
        )


def test_robustness_scenarios_are_seeded_and_bounded():
    first = generate_robustness_scenarios((11, 23, 37))
    second = generate_robustness_scenarios((11, 23, 37))

    assert first == second
    assert [scenario.seed for scenario in first] == [11, 23, 37]
    assert all(0.85 <= scenario.perturbation.mass_scale <= 1.15 for scenario in first)
    assert all(0 <= scenario.perturbation.control_delay_steps <= 3 for scenario in first)


def test_safe_controller_passes_deterministic_robustness_matrix():
    first = run_robustness_suite(lambda: _safe_trajectory_controller)
    second = run_robustness_suite(lambda: _safe_trajectory_controller)

    assert first == second
    assert first.passed
    assert first.pass_rate >= 0.8
    assert first.safety_violations == 0
    assert first.total_trials == 5
    assert first.p50_tracking_rmse <= first.p95_tracking_rmse


def test_buggy_controller_fails_robustness_validator():
    result = validate(
        "sim_two_joint_robustness",
        "benchmarks/fixtures/sim_two_joint_trajectory.py",
    )

    assert not result["passed"]
    assert result["error"] is None
    assert any("pass rate" in detail for detail in result["details"])
    assert any("hard gate" in detail for detail in result["details"])


def test_minimal_two_joint_fix_passes_robustness_validator():
    source = PROJECT_ROOT / "benchmarks" / "fixtures" / "sim_two_joint_trajectory.py"
    workspace = PROJECT_ROOT / "benchmarks" / "workspaces" / "_test_sim_robustness.py"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, workspace)
    try:
        text = workspace.read_text(encoding="utf-8")
        text = text.replace("target_joint = 1 - joint", "target_joint = joint")
        text = text.replace(
            "return commands[0], commands[1]",
            "return tuple(\n"
            "            max(-limit, min(limit, command))\n"
            "            for command, limit in zip(commands, self.torque_limits)\n"
            "        )",
        )
        workspace.write_text(text, encoding="utf-8")
        result = validate(
            "sim_two_joint_robustness",
            "benchmarks/workspaces/_test_sim_robustness.py",
        )
    finally:
        workspace.unlink(missing_ok=True)

    assert result["passed"]
    assert result["error"] is None
    assert len(result["details"]) == 6
    assert result["details"][-1].startswith("aggregate: pass_rate=100.0%")


def test_perturbation_configuration_is_validated():
    with pytest.raises(ValueError, match="actuator_strength"):
        simulate_two_joint_trajectory(
            _safe_trajectory_controller,
            initial_positions=(-1.0, 0.5),
            goal_positions=(-0.3, -1.0),
            perturbation=TrajectoryPerturbation(actuator_strength=0.0),
        )


def test_robustness_report_persists_aggregate_and_each_seed(tmp_path):
    report = run_robustness_suite(lambda: _safe_trajectory_controller, seeds=(11, 23))
    output = tmp_path / "report.json"

    _write_report(report, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["total_trials"] == 2
    assert len(payload["diagnostics"]) == 2
    assert (tmp_path / "diagnostics.html").is_file()
    assert sorted(path.name for path in (tmp_path / "seeds").iterdir()) == [
        "seed-00011.json",
        "seed-00023.json",
    ]


def test_trace_capture_is_optional_downsampled_and_deterministic():
    without_trace = simulate_two_joint_trajectory(
        _safe_trajectory_controller,
        initial_positions=(-1.0, 0.5),
        goal_positions=(-0.3, -1.0),
    )
    first = simulate_two_joint_trajectory(
        _safe_trajectory_controller,
        initial_positions=(-1.0, 0.5),
        goal_positions=(-0.3, -1.0),
        capture_trace=True,
        trace_stride=25,
    )
    second = simulate_two_joint_trajectory(
        _safe_trajectory_controller,
        initial_positions=(-1.0, 0.5),
        goal_positions=(-0.3, -1.0),
        capture_trace=True,
        trace_stride=25,
    )

    assert without_trace.trace is None
    assert first == second
    assert first.trace is not None
    assert 50 <= len(first.trace) <= 65
    assert first.trace[-1].time_seconds == pytest.approx(3.0)
    assert first.first_safety_event is None


def test_failure_diagnosis_locates_first_event_and_renders_charts():
    report = run_robustness_suite(
        lambda: _unsafe_trajectory_controller,
        seeds=(11,),
        capture_traces=True,
        trace_stride=20,
    )
    trial = report.trials[0]
    diagnosis = diagnose_trial(trial)
    rendered = render_diagnostic_html(report)

    assert not trial.passed
    assert trial.metrics.first_safety_event is not None
    assert trial.metrics.first_safety_event.kind == "torque_limit"
    assert trial.metrics.first_safety_event.time_seconds == 0.0
    assert diagnosis.category == "torque_limit"
    assert diagnosis.first_failure_time == 0.0
    assert diagnosis.peak_tracking_error is not None
    assert "Seed 11" in rendered
    assert "Raw controller commands" in rendered
    assert rendered.count("<svg") == 3


def test_trace_stride_must_be_positive():
    with pytest.raises(ValueError, match="trace_stride"):
        simulate_two_joint_trajectory(
            _safe_trajectory_controller,
            initial_positions=(-1.0, 0.5),
            goal_positions=(-0.3, -1.0),
            capture_trace=True,
            trace_stride=0,
        )
