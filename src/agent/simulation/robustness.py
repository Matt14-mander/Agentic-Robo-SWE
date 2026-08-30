"""Seeded perturbation matrix and aggregate scoring for Phase 5.2."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable

from agent.simulation.trajectory_tracking import (
    JointVector,
    TrajectoryController,
    TrajectoryPerturbation,
    TrajectoryTrackingMetrics,
    simulate_two_joint_trajectory,
)


DEFAULT_ROBUSTNESS_SEEDS = (11, 23, 37, 53, 71)


@dataclass(frozen=True)
class RobustnessScenario:
    seed: int
    initial_positions: JointVector
    goal_positions: JointVector
    perturbation: TrajectoryPerturbation

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessTrial:
    scenario: RobustnessScenario
    metrics: TrajectoryTrackingMetrics
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RobustnessReport:
    trials: tuple[RobustnessTrial, ...]
    passed_trials: int
    total_trials: int
    pass_rate: float
    p50_tracking_rmse: float
    p95_tracking_rmse: float
    worst_final_error: float
    worst_tail_rmse: float
    total_collision_steps: int
    safety_violations: int
    robustness_score: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **asdict(self)}


def generate_robustness_scenarios(
    seeds: tuple[int, ...] = DEFAULT_ROBUSTNESS_SEEDS,
) -> tuple[RobustnessScenario, ...]:
    """Generate safe nominal paths with bounded, reproducible perturbations."""
    if not seeds:
        raise ValueError("at least one robustness seed is required")
    scenarios: list[RobustnessScenario] = []
    for seed in seeds:
        rng = random.Random(seed)
        initial = (
            rng.uniform(-1.10, -0.78),
            rng.uniform(0.28, 0.62),
        )
        goal = (
            rng.uniform(-0.46, -0.22),
            rng.uniform(-1.10, -0.76),
        )
        disturbance_sign = -1.0 if rng.random() < 0.5 else 1.0
        perturbation = TrajectoryPerturbation(
            seed=seed,
            mass_scale=rng.uniform(0.85, 1.15),
            damping_scale=rng.uniform(0.75, 1.25),
            actuator_strength=rng.uniform(0.88, 1.0),
            sensor_noise_std=rng.uniform(0.0005, 0.003),
            control_delay_steps=rng.randint(0, 3),
            disturbance_torque=(
                disturbance_sign * rng.uniform(1.0, 2.0),
                -disturbance_sign * rng.uniform(0.6, 1.2),
            ),
            disturbance_start=rng.uniform(0.8, 1.4),
            disturbance_duration=rng.uniform(0.05, 0.10),
        )
        scenarios.append(
            RobustnessScenario(
                seed=seed,
                initial_positions=initial,
                goal_positions=goal,
                perturbation=perturbation,
            )
        )
    return tuple(scenarios)


def run_robustness_suite(
    controller_factory: Callable[[], TrajectoryController],
    *,
    seeds: tuple[int, ...] = DEFAULT_ROBUSTNESS_SEEDS,
) -> RobustnessReport:
    """Evaluate a fresh controller instance for every deterministic scenario."""
    trials: list[RobustnessTrial] = []
    for scenario in generate_robustness_scenarios(seeds):
        metrics = simulate_two_joint_trajectory(
            controller_factory(),
            initial_positions=scenario.initial_positions,
            goal_positions=scenario.goal_positions,
            perturbation=scenario.perturbation,
        )
        failures = _trial_failures(metrics)
        trials.append(
            RobustnessTrial(
                scenario=scenario,
                metrics=metrics,
                passed=not failures,
                failures=tuple(failures),
            )
        )
    return _aggregate(tuple(trials))


def _trial_failures(metrics: TrajectoryTrackingMetrics) -> list[str]:
    failures: list[str] = []
    if not metrics.finite:
        failures.append("non-finite simulation state or command")
        return failures
    if metrics.max_abs_commands[0] > 10.0 + 1e-9:
        failures.append("shoulder command exceeds 10 N*m")
    if metrics.max_abs_commands[1] > 8.0 + 1e-9:
        failures.append("elbow command exceeds 8 N*m")
    if metrics.final_max_error > 0.055:
        failures.append(f"final error {metrics.final_max_error:.4f} exceeds 0.055")
    if metrics.tracking_rmse > 0.12:
        failures.append(f"tracking RMSE {metrics.tracking_rmse:.4f} exceeds 0.12")
    if metrics.tail_rmse > 0.045:
        failures.append(f"tail RMSE {metrics.tail_rmse:.4f} exceeds 0.045")
    if max(abs(value) for value in metrics.final_velocities) > 0.08:
        failures.append(f"final velocity {metrics.final_velocities} exceeds 0.08 rad/s")
    if metrics.max_joint_limit_violation > 1e-9:
        failures.append(f"joint limit violation {metrics.max_joint_limit_violation:.6f} rad")
    if metrics.collision_steps:
        failures.append(f"keep-out collision for {metrics.collision_steps} steps")
    return failures


def _aggregate(trials: tuple[RobustnessTrial, ...]) -> RobustnessReport:
    rmses = sorted(trial.metrics.tracking_rmse for trial in trials)
    tails = [trial.metrics.tail_rmse for trial in trials]
    finals = [trial.metrics.final_max_error for trial in trials]
    passed_trials = sum(trial.passed for trial in trials)
    safety_violations = sum(
        bool(
            trial.metrics.collision_steps
            or trial.metrics.max_joint_limit_violation > 1e-9
            or trial.metrics.max_abs_commands[0] > 10.0 + 1e-9
            or trial.metrics.max_abs_commands[1] > 8.0 + 1e-9
            or not trial.metrics.finite
        )
        for trial in trials
    )
    pass_rate = passed_trials / len(trials)
    p50 = _percentile(rmses, 0.50)
    p95 = _percentile(rmses, 0.95)
    tracking_score = max(0.0, 1.0 - p95 / 0.12)
    stability_score = max(0.0, 1.0 - max(tails) / 0.045)
    safety_score = 1.0 if safety_violations == 0 else 0.0
    score = 100.0 * (
        0.40 * pass_rate
        + 0.25 * tracking_score
        + 0.15 * stability_score
        + 0.20 * safety_score
    )
    return RobustnessReport(
        trials=trials,
        passed_trials=passed_trials,
        total_trials=len(trials),
        pass_rate=pass_rate,
        p50_tracking_rmse=p50,
        p95_tracking_rmse=p95,
        worst_final_error=max(finals),
        worst_tail_rmse=max(tails),
        total_collision_steps=sum(trial.metrics.collision_steps for trial in trials),
        safety_violations=safety_violations,
        robustness_score=score,
        passed=pass_rate >= 0.8 and safety_violations == 0,
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction
