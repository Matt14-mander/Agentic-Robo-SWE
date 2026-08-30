"""Phase 5 deterministic robot simulation primitives."""

from agent.simulation.joint_tracking import JointTrackingMetrics, simulate_joint_tracking
from agent.simulation.robustness import (
    DEFAULT_ROBUSTNESS_SEEDS,
    RobustnessReport,
    RobustnessScenario,
    RobustnessTrial,
    generate_robustness_scenarios,
    run_robustness_suite,
)
from agent.simulation.trajectory_tracking import (
    TrajectoryPerturbation,
    TrajectoryTrackingMetrics,
    simulate_two_joint_trajectory,
)

__all__ = [
    "JointTrackingMetrics",
    "DEFAULT_ROBUSTNESS_SEEDS",
    "RobustnessReport",
    "RobustnessScenario",
    "RobustnessTrial",
    "TrajectoryPerturbation",
    "TrajectoryTrackingMetrics",
    "generate_robustness_scenarios",
    "run_robustness_suite",
    "simulate_joint_tracking",
    "simulate_two_joint_trajectory",
]
