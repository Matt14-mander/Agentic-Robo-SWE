"""Phase 5 deterministic robot simulation primitives."""

from agent.simulation.diagnostics import (
    TrialDiagnosis,
    diagnose_trial,
    render_diagnostic_html,
)
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
    SafetyEvent,
    TrajectoryPerturbation,
    TrajectorySample,
    TrajectoryTrackingMetrics,
    simulate_two_joint_trajectory,
)

__all__ = [
    "JointTrackingMetrics",
    "DEFAULT_ROBUSTNESS_SEEDS",
    "RobustnessReport",
    "RobustnessScenario",
    "RobustnessTrial",
    "SafetyEvent",
    "TrialDiagnosis",
    "TrajectoryPerturbation",
    "TrajectorySample",
    "TrajectoryTrackingMetrics",
    "diagnose_trial",
    "generate_robustness_scenarios",
    "render_diagnostic_html",
    "run_robustness_suite",
    "simulate_joint_tracking",
    "simulate_two_joint_trajectory",
]
