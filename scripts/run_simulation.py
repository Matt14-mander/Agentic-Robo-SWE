"""Inspect Phase 5 MuJoCo closed-loop metrics without calling an LLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from agent.simulation import simulate_joint_tracking, simulate_two_joint_trajectory
from agent.tools._paths import resolve_within_root


def _load_module(path: str):
    resolved = resolve_within_root(path)
    spec = importlib.util.spec_from_file_location("simulation_controller", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load controller: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _single_joint_reports(module):
    controller = module.JointPDController(kp=20.0, kd=5.0, torque_limit=8.0)
    scenarios = ((1.0, 0.0), (-0.75, 0.5))
    return [
        simulate_joint_tracking(
            controller.compute,
            target=target,
            initial_position=initial,
        ).to_dict()
        for target, initial in scenarios
    ]


def _two_joint_reports(module):
    controller = module.TwoJointTrajectoryController(
        kp=(28.0, 22.0),
        kd=(7.0, 5.5),
        torque_limits=(10.0, 8.0),
    )
    scenarios = (
        ((-1.00, 0.50), (-0.30, -1.00)),
        ((0.55, -0.80), (0.10, 0.90)),
    )
    return [
        simulate_two_joint_trajectory(
            controller.compute,
            initial_positions=initial,
            goal_positions=goal,
        ).to_dict()
        for initial, goal in scenarios
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 headless MuJoCo tracking scenarios")
    parser.add_argument(
        "--case",
        choices=("single-joint", "two-joint"),
        default="single-joint",
        help="simulation scenario family",
    )
    parser.add_argument(
        "controller",
        nargs="?",
        help="controller file; defaults to the selected case's buggy fixture",
    )
    args = parser.parse_args()
    default_controllers = {
        "single-joint": "benchmarks/fixtures/sim_joint_pd_tracking.py",
        "two-joint": "benchmarks/fixtures/sim_two_joint_trajectory.py",
    }
    controller_path = args.controller or default_controllers[args.case]
    try:
        module = _load_module(controller_path)
        reports = (
            _single_joint_reports(module)
            if args.case == "single-joint"
            else _two_joint_reports(module)
        )
    except (ImportError, RuntimeError, ValueError, OSError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"case": args.case, "controller": str(Path(controller_path)), "scenarios": reports},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
