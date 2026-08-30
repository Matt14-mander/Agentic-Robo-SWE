"""Inspect Phase 5 MuJoCo closed-loop metrics without calling an LLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from agent.simulation import simulate_joint_tracking
from agent.tools._paths import resolve_within_root


def _load_controller(path: str):
    resolved = resolve_within_root(path)
    spec = importlib.util.spec_from_file_location("simulation_controller", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load controller: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JointPDController(kp=20.0, kd=5.0, torque_limit=8.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 headless MuJoCo tracking scenarios")
    parser.add_argument(
        "controller",
        nargs="?",
        default="benchmarks/fixtures/sim_joint_pd_tracking.py",
    )
    args = parser.parse_args()
    try:
        controller = _load_controller(args.controller)
        scenarios = ((1.0, 0.0), (-0.75, 0.5))
        reports = [
            simulate_joint_tracking(
                controller.compute,
                target=target,
                initial_position=initial,
            ).to_dict()
            for target, initial in scenarios
        ]
    except (ImportError, RuntimeError, ValueError, OSError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"controller": str(Path(args.controller)), "scenarios": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
