"""Deterministic validators for the M1 robot-code benchmark cases."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

ValidatorOutput = list[str] | tuple[list[str], list[str]]


def _load_workspace(path: str) -> ModuleType:
    workspace = Path(path)
    if not workspace.is_absolute():
        # Validators are also called from execute_python, whose cwd is an isolated
        # temporary directory. Resolve benchmark paths from the uploaded/project
        # repository root instead of depending on the caller's cwd.
        workspace = Path(__file__).resolve().parents[1] / workspace
    if not workspace.is_file():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    module_name = f"benchmark_workspace_{workspace.stem}"
    spec = importlib.util.spec_from_file_location(module_name, workspace)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workspace: {workspace}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ik_reachability(module: ModuleType) -> list[str]:
    failures: list[str] = []
    inverse = module.inverse_kinematics_2link

    try:
        inverse(3.0, 0.0)
        failures.append("unreachable target did not raise ValueError")
    except ValueError as exc:
        text = str(exc).lower()
        if not any(word in text for word in ("reach", "range", "可达", "范围", "超出")):
            failures.append("unreachable target error is not descriptive")
    except Exception as exc:
        failures.append(f"unreachable target raised {type(exc).__name__}, expected ValueError")

    for x, y in ((1.0, 1.0), (2.0, 0.0)):
        try:
            theta1, theta2 = inverse(x, y)
            rebuilt_x = math.cos(theta1) + math.cos(theta1 + theta2)
            rebuilt_y = math.sin(theta1) + math.sin(theta1 + theta2)
            if not (math.isclose(rebuilt_x, x, abs_tol=1e-7)
                    and math.isclose(rebuilt_y, y, abs_tol=1e-7)):
                failures.append(f"forward reconstruction failed for ({x}, {y})")
        except Exception as exc:
            failures.append(f"reachable target ({x}, {y}) raised {type(exc).__name__}: {exc}")
    return failures


def _angle_units(module: ModuleType) -> list[str]:
    failures: list[str] = []
    convert = module.degrees_to_radians
    for degrees, expected in ((0.0, 0.0), (90.0, math.pi / 2), (180.0, math.pi), (-45.0, -math.pi / 4)):
        actual = convert(degrees)
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            failures.append(f"{degrees} degrees produced {actual}, expected {expected}")
    return failures


def _quaternion_normalization(module: ModuleType) -> list[str]:
    failures: list[str] = []
    normalize = module.normalize_quaternion
    for quaternion in ((1.0, 2.0, 3.0, 4.0), (-2.0, 0.0, 0.0, 0.0)):
        normalized = normalize(quaternion)
        norm = math.sqrt(sum(component * component for component in normalized))
        if not math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            failures.append(f"normalized quaternion has norm {norm}, expected 1.0")
    try:
        normalize((0.0, 0.0, 0.0, 0.0))
        failures.append("zero quaternion did not raise ValueError")
    except ValueError as exc:
        if not str(exc).strip():
            failures.append("zero quaternion ValueError has no explanation")
    except Exception as exc:
        failures.append(f"zero quaternion raised {type(exc).__name__}, expected ValueError")
    return failures


def _pid_anti_windup(module: ModuleType) -> list[str]:
    failures: list[str] = []
    controller = module.PIDController(kp=1.0, ki=1.0, output_limit=1.0)
    saturated_outputs = [controller.update(error=10.0, dt=0.1) for _ in range(20)]
    if any(abs(output) > 1.0 + 1e-12 for output in saturated_outputs):
        failures.append("controller output exceeds configured limit")
    reverse_output = controller.update(error=-0.5, dt=1.0)
    if reverse_output >= 0.0:
        failures.append(
            f"output stayed non-negative ({reverse_output}) after error reversal; integral windup remains"
        )
    return failures


def _trajectory_endpoints(module: ModuleType) -> list[str]:
    failures: list[str] = []
    trajectory = module.linear_trajectory
    points = trajectory(2.0, 10.0, 5)
    if len(points) != 5:
        failures.append(f"returned {len(points)} samples, expected 5")
    if not points or not math.isclose(points[0], 2.0, abs_tol=1e-12):
        failures.append("trajectory does not include the start point")
    if not points or not math.isclose(points[-1], 10.0, abs_tol=1e-12):
        failures.append("trajectory does not include the end point")
    if len(points) == 5:
        expected = [2.0, 4.0, 6.0, 8.0, 10.0]
        if any(not math.isclose(actual, target, abs_tol=1e-12)
               for actual, target in zip(points, expected, strict=True)):
            failures.append("trajectory samples are not uniformly spaced")
    try:
        trajectory(0.0, 1.0, 1)
        failures.append("samples < 2 did not raise ValueError")
    except ValueError:
        pass
    except Exception as exc:
        failures.append(f"samples < 2 raised {type(exc).__name__}, expected ValueError")
    return failures


def _sim_joint_pd_tracking(module: ModuleType) -> ValidatorOutput:
    from agent.simulation import simulate_joint_tracking

    failures: list[str] = []
    diagnostics: list[str] = []
    controller = module.JointPDController(kp=20.0, kd=5.0, torque_limit=8.0)
    scenarios = ((1.0, 0.0), (-0.75, 0.5))
    for target, initial_position in scenarios:
        metrics = simulate_joint_tracking(
            controller.compute,
            target=target,
            initial_position=initial_position,
        )
        label = f"target={target}, initial={initial_position}"
        diagnostics.append(
            f"{label}: final_error={metrics.final_error:.6f}, "
            f"tail_rmse={metrics.tail_rmse:.6f}, "
            f"settling_time={metrics.settling_time}, "
            f"max_command={metrics.max_abs_command:.4f}"
        )
        if not metrics.finite:
            failures.append(f"{label}: simulation produced non-finite state or command")
            continue
        if metrics.max_abs_command > 8.0 + 1e-9:
            failures.append(
                f"{label}: controller command {metrics.max_abs_command:.4f} exceeds torque_limit=8"
            )
        if metrics.final_error > 0.05:
            failures.append(f"{label}: final tracking error {metrics.final_error:.4f} exceeds 0.05")
        if metrics.tail_rmse > 0.06:
            failures.append(f"{label}: tail RMSE {metrics.tail_rmse:.4f} exceeds 0.06")
        if abs(metrics.final_velocity) > 0.08:
            failures.append(
                f"{label}: final velocity {metrics.final_velocity:.4f} exceeds 0.08 rad/s"
            )
        if metrics.max_abs_position > 1.6:
            failures.append(
                f"{label}: joint excursion {metrics.max_abs_position:.4f} exceeds safety bound 1.6"
            )
        if metrics.settling_time is None or metrics.settling_time > 2.0:
            failures.append(f"{label}: controller did not settle within 2.0 seconds")
    return failures, diagnostics


def _sim_two_joint_trajectory(module: ModuleType) -> ValidatorOutput:
    from agent.simulation import simulate_two_joint_trajectory

    failures: list[str] = []
    diagnostics: list[str] = []
    torque_limits = (10.0, 8.0)
    controller = module.TwoJointTrajectoryController(
        kp=(28.0, 22.0),
        kd=(7.0, 5.5),
        torque_limits=torque_limits,
    )
    scenarios = (
        ((-1.00, 0.50), (-0.30, -1.00)),
        ((0.55, -0.80), (0.10, 0.90)),
    )
    for initial_positions, goal_positions in scenarios:
        metrics = simulate_two_joint_trajectory(
            controller.compute,
            initial_positions=initial_positions,
            goal_positions=goal_positions,
        )
        label = f"initial={initial_positions}, goal={goal_positions}"
        diagnostics.append(
            f"{label}: final_error={metrics.final_max_error:.6f}, "
            f"tracking_rmse={metrics.tracking_rmse:.6f}, "
            f"tail_rmse={metrics.tail_rmse:.6f}, "
            f"max_commands={metrics.max_abs_commands}, "
            f"collisions={metrics.collision_steps}"
        )
        if not metrics.finite:
            failures.append(f"{label}: simulation produced non-finite state or command")
            continue
        for joint, (actual, limit) in enumerate(
            zip(metrics.max_abs_commands, torque_limits, strict=True)
        ):
            if actual > limit + 1e-9:
                failures.append(
                    f"{label}: joint {joint} command {actual:.4f} exceeds torque limit {limit}"
                )
        if metrics.final_max_error > 0.04:
            failures.append(
                f"{label}: final maximum tracking error "
                f"{metrics.final_max_error:.4f} exceeds 0.04"
            )
        if metrics.tracking_rmse > 0.09:
            failures.append(
                f"{label}: trajectory RMSE {metrics.tracking_rmse:.4f} exceeds 0.09"
            )
        if metrics.tail_rmse > 0.035:
            failures.append(f"{label}: tail RMSE {metrics.tail_rmse:.4f} exceeds 0.035")
        if max(abs(value) for value in metrics.final_velocities) > 0.06:
            failures.append(
                f"{label}: final joint velocity exceeds 0.06 rad/s: "
                f"{metrics.final_velocities}"
            )
        if metrics.max_joint_limit_violation > 1e-9:
            failures.append(
                f"{label}: joint limit violation {metrics.max_joint_limit_violation:.6f} rad"
            )
        if metrics.collision_steps:
            failures.append(
                f"{label}: keep-out obstacle contacted for {metrics.collision_steps} steps"
            )
    return failures, diagnostics


_VALIDATORS: dict[str, Callable[[ModuleType], ValidatorOutput]] = {
    "ik_reachability": _ik_reachability,
    "angle_units": _angle_units,
    "quaternion_normalization": _quaternion_normalization,
    "pid_anti_windup": _pid_anti_windup,
    "trajectory_endpoints": _trajectory_endpoints,
    "sim_joint_pd_tracking": _sim_joint_pd_tracking,
    "sim_two_joint_trajectory": _sim_two_joint_trajectory,
}


def validate(validator: str, workspace: str) -> dict[str, object]:
    """Validate one workspace and return a JSON-serializable result."""
    try:
        check = _VALIDATORS[validator]
    except KeyError:
        return {
            "passed": False,
            "details": [],
            "error": f"Unknown validator: {validator}",
        }
    try:
        module = _load_workspace(workspace)
        outcome = check(module)
        if isinstance(outcome, tuple):
            failures, success_details = outcome
        else:
            failures, success_details = outcome, ["all deterministic checks passed"]
    except Exception as exc:
        return {
            "passed": False,
            "details": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "passed": not failures,
        "details": failures if failures else success_details,
        "error": None,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(json.dumps({"passed": False, "details": [], "error": "usage: VALIDATOR WORKSPACE"}))
        return 2
    result = validate(args[0], args[1])
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
