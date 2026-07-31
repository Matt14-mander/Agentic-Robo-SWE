"""Deterministic validators for the M1 robot-code benchmark cases."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable


def _load_workspace(path: str) -> ModuleType:
    workspace = Path(path)
    if not workspace.is_absolute():
        workspace = Path.cwd() / workspace
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


_VALIDATORS: dict[str, Callable[[ModuleType], list[str]]] = {
    "ik_reachability": _ik_reachability,
    "angle_units": _angle_units,
    "quaternion_normalization": _quaternion_normalization,
    "pid_anti_windup": _pid_anti_windup,
    "trajectory_endpoints": _trajectory_endpoints,
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
        failures = check(module)
    except Exception as exc:
        return {
            "passed": False,
            "details": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "passed": not failures,
        "details": failures if failures else ["all deterministic checks passed"],
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
