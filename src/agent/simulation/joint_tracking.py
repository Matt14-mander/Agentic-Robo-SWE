"""Headless MuJoCo closed-loop benchmark for a single revolute joint."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable


_MODEL_XML = """
<mujoco model="single_joint_tracking">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <body name="link" pos="0 0 0">
      <joint name="joint" type="hinge" axis="0 0 1" damping="0.15"
             limited="true" range="-2 2"/>
      <geom name="link_geom" type="capsule" fromto="0 0 0 0.5 0 0"
            size="0.035" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="joint" gear="1" ctrllimited="true" ctrlrange="-8 8"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class JointTrackingMetrics:
    target: float
    initial_position: float
    duration_seconds: float
    final_position: float
    final_velocity: float
    final_error: float
    tail_rmse: float
    max_abs_position: float
    max_abs_command: float
    settling_time: float | None
    finite: bool

    def to_dict(self) -> dict[str, float | bool | None]:
        return asdict(self)


def simulate_joint_tracking(
    controller: Callable[[float, float, float], float],
    *,
    target: float,
    initial_position: float = 0.0,
    duration_seconds: float = 2.5,
    timestep: float = 0.002,
) -> JointTrackingMetrics:
    """Run a deterministic closed loop and return controller-independent metrics."""
    if duration_seconds <= 0 or timestep <= 0:
        raise ValueError("duration_seconds and timestep must be positive")
    try:
        import mujoco  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed. Run: uv sync --extra simulation"
        ) from exc

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    data.qpos[0] = initial_position
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    steps = max(1, round(duration_seconds / timestep))
    positions: list[float] = []
    errors: list[float] = []
    commands: list[float] = []
    finite = True
    for _ in range(steps):
        position = float(data.qpos[0])
        velocity = float(data.qvel[0])
        command = float(controller(target, position, velocity))
        if not all(math.isfinite(value) for value in (position, velocity, command)):
            finite = False
            break
        data.ctrl[0] = command
        mujoco.mj_step(model, data)
        positions.append(float(data.qpos[0]))
        errors.append(target - positions[-1])
        commands.append(command)

    if not positions:
        return JointTrackingMetrics(
            target=target,
            initial_position=initial_position,
            duration_seconds=duration_seconds,
            final_position=float("nan"),
            final_velocity=float("nan"),
            final_error=float("inf"),
            tail_rmse=float("inf"),
            max_abs_position=float("inf"),
            max_abs_command=float("inf"),
            settling_time=None,
            finite=False,
        )

    tail = errors[-max(1, len(errors) // 4):]
    settling_index = _settling_index(errors, tolerance=0.05)
    return JointTrackingMetrics(
        target=target,
        initial_position=initial_position,
        duration_seconds=duration_seconds,
        final_position=positions[-1],
        final_velocity=float(data.qvel[0]),
        final_error=abs(errors[-1]),
        tail_rmse=math.sqrt(sum(error * error for error in tail) / len(tail)),
        max_abs_position=max(abs(position) for position in positions),
        max_abs_command=max(abs(command) for command in commands),
        settling_time=settling_index * timestep if settling_index is not None else None,
        finite=finite and all(math.isfinite(value) for value in positions + commands),
    )


def _settling_index(errors: list[float], *, tolerance: float) -> int | None:
    """Return the first sample after which every remaining error stays bounded."""
    suffix_max = 0.0
    result: int | None = None
    for index in range(len(errors) - 1, -1, -1):
        suffix_max = max(suffix_max, abs(errors[index]))
        if suffix_max <= tolerance:
            result = index
    return result
