"""Deterministic two-joint MuJoCo trajectory tracking benchmark."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence


JointVector = tuple[float, float]
TrajectoryController = Callable[
    [float, JointVector, JointVector, JointVector, JointVector],
    Sequence[float],
]

_MODEL_XML = """
<mujoco model="two_joint_trajectory">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <geom name="keepout" type="sphere" pos="0.30 0.52 0" size="0.08"
          rgba="0.9 0.2 0.2 1" contype="1" conaffinity="1"/>
    <body name="link1" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.18"
             limited="true" range="-1.5 1.5"/>
      <geom name="link1_geom" type="capsule" fromto="0 0 0 0.55 0 0"
            size="0.035" mass="1.2" contype="1" conaffinity="1"/>
      <body name="link2" pos="0.55 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" damping="0.12"
               limited="true" range="-1.7 1.7"/>
        <geom name="link2_geom" type="capsule" fromto="0 0 0 0.45 0 0"
              size="0.03" mass="0.8" contype="1" conaffinity="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1"
           ctrllimited="true" ctrlrange="-10 10"/>
    <motor name="elbow_motor" joint="elbow" gear="1"
           ctrllimited="true" ctrlrange="-8 8"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class TrajectoryTrackingMetrics:
    initial_positions: JointVector
    goal_positions: JointVector
    duration_seconds: float
    final_positions: JointVector
    final_velocities: JointVector
    final_max_error: float
    tracking_rmse: float
    tail_rmse: float
    max_abs_commands: JointVector
    max_joint_limit_violation: float
    collision_steps: int
    finite: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def simulate_two_joint_trajectory(
    controller: TrajectoryController,
    *,
    initial_positions: JointVector,
    goal_positions: JointVector,
    move_seconds: float = 2.0,
    duration_seconds: float = 3.0,
    timestep: float = 0.002,
) -> TrajectoryTrackingMetrics:
    """Track a smooth point-to-point trajectory and report safety metrics."""
    if move_seconds <= 0 or duration_seconds <= 0 or timestep <= 0:
        raise ValueError("move_seconds, duration_seconds and timestep must be positive")
    if move_seconds > duration_seconds:
        raise ValueError("move_seconds must not exceed duration_seconds")
    if any(abs(value) > limit for value, limit in zip(initial_positions, (1.5, 1.7), strict=True)):
        raise ValueError("initial_positions exceed model joint limits")
    try:
        import mujoco  # type: ignore[import-not-found,import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed. Run: uv sync --extra simulation"
        ) from exc

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    data.qpos[:] = initial_positions
    data.qvel[:] = (0.0, 0.0)
    mujoco.mj_forward(model, data)

    keepout_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "keepout")
    steps = max(1, round(duration_seconds / timestep))
    squared_errors: list[float] = []
    commands: list[JointVector] = []
    positions: list[JointVector] = []
    collision_steps = 0
    finite = True

    for step in range(steps):
        elapsed = step * timestep
        target_positions, target_velocities = _smooth_target(
            elapsed,
            initial_positions,
            goal_positions,
            move_seconds,
        )
        current_positions = (float(data.qpos[0]), float(data.qpos[1]))
        current_velocities = (float(data.qvel[0]), float(data.qvel[1]))
        raw = controller(
            elapsed,
            target_positions,
            target_velocities,
            current_positions,
            current_velocities,
        )
        if len(raw) != 2:
            raise ValueError("controller must return exactly two joint commands")
        command = (float(raw[0]), float(raw[1]))
        values = current_positions + current_velocities + command
        if not all(math.isfinite(value) for value in values):
            finite = False
            break
        data.ctrl[:] = command
        mujoco.mj_step(model, data)

        position = (float(data.qpos[0]), float(data.qpos[1]))
        positions.append(position)
        commands.append(command)
        squared_errors.append(
            sum((target - actual) ** 2 for target, actual in zip(
                target_positions, position, strict=True
            )) / 2
        )
        if _touches_keepout(data, keepout_id):
            collision_steps += 1

    if not positions:
        return _failed_metrics(initial_positions, goal_positions, duration_seconds)

    final_positions = positions[-1]
    final_velocities = (float(data.qvel[0]), float(data.qvel[1]))
    tail = squared_errors[-max(1, len(squared_errors) // 5):]
    joint_limits = (1.5, 1.7)
    max_violation = max(
        max(abs(position[index]) - joint_limits[index], 0.0)
        for position in positions
        for index in range(2)
    )
    return TrajectoryTrackingMetrics(
        initial_positions=initial_positions,
        goal_positions=goal_positions,
        duration_seconds=duration_seconds,
        final_positions=final_positions,
        final_velocities=final_velocities,
        final_max_error=max(
            abs(goal - actual)
            for goal, actual in zip(goal_positions, final_positions, strict=True)
        ),
        tracking_rmse=math.sqrt(sum(squared_errors) / len(squared_errors)),
        tail_rmse=math.sqrt(sum(tail) / len(tail)),
        max_abs_commands=tuple(
            max(abs(command[index]) for command in commands) for index in range(2)
        ),  # type: ignore[arg-type]
        max_joint_limit_violation=max_violation,
        collision_steps=collision_steps,
        finite=finite and all(
            math.isfinite(value)
            for position in positions
            for value in position
        ),
    )


def _smooth_target(
    elapsed: float,
    start: JointVector,
    goal: JointVector,
    move_seconds: float,
) -> tuple[JointVector, JointVector]:
    phase = min(max(elapsed / move_seconds, 0.0), 1.0)
    blend = phase * phase * (3.0 - 2.0 * phase)
    blend_rate = 6.0 * phase * (1.0 - phase) / move_seconds if phase < 1.0 else 0.0
    positions = tuple(
        initial + (target - initial) * blend
        for initial, target in zip(start, goal, strict=True)
    )
    velocities = tuple(
        (target - initial) * blend_rate
        for initial, target in zip(start, goal, strict=True)
    )
    return positions, velocities  # type: ignore[return-value]


def _touches_keepout(data: object, keepout_id: int) -> bool:
    for index in range(data.ncon):  # type: ignore[attr-defined]
        contact = data.contact[index]  # type: ignore[attr-defined]
        if contact.geom1 == keepout_id or contact.geom2 == keepout_id:
            return True
    return False


def _failed_metrics(
    initial_positions: JointVector,
    goal_positions: JointVector,
    duration_seconds: float,
) -> TrajectoryTrackingMetrics:
    return TrajectoryTrackingMetrics(
        initial_positions=initial_positions,
        goal_positions=goal_positions,
        duration_seconds=duration_seconds,
        final_positions=(float("nan"), float("nan")),
        final_velocities=(float("nan"), float("nan")),
        final_max_error=float("inf"),
        tracking_rmse=float("inf"),
        tail_rmse=float("inf"),
        max_abs_commands=(float("inf"), float("inf")),
        max_joint_limit_violation=float("inf"),
        collision_steps=0,
        finite=False,
    )
