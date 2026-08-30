"""Buggy two-joint trajectory controller for the Phase 5.1 benchmark."""

from __future__ import annotations

from collections.abc import Sequence


class TwoJointTrajectoryController:
    def __init__(
        self,
        kp: Sequence[float],
        kd: Sequence[float],
        torque_limits: Sequence[float],
    ) -> None:
        self.kp = tuple(kp)
        self.kd = tuple(kd)
        self.torque_limits = tuple(torque_limits)

    def compute(
        self,
        time_seconds: float,
        target_positions: Sequence[float],
        target_velocities: Sequence[float],
        positions: Sequence[float],
        velocities: Sequence[float],
    ) -> tuple[float, float]:
        del time_seconds
        commands = []
        for joint in range(2):
            # BUG: the target indices are reversed, coupling both joint loops.
            target_joint = 1 - joint
            position_error = target_positions[target_joint] - positions[joint]
            velocity_error = target_velocities[target_joint] - velocities[joint]
            commands.append(
                self.kp[joint] * position_error + self.kd[joint] * velocity_error
            )
        # BUG: raw commands bypass the controller's per-joint torque limits.
        return commands[0], commands[1]
