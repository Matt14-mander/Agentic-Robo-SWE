"""Buggy single-joint PD controller for the Phase 5 MuJoCo benchmark."""


class JointPDController:
    def __init__(self, kp: float, kd: float, torque_limit: float) -> None:
        self.kp = kp
        self.kd = kd
        self.torque_limit = torque_limit

    def compute(self, target: float, position: float, velocity: float) -> float:
        # BUG: error direction is reversed and the requested torque is never limited.
        error = position - target
        return self.kp * error - self.kd * velocity
