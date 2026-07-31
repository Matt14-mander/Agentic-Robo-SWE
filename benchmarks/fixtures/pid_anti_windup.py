"""Intentionally buggy PID controller without anti-windup."""

from __future__ import annotations


class PIDController:
    def __init__(self, kp: float, ki: float, output_limit: float) -> None:
        self.kp = kp
        self.ki = ki
        self.output_limit = output_limit
        self.integral = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        raw_output = self.kp * error + self.ki * self.integral
        return max(-self.output_limit, min(self.output_limit, raw_output))
