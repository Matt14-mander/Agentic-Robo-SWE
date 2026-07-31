"""Intentionally buggy 2-link planar inverse kinematics."""

from __future__ import annotations

import math


def inverse_kinematics_2link(
    x: float,
    y: float,
    l1: float = 1.0,
    l2: float = 1.0,
) -> tuple[float, float]:
    distance_sq = x * x + y * y
    cos_theta2 = (distance_sq - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    theta2 = math.acos(cos_theta2)
    k1 = l1 + l2 * math.cos(theta2)
    k2 = l2 * math.sin(theta2)
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)
    return theta1, theta2
