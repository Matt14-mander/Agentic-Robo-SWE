"""Intentionally buggy angle conversion helper."""

from __future__ import annotations

import math


def degrees_to_radians(angle_degrees: float) -> float:
    return angle_degrees * 180.0 / math.pi
