"""Intentionally buggy linear trajectory sampler."""

from __future__ import annotations


def linear_trajectory(start: float, end: float, samples: int) -> list[float]:
    step = (end - start) / samples
    return [start + index * step for index in range(samples)]
