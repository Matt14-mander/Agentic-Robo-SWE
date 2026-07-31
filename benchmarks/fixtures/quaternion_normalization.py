"""Intentionally buggy quaternion normalization helper."""

from __future__ import annotations


def normalize_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    squared_norm = sum(component * component for component in quaternion)
    return tuple(component / squared_norm for component in quaternion)  # type: ignore[return-value]
