"""Fixed robot/input and sampling contract, independent of candidate source code."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any


PACK_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PACK_ROOT / "models" / "two_link.urdf"
SEED = 20260904
INPUT_NAMES = ("q_shoulder", "q_elbow", "v_shoulder", "v_elbow", "a_shoulder", "a_elbow")
WARMUP = 32
REPEATS = 50
BATCH_SIZE = 16


def samples() -> list[list[float]]:
    rng = random.Random(SEED)
    points = [
        [0.0] * 6,
        [0.4, -0.8, 0.6, -0.3, 1.1, -0.7],
        [-0.9, 0.2, -0.5, 1.0, -0.4, 0.8],
        [2.5, 2.5, 4.0, 4.0, 5.0, 5.0],
        [-2.5, -2.5, -4.0, -4.0, -5.0, -5.0],
        [2.5, -2.5, 4.0, -4.0, 5.0, -5.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, -3.0],
        [0.0, 0.0, 1.5, -2.0, 0.0, 0.0],
    ]
    points.extend([[rng.uniform(-bound, bound) for bound in (2.5, 2.5, 4, 4, 5, 5)]
                   for _ in range(16)])
    return points


def model_contract() -> dict[str, Any]:
    return {
        "name": "robo_swe_two_link", "nq": 2, "nv": 2,
        "urdf_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "input_names": list(INPUT_NAMES), "input_layout": "[q(2),v(2),a(2)]",
        "output": "joint_torque_Nm", "jacobian_shape": [2, 6],
        "jacobian_layout": "row-major", "base": "fixed",
        "gravity": [0.0, 0.0, -9.81], "seed": SEED,
        "sample_count": len(samples()),
        "output_atol": 1e-9, "output_rtol": 1e-8,
        "jacobian_atol": 1e-7, "jacobian_rtol": 1e-5,
        "warmup": WARMUP, "repeats": REPEATS, "batch_size": BATCH_SIZE,
    }
