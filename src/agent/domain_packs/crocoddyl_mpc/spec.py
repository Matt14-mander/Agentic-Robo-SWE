"""Fixed Phase 6.4a optimal-control and timing contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PACK_ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
PACK_VERSION = "0.1.0"
CROCODDYL_VERSION = "3.0.1"
CROCODDYL_COMMIT = "420daeab516e56583c03a96a3b2846510119d740"
STATE_DIM = 4
CONTROL_DIM = 2
HORIZON = 40
MAX_ITERATIONS = 20
WARMUP = 5
REPEATS = 30
TIMING_KEYS = (
    "analytic_calc",
    "analytic_calc_diff",
    "analytic_backward_pass",
    "analytic_try_step",
    "analytic_solve",
    "analytic_rollout",
    "numdiff_calc_diff",
)


def benchmark_contract() -> dict[str, Any]:
    return {
        "name": "phase64_lqr_fddp_profile",
        "state_dim": STATE_DIM,
        "control_dim": CONTROL_DIM,
        "horizon": HORIZON,
        "solver": "SolverFDDP",
        "max_iterations": MAX_ITERATIONS,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "timing_keys": list(TIMING_KEYS),
        "comparison": "analytic ActionModelLQR vs ActionModelNumDiff first derivatives",
        "codegen_replacement": False,
    }
