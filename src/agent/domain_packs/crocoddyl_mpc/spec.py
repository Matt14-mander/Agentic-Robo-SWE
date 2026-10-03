"""Fixed Phase 6.4a optimal-control and timing contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PACK_ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
PACK_VERSION = "0.3.0"
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


MPC_STEPS = 32
MPC_REPEATS = 10
MPC_WARMUP = 2
MPC_TIMING_KEYS = ("calc", "calc_diff", "backward_pass", "try_step", "solve", "rollout")


def robot_contract() -> dict[str, Any]:
    from agent.domain_packs.pinocchio_rnea.spec import model_contract

    return {
        "name": "phase64b_two_link_mpc", "urdf_sha256": model_contract()["urdf_sha256"],
        "state_dim": 4, "control_dim": 2, "horizon": 40, "dt": 0.02,
        "steps": MPC_STEPS, "max_iterations": 100, "solver_tolerance": 1e-6,
        "warmup": MPC_WARMUP, "repeats": MPC_REPEATS,
        "initial_state": [0.4, -0.3, 0.0, 0.0], "reference_state": [0.0] * 4,
        "state_weights": [100.0, 100.0, 5.0, 5.0], "control_weight": 0.001,
        "terminal_scale": 20.0, "strategies": ["cold", "shifted_warm"],
        "timing_keys": list(MPC_TIMING_KEYS), "codegen_replacement": False,
        "plant": "independent Pinocchio ABA + symplectic Euler",
    }


TRIAL_BACKENDS = ("native", "cppad", "codegen")


def trial_samples() -> list[list[float]]:
    import math
    return [[0.6 * math.sin((i + 1) * (j + 1) * 0.7) for j in range(6)]
            for i in range(24)] + [[0., 0., 0., 0., -16.4808, -2.7468]]


def trial_contract() -> dict[str, Any]:
    return {"name": "phase64c_controlled_aba_jacobian_replacement",
            "baseline": robot_contract(), "backends": list(TRIAL_BACKENDS),
            "input_order": ["q1", "q2", "v1", "v2", "tau1", "tau2"],
            "jacobian_layout": "row_major_2_by_6", "samples": trial_samples(),
            "replacement": "ABA Jacobian only in running action calcDiff",
            "unchanged": ["ABA values", "costs", "terminal", "Euler integrator",
                          "FDDP", "shifted warm start", "independent ABA plant"],
            "repeats": MPC_REPEATS, "warmup": MPC_WARMUP,
            "processes": 1, "minimum_speedup": 1.05, "maximum_cv": 0.3,
            "scope": "fixed fully actuated two-link; no contacts or armature"}
