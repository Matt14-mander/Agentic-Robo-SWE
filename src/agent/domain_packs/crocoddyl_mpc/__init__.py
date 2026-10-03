"""Optional Crocoddyl solver-chain diagnostics without import-time installation."""

import json

from langchain_core.tools import tool

from agent.domain.loader import load_domain_manifest
from agent.domain.models import DomainPack
from agent.domain_packs.crocoddyl_mpc.dependencies import inspect_dependencies
from agent.domain_packs.crocoddyl_mpc.spec import PACK_ROOT, benchmark_contract, robot_contract
from agent.domain_packs.crocoddyl_mpc.validators import validate_crocoddyl_mpc, validate_robot_mpc


@tool
def inspect_crocoddyl_mpc_environment() -> str:
    """Inspect pinned Crocoddyl capabilities and the fixed Phase 6.4a profile contract."""
    return json.dumps(
        {"environment": inspect_dependencies(), "contract": benchmark_contract(),
         "robot_contract": robot_contract()}, indent=2
    )


@tool
def profile_crocoddyl_mpc_solver_chain() -> str:
    """Run the deterministic LQR/FDDP correctness gate and stage-level latency profile."""
    return json.dumps(validate_crocoddyl_mpc(), ensure_ascii=False, indent=2)


@tool
def benchmark_crocoddyl_robot_mpc() -> str:
    """Validate real two-link Pinocchio MPC and compare cold vs shifted warm start."""
    return json.dumps(validate_robot_mpc(), ensure_ascii=False, indent=2)


def create_pack() -> DomainPack:
    return DomainPack(
        manifest=load_domain_manifest(PACK_ROOT / "pack.toml"),
        tools=(inspect_crocoddyl_mpc_environment, profile_crocoddyl_mpc_solver_chain,
               benchmark_crocoddyl_robot_mpc),
        validators={"crocoddyl_mpc.solver_chain": validate_crocoddyl_mpc,
                    "crocoddyl_mpc.robot_mpc": validate_robot_mpc},
        prompt_fragment=(PACK_ROOT / "prompt.md").read_text(encoding="utf-8"),
        capability_probe=inspect_dependencies,
        source=PACK_ROOT,
    )
