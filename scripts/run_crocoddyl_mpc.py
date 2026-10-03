"""Run the Phase 6.4a deterministic Crocoddyl solver-chain profile."""

import json
import argparse

from agent.domain_packs.crocoddyl_mpc.validators import validate_crocoddyl_mpc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--robot", action="store_true", help="Phase 6.4b two-link MPC")
    mode.add_argument("--codegen", action="store_true", help="Controlled three-way ABA Jacobian replacement")
    parser.add_argument("--quiet", action="store_true", help="Print final JSON only")
    args = parser.parse_args()
    progress = None if args.quiet else lambda message: print(f"[crocoddyl] {message}", flush=True)
    result = validate_crocoddyl_mpc(robot=args.robot, codegen=args.codegen, progress=progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("passed") is True else 1)
