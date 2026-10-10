"""Run Crocoddyl baselines and single/independent-process controlled CodeGen trials."""

import json
import argparse

from agent.domain_packs.crocoddyl_mpc.validators import validate_crocoddyl_mpc
from agent.domain_packs.crocoddyl_mpc.process_trial import validate_process_trial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--robot", action="store_true", help="Phase 6.4b two-link MPC")
    mode.add_argument("--codegen", action="store_true", help="Controlled three-way ABA Jacobian replacement")
    parser.add_argument("--quiet", action="store_true", help="Print final JSON only")
    parser.add_argument("--processes", type=int, default=1,
                        help="With --codegen: 3-20 sequential fresh native processes (default: single trial)")
    args = parser.parse_args()
    progress = None if args.quiet else lambda message: print(f"[crocoddyl] {message}", flush=True)
    if args.processes != 1 and (not args.codegen or not 3 <= args.processes <= 20):
        parser.error("--processes requires --codegen and a count in [3, 20]")
    result = (validate_process_trial(processes=args.processes, progress=progress) if args.processes > 1
              else validate_crocoddyl_mpc(robot=args.robot, codegen=args.codegen, progress=progress))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("passed") is True else 1)
