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
    parser.add_argument("--repeats", type=int, default=None,
                        help="With --codegen: paired repetitions per process, 10-200 (default: 10; follow-up: 50-100)")
    parser.add_argument("--processes", type=int, default=1,
                        help="With --codegen: 3-20 sequential fresh native processes (default: single trial)")
    args = parser.parse_args()
    progress = None if args.quiet else lambda message: print(f"[crocoddyl] {message}", flush=True)
    if args.processes != 1 and (not args.codegen or not 3 <= args.processes <= 20):
        parser.error("--processes requires --codegen and a count in [3, 20]")
    if args.repeats is not None and (not args.codegen or not 10 <= args.repeats <= 200):
        parser.error("--repeats requires --codegen and a count in [10, 200]")
    repeats = args.repeats if args.repeats is not None else 10
    result = (validate_process_trial(processes=args.processes, repeats=repeats, progress=progress) if args.processes > 1
              else validate_crocoddyl_mpc(robot=args.robot, codegen=args.codegen, repeats=repeats, progress=progress))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("passed") is True else 1)
