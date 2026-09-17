"""Run sequential fresh-process Phase 6.3b RNEA control-loop measurements."""

import argparse
import json

from agent.domain_packs.pinocchio_rnea.control_benchmark import run_control_loop_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=3,
                        help="Sequential fresh processes (3-9, default: 3)")
    parser.add_argument("--require-benefit", action="store_true",
                        help="Exit nonzero unless the repeated-process benefit is validated")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stage progress and print only the final JSON")
    args = parser.parse_args()
    progress = None if args.quiet else lambda message: print(f"[phase6.3b] {message}", flush=True)
    result = run_control_loop_benchmark(args.processes, progress=progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = result.get("passed") is True
    if args.require_benefit:
        passed = passed and result.get("recommendation") == "validated_end_to_end_candidate"
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
