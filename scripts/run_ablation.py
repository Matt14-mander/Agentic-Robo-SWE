"""Run the M7.1 four-arm public C++ experiment; smoke mode never calls an LLM."""

import argparse
import json
import uuid
from pathlib import Path

from agent.ablation import Budget, run_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Scripted patches + real native validation; no LLM")
    mode.add_argument("--live", action="store_true", help="Run configured paid LLM")
    parser.add_argument("--trusted-local", action="store_true", help="Allow native candidate code on host")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tools", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    output = args.output or Path(".agent_state/ablations") / uuid.uuid4().hex
    try:
        report = run_matrix(output=output, budget=Budget(args.max_steps, args.max_tools,
                            args.max_tokens, args.timeout), repeats=args.repeats, seed=args.seed,
                            mode="scripted_smoke" if args.smoke else "live", compiler=args.compiler,
                            selected=args.cases, trusted_local=args.trusted_local)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps({"mode": report["mode"], "status": report["status"],
                      "summary": report["summary"], "report": str(output.resolve() / "report.json")},
                     ensure_ascii=False, indent=2))
    return 0 if report["results"] and all(row["success"] for row in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
