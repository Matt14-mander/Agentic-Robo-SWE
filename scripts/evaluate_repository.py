"""Operator-only final evaluation; never imported into the Agent tool registry."""

import argparse
import json
from pathlib import Path

from agent.evaluation import evaluate_repository


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated final evaluation of a candidate repository")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True, help="Private spec outside candidate workspace")
    parser.add_argument("--output", type=Path, required=True, help="Private artifacts outside candidate workspace")
    args = parser.parse_args()
    try:
        report = evaluate_repository(args.workspace, args.spec, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2 if report["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
