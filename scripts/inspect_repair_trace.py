"""Inspect an operator-owned trace without calling a model or final evaluator."""

import argparse
import json

from agent.repair_trace import inspect_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_trace(args.directory), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
