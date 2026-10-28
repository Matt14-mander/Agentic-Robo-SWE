"""One ablation attempt; private request data never becomes graph state."""

import json
import sys
from pathlib import Path

from agent.ablation import run_attempt
from agent.benchmark import _write_json_atomic
from agent.failures import Failure


def main() -> int:
    request_path, output_path = map(Path, sys.argv[1:])
    try:
        result = run_attempt(json.loads(request_path.read_text(encoding="utf-8")))
    except Exception as exc:
        result = {"failure": Failure("unclassified", "worker", type(exc).__name__).to_dict()}
    _write_json_atomic(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
