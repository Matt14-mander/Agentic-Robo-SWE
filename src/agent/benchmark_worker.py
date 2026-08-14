"""Isolated graph invocation used by the benchmark timeout boundary.

The parent benchmark process owns reporting and validation. This worker only runs
one Agent attempt and serializes its LangGraph result back to the parent.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    request = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    payload: tuple[str, Any]
    try:
        from agent.graph import graph

        payload = ("ok", graph.invoke(request["state"], config=request["config"]))
    except BaseException as exc:  # Return failures to the parent without losing their type.
        payload = ("error", f"{type(exc).__name__}: {exc}")

    Path(args.output_path).write_bytes(pickle.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
