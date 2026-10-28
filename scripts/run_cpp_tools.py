"""Inspect/build an external C++ repository without an LLM call."""

import argparse
import json

from agent.cpp import build_root, compilation_database, configure_build, semantic_query
from agent.workspace import Workspace, use_workspace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("configure", "build", "commands", "definition", "references", "hover", "diagnostics"))
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--trusted-local", action="store_true")
    parser.add_argument("--execution-image")
    parser.add_argument("--target", default="all")
    parser.add_argument("--path")
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--column", type=int, default=1)
    args = parser.parse_args()
    try:
        workspace = Workspace.create(args.workspace, args.artifact_root,
                                     trusted_local=args.trusted_local, execution_image=args.execution_image)
        with use_workspace(workspace):
            if args.action in ("configure", "build"):
                result = configure_build(args.target, args.action == "configure")
            elif args.action == "commands":
                result = compilation_database(build_root(workspace) / "compile_commands.json")
            else:
                if not args.path:
                    raise ValueError("Semantic queries require --path")
                result = semantic_query(args.path, args.action, args.line, args.column)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if isinstance(result, dict) and result.get("passed") is False else 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
