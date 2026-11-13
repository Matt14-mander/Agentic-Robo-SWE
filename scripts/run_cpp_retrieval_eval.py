"""Frozen, public C++ dev/heldout retrieval comparison, separate from Harness code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from agent.rag.cli import add_embedding_arguments, embedding_config
from agent.rag.evaluation import load_rag_cases
from agent.tools._paths import PROJECT_ROOT, resolve_within_root
from agent.workspace import Workspace
if __package__:
    from .run_rag_eval import compare
else:
    from run_rag_eval import compare

DATASET = PROJECT_ROOT / "benchmarks/retrieval_cpp"


def validate_dataset(root: Path = DATASET) -> dict:
    splits = {name: load_rag_cases(root / f"{name}.json") for name in ("dev", "heldout")}
    targets = {name: {target.path for case in cases for target in case.targets}
               for name, cases in splits.items()}
    if targets["dev"] & targets["heldout"]:
        raise ValueError("C++ dev/heldout target files must be disjoint")
    identifiers = [case.id for cases in splits.values() for case in cases]
    queries = [" ".join(case.query.lower().split()) for cases in splits.values() for case in cases]
    if len(set(identifiers)) != len(identifiers) or len(set(queries)) != len(queries):
        raise ValueError("C++ splits contain duplicate case IDs or queries")
    corpus = root / "corpus"
    files = {path.relative_to(corpus).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(corpus.rglob("*")) if path.is_file()}
    for paths in targets.values():
        if not paths <= files.keys():
            raise ValueError("C++ retrieval target is missing from the corpus")
    if not files or any(not name.endswith(".hpp") for name in files):
        raise ValueError("The C++ corpus must contain only frozen headers, not query labels")
    return {
        "name": "independent-cpp-v1", "public_heldout": True,
        "corpus_files": files, "corpus_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "manifest_sha256": {name: hashlib.sha256((root / f"{name}.json").read_bytes()).hexdigest()
                            for name in splits},
        "case_counts": {name: len(cases) for name, cases in splits.items()},
        "target_files_disjoint": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--hash-only", action="store_true", help="Offline CI smoke, no semantic model")
    parser.add_argument("--output", help="Report and isolated index directory")
    parser.add_argument("--rebuild", action="store_true")
    add_embedding_arguments(parser)
    args = parser.parse_args()
    try:
        if args.embedding is not None:
            raise ValueError("Comparison controls backends; use --hash-only or omit --embedding")
        frozen = validate_dataset()
        cases = load_rag_cases(DATASET / f"{args.split}.json")
        destination = resolve_within_root(args.output or f".agent_state/cpp-retrieval/{args.split}")
        workspace = Workspace.create(DATASET / "corpus", destination / "index-artifacts")
        options = SimpleNamespace(output=str(destination), persist_dir=None, roots=["."],
                                  rebuild=args.rebuild, top_ks=[1, 3, 5])
        compare(options, cases, embedding_config(args), workspace,
                backends=("hash",) if args.hash_only else ("hash", "semantic"))
        report_path = destination / "comparison.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["dataset"] = {**frozen, "split": args.split,
                             "purpose": "offline-smoke" if args.hash_only else "retrieval-comparison"}
        try:
            unchanged = frozen == validate_dataset()
        except ValueError:
            unchanged = False
        if not unchanged:
            report["status"] = "incomplete"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not unchanged:
            raise ValueError("C++ dataset changed during evaluation")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
