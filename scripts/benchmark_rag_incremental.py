"""Compare actual historical/new retrieval on synthetic C++ and persistent Chroma.

Offline, no LLM/semantic weights. Run from the repository with its rag extra.
The trusted baseline source is loaded directly from the specified local Git commit.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types
from importlib.metadata import version
from pathlib import Path

from agent.rag.embedding import EmbeddingConfig
from agent.rag.index import open_code_index
from agent.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


def baseline_module(ref):
    commit = subprocess.check_output(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                                     cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{commit}:src/agent/rag/index.py"], cwd=ROOT)
    module = types.ModuleType("_rag_baseline_benchmark")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{commit}:src/agent/rag/index.py", "exec"), module.__dict__)
    return module, commit


class BaselineMetrics:
    def __init__(self, module, collection):
        self.raw = collection
        self.values = dict(source_content_reads=0, source_bytes_read=0,
                           collection_full_reads=0, lexical_documents_tokenized=0)
        read = module.read_workspace_source

        def measured_read(*args):
            path, source = read(*args)
            self.values["source_content_reads"] += 1
            self.values["source_bytes_read"] += len(source.encode("utf-8"))
            return path, source

        module.read_workspace_source = measured_read

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def get(self, include):
        result = self.raw.get(include=include)
        self.values["collection_full_reads"] += 1
        if "documents" in include:
            self.values["lexical_documents_tokenized"] += len(result["ids"])
        return result


def timed(index, queries, metrics, *, cache):
    before = dict(metrics())
    times, signatures = [], []
    for query in queries:
        start = time.perf_counter()
        results = index.search(query, strategy="lexical", top_k=5, use_cache=cache)
        times.append((time.perf_counter() - start) * 1000)
        signatures.append([(item.path, item.start_line, item.end_line, item.content, item.score)
                           for item in results])
    after = metrics()
    p95 = sorted(times)[max(0, int(len(times) * .95 + .999) - 1)]
    report = {"queries": len(queries), "p50_ms": round(statistics.median(times), 3),
              "p95_ms": round(p95, 3), "samples_ms": [round(value, 3) for value in times],
              "work": {key: after[key] - before[key] for key in before
                       if not key.startswith("last_")}}
    return report, signatures


def run_size(directory, files, queries, ref):
    root = directory / "repo"
    root.mkdir(parents=True)
    for number in range(files):
        # Distinct query marker in each file, common C++ vocabulary and body.
        source = f"// probe{number:05d}\n" + "".join(
            f"double value_{line}(double torque) {{ return torque * {line}.125; }} // controller joint velocity\n"
            for line in range(119))
        (root / f"unit_{number:05d}.cpp").write_text(source, encoding="utf-8")
    workspace = Workspace.create(root, directory / "incremental-artifacts")
    old_workspace = Workspace.create(root, directory / "baseline-artifacts")
    new = open_code_index(workspace=workspace, embedding=EmbeddingConfig(backend="hash"))
    seed = open_code_index(workspace=old_workspace, embedding=EmbeddingConfig(backend="hash"))
    start = time.perf_counter()
    stats = new.sync()
    build_ms = (time.perf_counter() - start) * 1000
    seed.sync()
    module, commit = baseline_module(ref)
    counter = BaselineMetrics(module, seed.collection.raw)
    old = module.CodeIndex(counter, workspace=old_workspace)
    report = {"files": files, "chunks": stats.chunks,
              "source_bytes": sum(path.stat().st_size for path in root.iterdir()),
              "baseline_commit": commit, "incremental_initial_sync_ms": round(build_ms, 3)}
    arms = [("baseline", old, lambda: counter.values), ("incremental", new, new.performance_stats)]
    signatures = {}
    for name, index, metrics in arms:
        report[name] = {}
        cold, _ = timed(index, ["probe00000"], metrics, cache=True)
        report[name]["cold_first_search"] = cold
    # Alternate arm order per phase; all source reads are OS-cache-warm, not disk throughput.
    for phase, requests, cache in [
        ("warm_uncached", [f"probe{number % files:05d}" for number in range(queries)], False),
        ("warm_cached", ["probe00000"] * queries, True),
        ("common_term_uncached", ["torque"] * queries, False),
    ]:
        for name, index, metrics in reversed(arms) if phase == "warm_cached" else arms:
            report[name][phase], signatures[name] = timed(index, requests, metrics, cache=cache)
        assert signatures["baseline"] == signatures["incremental"], phase
    source = root / "unit_00000.cpp"
    source.write_text(source.read_text(encoding="utf-8").replace("probe00000", "fresh00000"), encoding="utf-8")
    for name, index, metrics in arms:
        report[name]["single_file_edit"], signatures[name] = timed(index, ["fresh00000"], metrics, cache=True)
        assert signatures[name][0] and "fresh00000" in signatures[name][0][0][3]
        assert not index.search("probe00000", strategy="lexical")
    assert signatures["baseline"] == signatures["incremental"]
    report["ranking_and_content_equal"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, nargs="+", default=[200, 1000])
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--baseline-ref", default="175c11b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.files) < 1 or args.queries < 1:
        parser.error("files and queries must be positive")
    state = ROOT / ".agent_state" / "incremental-benchmarks"
    state.mkdir(parents=True, exist_ok=True)
    # Keep generated Chroma files: Windows holds open database handles in-process.
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=state))
    report = {"schema": "rag-incremental-performance-v1", "platform": platform.platform(),
              "python": platform.python_version(), "chromadb": version("chromadb"),
              "encoder": "offline hash", "strategy": "lexical", "top_k": 5,
              "notes": "Synthetic C++; OS-cache-warm. Cold = first search after sync. "
                       "Times include freshness, Chroma and result verification. "
                       "Eight samples default: p95 is descriptive, not an SLA. "
                       "Baseline does not have the new return-time verification.",
              "results": []}
    for size in args.files:
        result = run_size(directory / str(size), size, args.queries, args.baseline_ref)
        report["results"].append(result)
        print(json.dumps({"files": size, "equal": result["ranking_and_content_equal"],
                          "baseline_ms": result["baseline"]["warm_uncached"]["p50_ms"],
                          "incremental_ms": result["incremental"]["warm_uncached"]["p50_ms"]}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
