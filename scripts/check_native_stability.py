"""Repeated fresh-process reference checks; no retries hide failed observations."""
import argparse
import json
import time
from pathlib import Path

from agent.ablation_native import compile_candidate, execution_receipt
from agent.containers import run_bounded
from agent.domain_packs.cpp_robotics.spec import samples
from agent.domain_packs.cpp_robotics.validators import PACK_ROOT, judge_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100:
        parser.error("runs must be 1..100")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    built = compile_candidate(PACK_ROOT / "harness/reference.hpp", "anti_windup", root, "g++")
    report = {"kind": "reference_stability_not_agent_quality", "build": built, "runs": [], "passed": False}
    if built["passed"]:
        directory = Path(built["artifact_dir"])
        points = samples("anti_windup")
        (directory / "inputs.txt").write_text(str(len(points)) + "\n" + "\n".join(
            str(len(x)) + " " + " ".join(format(v, ".17g") for v in x) for x in points), encoding="ascii")
        for index in range(args.runs):
            output = directory / f"observations-{index}.txt"
            start = time.perf_counter()
            result = run_bounded([built["executable"], "inputs.txt", output.name], cwd=directory, timeout=30)
            receipt = execution_receipt(result, time.perf_counter() - start, output)
            try:
                verdict = judge_observations("anti_windup", output.read_text()) if result.returncode == 0 else None
                receipt["passed"] = bool(verdict and verdict["passed"])
            except (OSError, ValueError) as exc:
                receipt.update(passed=False, error=str(exc))
            report["runs"].append(receipt)
            (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["passed"] = all(r["passed"] for r in report["runs"])
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "runs": len(report["runs"]), "report": str(root / "report.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
