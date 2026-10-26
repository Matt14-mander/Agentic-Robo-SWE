"""Verify both sides of each controlled C++ repair task, without calling an LLM."""
from __future__ import annotations

import argparse
import json
import uuid

from agent.benchmark import load_cases, select_cases
from agent.domain_packs.cpp_robotics.validators import PACK_ROOT, validate_robotics_cpp
from agent.tools._paths import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], dest="case_ids")
    args = parser.parse_args()
    cases = select_cases(load_cases("benchmarks/cpp_robotics_cases.json"), args.case_ids)
    directory = PROJECT_ROOT / "benchmarks/results" / ("cpp-fixture-check-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema_version": 1, "kind": "fixture_self_check_not_agent_evaluation",
                    "provenance": "synthetic_controlled", "cases": [],
                    "status": "running", "planned_cases": len(cases), "passed": False}
    for case in cases:
        scenario = case.validator.split(".", 1)[1]
        broken = validate_robotics_cpp(case.template, scenario=scenario)
        repaired = validate_robotics_cpp(str(PACK_ROOT / "harness/reference.hpp"), scenario=scenario)
        passed = (broken["passed"] is False and broken["error"] is None
                  and repaired["passed"] is True and repaired["error"] is None)
        report["cases"].append({"case_id": case.id, "passed": passed, "broken": broken, "reference": repaired})
        complete = len(report["cases"]) == len(cases)
        report.update(status="completed" if complete else "running", completed_cases=len(report["cases"]),
                      passed=complete and all(item["passed"] for item in report["cases"]))
        path = directory / "report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{case.id}: {'PASS' if passed else 'FAIL'} (broken + reference)", flush=True)
    print(f"Report: {directory / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
