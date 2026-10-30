"""Offline fairness/dispatch/taxonomy checks, independent of model and C++ availability."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agent.ablation as module
from agent.ablation import ARMS, Budget, ScriptedSmokeModel, build_experiment_graph, run_matrix
from agent.failures import classify, classify_trial
from agent.repair_trace import inspect_trace, use_repair_trace
from agent.workspace import HARNESS_ROOT, Workspace, use_workspace


@pytest.mark.parametrize("payload,expected", [
    ({"passed": False, "stage": "build", "error": "C++ failed"}, "compile_error"),
    ({"passed": True, "stage": "capability"}, "none"),
    ({"passed": False, "stage": "capability"}, "environment"),
    ({"passed": False, "stage": "runtime"}, "runtime_error"),
    ({"passed": False, "stage": "correctness"}, "numerical_mismatch"),
    ({"passed": False, "stage": "dense_correctness"}, "derivative_mismatch"),
    ({"passed": False, "stage": "sparse_correctness"}, "derivative_mismatch"),
    ({"passed": False, "stage": "safety"}, "safety_violation"),
    ({"passed": False, "stage": "behavior"}, "behavioral_mismatch"),
    ({"passed": False, "stage": "runtime", "timed_out": True}, "timeout"),
    ({"passed": False, "budget_exhausted": True}, "budget_exhausted"),
    ({"passed": "true"}, "invalid_evidence"),
    ({"passed": True, "error": "contradiction"}, "invalid_evidence"),
    ({"passed": False, "error": "unknown words"}, "unclassified"),
])
def test_failure_taxonomy_uses_stage_not_error_text(payload, expected):
    assert classify(payload).category == expected


def make_workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "candidate.hpp"
    source.write_text("double dt=1/static_cast<int>(x[3]);", encoding="utf-8")
    return Workspace.create(root, tmp_path / "artifacts", trusted_local=True), source


def test_four_arm_policy_and_trace_boundary(tmp_path):
    workspace, source = make_workspace(tmp_path)
    policies = []
    for arm in ARMS:
        source.write_text("double dt=1/static_cast<int>(x[3]);", encoding="utf-8")
        calls = []
        def build():
            calls.append("compile")
            return {"passed": True, "stage": "build"}
        def validate():
            calls.append("dev")
            return {"passed": False, "stage": "correctness", "private_artifact_path": "SECRET"}
        directory = workspace.artifacts / arm
        with use_workspace(workspace), use_repair_trace(directory, workspace.root, {}):
            graph, policy = build_experiment_graph(arm=arm, source=source, budget=Budget(),
                notes="DOMAIN", model=ScriptedSmokeModel("sample_period"),
                compile_fn=build, validate_fn=validate)
            state = graph.invoke({"messages": [HumanMessage(content="TASK")]})
        policies.append(policy)
        assert state.get("stop") is None
        assert calls == (["compile", "dev"] if arm == "feedback" else
                         ["compile"] if arm == "tools" else [])
        text = (directory / "events.jsonl").read_text(encoding="utf-8")
        assert "SECRET" not in text
        summary = inspect_trace(directory)
        assert summary["development_validations"] == int(arm == "feedback")
        assert summary["source_changes"] == 1
    assert policies[0]["tools"] == policies[1]["tools"] == ["read_candidate", "patch_candidate"]
    assert policies[0]["system_sha256"] != policies[1]["system_sha256"]
    assert policies[1]["system_sha256"] == policies[2]["system_sha256"] == policies[3]["system_sha256"]
    assert policies[0]["tool_schema_sha256"] == policies[1]["tool_schema_sha256"]
    assert policies[3]["tools"] == [*policies[2]["tools"], "validate_development"]


class Model:
    def __init__(self, names, usage=True):
        self.names, self.usage = names, usage
        self.calls = 0
    def bind_tools(self, _tools):
        return self
    def bind(self, **kwargs):
        assert kwargs["max_tokens"] > 0
        return self
    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": {}, "id": str(i)} for i, name in enumerate(self.names)],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            if self.usage else None)


@pytest.mark.parametrize("names,budget,usage,expected", [
    (["validate_development"], Budget(), True, "tool_error"),
    (["execute_python"], Budget(), True, "tool_error"),
    (["read_candidate", "read_candidate"], Budget(max_tools=1), True, "budget_exhausted"),
    (["read_candidate"], Budget(max_tokens=1), True, "budget_exhausted"),
    (["read_candidate"], Budget(), False, "invalid_evidence"),
    (["read_candidate"], Budget(max_steps=1), True, "budget_exhausted"),
])
def test_dispatch_and_budgets_cannot_be_bypassed(tmp_path, names, budget, usage, expected):
    workspace, source = make_workspace(tmp_path)
    def forbidden():
        pytest.fail("No evaluator should be reachable")
    model = Model(names, usage)
    trace = workspace.artifacts / "budget-trace"
    with use_workspace(workspace), use_repair_trace(trace, workspace.root, {}):
        graph, _ = build_experiment_graph(arm="base", source=source, budget=budget,
            notes="", model=model, compile_fn=forbidden, validate_fn=forbidden)
        result = graph.invoke({"messages": [HumanMessage(content="fix")]})
    assert result["failure"]["category"] == expected
    assert result.get("tools_used", 0) <= budget.max_tools
    assert model.calls == 1
    assert source.read_text() == "double dt=1/static_cast<int>(x[3]);"
    rows = [json.loads(line) for line in (trace / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["kind"] == "node_completed" and
               (row["data"].get("failure") or {}).get("category") == expected for row in rows)


def test_matrix_freezes_inputs_resets_workspaces_and_keeps_final_verdict_outside(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "compiler_identity", lambda _: {
        "passed": True, "compiler": "offline-stub", "version": "test"})
    monkeypatch.setattr(module, "compile_candidate", lambda *_: {"passed": True, "stage": "build"})
    def evaluate(source, scenario, *_):
        broken = (HARNESS_ROOT / f"benchmarks/fixtures/cpp_robotics/{scenario}.hpp").read_bytes()
        return {"passed": source.read_bytes() != broken, "stage": "correctness"}
    monkeypatch.setattr(module, "evaluate_candidate", evaluate)
    requests = []
    def worker(request, _directory):
        requests.append(request)
        return module.run_attempt(request)
    monkeypatch.setattr(module, "run_worker", worker)
    report = run_matrix(output=tmp_path / "matrix", budget=Budget(), repeats=2, seed=3,
                        selected=["cpp_sample_period", "cpp_joint_wrap"], trusted_local=True)
    assert report["status"] == "completed"
    assert len(report["results"]) == 16
    assert all(row["success"] for row in report["results"])
    assert len({request["source"] for request in requests}) == 16
    assert len({json.dumps(request["budget"]) for request in requests}) == 1
    for case_id in {row["case_id"] for row in report["results"]}:
        members = [r for r in report["results"] if r["case_id"] == case_id]
        assert len({r["source_before"] for r in members}) == 1
        for row in members:
            directory = Path(row["trace"])
            assert inspect_trace(directory)["development_validations"] == int(row["arm"] == "feedback")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["environment"]["model"] == "scripted_smoke_v1"
    assert all(value["case_cluster_bootstrap_95"] is None for value in report["summary"].values())
    # Audit real traces against aggregate counters, then demonstrate evidence tampering detection.
    import runpy
    helpers = runpy.run_path(str(HARNESS_ROOT / "scripts/analyze_ablation.py"))
    analyze = helpers["analyze"]
    analysis = analyze(tmp_path / "matrix")
    assert analysis["audit_ok"]
    assert analysis["summary"]["base"]["usage"]["total_tokens"] == 4 * 45
    assert analysis["summary"]["feedback"]["tool_calls"]["validate_development"] == 4
    assert not analysis["summary"]["base"]["cache_usage_complete"]
    from copy import deepcopy
    merge = helpers["merge_calibration_recovery"]
    partial_report = deepcopy(report)
    partial_report.update(status="interrupted", results=report["results"][:-1])
    report_path = tmp_path / "matrix" / "report.json"
    report_path.write_text(json.dumps(partial_report), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        analyze(tmp_path / "matrix")
    partial = analyze(tmp_path / "matrix", allow_incomplete=True)
    assert partial["audit_ok"] and not partial["experiment_complete"]
    assert partial["observed_runs"] == 15 and len(partial["missing_attempts"]) == 1
    assert partial["summary_scope"] == "observed_attempts_only"
    with pytest.raises(ValueError, match="Incomplete"):
        merge(partial, analysis)
    partial_report["planned_runs"] = 15
    report_path.write_text(json.dumps(partial_report), encoding="utf-8")
    with pytest.raises(ValueError, match="Planned runs"):
        analyze(tmp_path / "matrix", allow_incomplete=True)
    partial_report["planned_runs"] = report["planned_runs"]
    partial_report["results"].append(partial_report["results"][0])
    report_path.write_text(json.dumps(partial_report), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        analyze(tmp_path / "matrix", allow_incomplete=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    blocked = deepcopy(analysis)
    for row in blocked["results"]:
        row.update(calibrated=False, success=False, usage={})
        row["audit"]["trace_available"] = False
    recovered = merge(blocked, analysis)
    assert recovered["summary"]["base"]["usage"] == analysis["summary"]["base"]["usage"]
    assert len(recovered["recovered_pre_model_blocks"]) == 16
    with pytest.raises(ValueError, match="blocked before model"):
        merge(analysis, analysis)
    blocked["results"][0]["source_before"] = "different-source"
    with pytest.raises(ValueError, match="starting source"):
        merge(blocked, analysis)
    changed = tmp_path / "matrix" / "runs" / "cpp_sample_period-1-base" / "workspace" / "candidate.hpp"
    changed.write_text("tampered", encoding="utf-8")
    assert not analyze(tmp_path / "matrix")["audit_ok"]
    events_file = Path(report["results"][0]["trace"]) / "events.jsonl"
    with events_file.open("ab") as stream:
        stream.write(b'{"partial":')
    assert not analyze(tmp_path / "matrix")["audit_ok"]
    report["status"] = "running"
    (tmp_path / "matrix" / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        analyze(tmp_path / "matrix")


def test_invalid_calibration_blocks_all_arms_without_model_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "compiler_identity", lambda _: {
        "passed": False, "stage": "capability", "error": "missing compiler"})
    monkeypatch.setattr(module, "run_worker", lambda *_: pytest.fail("Invalid fixture spent model budget"))
    report = run_matrix(output=tmp_path / "blocked", budget=Budget(),
                        selected=["cpp_joint_wrap"], trusted_local=True)
    assert len(report["results"]) == 4
    assert all(row["failure"]["category"] == "environment" for row in report["results"])
    assert all(value["calibrated_runs"] == 0 for value in report["summary"].values())


def test_rejects_unsafe_or_ambiguous_matrix_options(tmp_path):
    with pytest.raises(ValueError, match="trusted_local"):
        run_matrix(output=tmp_path / "no", budget=Budget())
    with pytest.raises(ValueError, match="unique cases"):
        run_matrix(output=tmp_path / "no", budget=Budget(), selected=["unknown"], trusted_local=True)
    with pytest.raises(ValueError, match="positive"):
        Budget(max_tools=0)


@pytest.mark.parametrize("result,expected", [
    ({}, "invalid_evidence"),
    ({"failure": {"category": "none", "schema_version": 1}}, "invalid_evidence"),
    ({"failure": {"category": "imaginary", "schema_version": 1}}, "invalid_evidence"),
    ({"loop_step": 1, "tools_used": 1, "input_tokens": 10, "output_tokens": 5,
      "total_tokens": 16}, "invalid_evidence"),
    ({"loop_step": 1, "tools_used": 100, "input_tokens": 10, "output_tokens": 5,
      "total_tokens": 15}, "budget_exhausted"),
])
def test_parent_rejects_incomplete_or_over_budget_worker_results(result, expected):
    assert module.worker_failure(result, Budget())["category"] == expected


@pytest.mark.parametrize("diagnosis,expected", [
    ({"status": "fail", "first_safety_event": {"kind": "collision"}}, "safety_violation"),
    ({"status": "fail", "category": "tracking"}, "behavioral_mismatch"),
    ({"status": "pass", "category": "none"}, "none"),
    ({"status": []}, "invalid_evidence"),
])
def test_simulation_failure_adapter(diagnosis, expected):
    assert classify_trial(diagnosis).category == expected


@pytest.mark.parametrize("output", ['{"passed":"true"}', '{"passed":true,"passed":false}',
                                   '{"passed":true,"error":"bad"}', '{"passed":false,"details":null}'])
def test_legacy_validator_malformed_verdicts_are_classified(monkeypatch, output):
    from subprocess import CompletedProcess
    import agent.benchmark as benchmark
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_args, **_kwargs:
                        CompletedProcess([], 0, output, ""))
    result = benchmark.validate_case(benchmark.load_cases()[0])
    assert not result.passed
    assert result.failure["category"] == "invalid_evidence"
