"""Offline split/routing checks are not substitutes for native AD/RNEA smoke."""
import json
from pathlib import Path

import pytest

import agent.ablation as runner
import agent.ablation_native as native
import agent.ablation_suite as suite
from agent.benchmark import load_cases
from agent.containers import ProcessResult, run_bounded
from agent.repair_trace import inspect_trace


def test_all_seventeen_cases_have_disjoint_reproducible_populations():
    cases = load_cases("benchmarks/cpp_professional_cases.json")
    assert len(cases) == 17
    for case in cases:
        manifest = suite.split_manifest(case.id)
        dev, final = (suite.split_inputs(case.id, split) for split in ("development", "heldout"))
        assert dev == suite.split_inputs(case.id, "development")
        assert set(map(tuple, dev)).isdisjoint(map(tuple, final))
        assert manifest["splits"]["heldout"]["sha256"] != manifest["splits"]["development"]["sha256"]
        assert len(dev) == len(final) and len(dev) >= 24
        assert suite.reference_text(case.id)
    with pytest.raises(ValueError, match="Unsupported"):
        suite.family("unknown")
    with pytest.raises(KeyError):
        suite.split_inputs("cpp_joint_wrap", "unknown")


@pytest.mark.parametrize("exit_code,termination,output,expected", [
    (17, None, "1 0\n", "runtime_error"),
    (124, "timeout", "", "timeout"),
    (0, None, "1 0\n", "invalid_evidence"),
])
def test_native_failure_keeps_exit_code_and_partial_output(tmp_path, monkeypatch, exit_code,
                                                          termination, output, expected):
    monkeypatch.setattr(native, "compile_candidate", lambda *_: {
        "passed": True, "artifact_dir": str(tmp_path), "executable": str(tmp_path / "candidate.exe")})
    def execute(*_args, **_kwargs):
        (tmp_path / "observations.txt").write_text(output)
        return ProcessResult(exit_code, "", "", native_returncode=-9 if termination else exit_code,
                             termination=termination)
    monkeypatch.setattr(native, "run_bounded", execute)
    result = native.evaluate_candidate(tmp_path / "candidate.hpp", "anti_windup", tmp_path, "unused")
    assert result["failure"]["category"] == expected
    assert result["exit_code"] == exit_code
    assert result["native_exit_code"] == (-9 if termination else exit_code)
    assert result["output_lines"] == len(output.splitlines())
    assert result["error"]
    assert json.loads((tmp_path / "execution.json").read_text())["native_exit_code"] == result["native_exit_code"]


def test_real_process_exit_is_not_confused_with_supervisor_timeout():
    import sys
    natural = run_bounded([sys.executable, "-c", "raise SystemExit(124)"])
    timed = run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.1)
    assert natural.returncode == natural.native_returncode == 124
    assert natural.termination is None
    assert timed.returncode == 124 and timed.termination == "timeout"
    assert timed.native_returncode is not None


def test_seventeen_case_four_arm_protocol_and_heldout_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "compiler_identity", lambda _: {
        "passed": True, "compiler": "offline-stub", "version": "test"})
    monkeypatch.setattr(suite, "capability", lambda _, **__: {"passed": True, "stage": "capability"})
    monkeypatch.setattr(suite, "compile_source", lambda *_, **__: {"passed": True, "stage": "build"})
    calls = []
    def evaluate(source, case_id, artifacts, compiler, *, split, **_kwargs):
        calls.append((source, case_id, split))
        return {"passed": source.read_text(encoding="utf-8") == suite.reference_text(case_id),
                "stage": "correctness", "stdout": f"{split}-evidence-only",
                "inputs_sha256": suite.digest(suite.split_inputs(case_id, split))}
    monkeypatch.setattr(suite, "evaluate_source", evaluate)
    requests = []
    def worker(request, _directory):
        requests.append(request)
        return runner.run_attempt(request)
    monkeypatch.setattr(runner, "run_worker", worker)
    report = runner.run_matrix(output=tmp_path / "matrix", budget=runner.Budget(),
                               trusted_local=True, suite="professional", split="independent")
    assert report["status"] == "completed" and len(report["results"]) == 68
    assert all(row["success"] for row in report["results"])
    assert all(set(item["splits"]) == {"development", "heldout"} and item["valid"]
               for item in report["calibration"])
    assert len(requests) == 68 and all(r["split_policy"] == "independent" for r in requests)
    for row in report["results"]:
        trace = Path(row["trace"])
        text = (trace / "events.jsonl").read_text(encoding="utf-8")
        assert "heldout-evidence-only" not in text
        assert ("development-evidence-only" in text) == (row["arm"] == "feedback")
        assert inspect_trace(trace)["session_closed"]
        assert row["final_validation"]["inputs_sha256"] == suite.split_manifest(row["case_id"])["splits"]["heldout"]["sha256"]
    assert sum(split == "heldout" for _, _, split in calls) == 17 * 2 + 68


def test_missing_heavy_dependencies_block_before_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "compiler_identity", lambda _: {"passed": True, "compiler": "stub"})
    monkeypatch.setattr(suite, "capability", lambda _, **__: {
        "passed": False, "stage": "capability", "error": "pinned dependencies missing"})
    monkeypatch.setattr(runner, "run_worker", lambda *_: pytest.fail("Spent model budget"))
    report = runner.run_matrix(output=tmp_path / "blocked", budget=runner.Budget(), trusted_local=True,
                               suite="professional", split="independent", selected=["rnea_missing_velocity"])
    assert len(report["results"]) == 4
    assert all(row["failure"]["category"] == "environment" and not row["calibrated"] for row in report["results"])


def test_development_dense_report_cannot_be_relabelled_as_heldout(tmp_path):
    dev, held = suite.dense_spec("development"), suite.dense_spec("heldout")
    report = {"schema_version": 1, "passed": True, "seed": dev.seed,
              "random_samples": dev.random_samples, "sample_count": dev.sample_count,
              "thresholds": {key: getattr(dev, key) for key in
                             ("output_atol", "output_rtol", "jacobian_atol", "jacobian_rtol")},
              "metrics": {"failures": 0, "non_finite": 0, "dimension_failures": 0},
              "samples": suite.deterministic_samples(dev)}
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(report))
    assert suite.load_dense_report(path, dev)["passed"]
    with pytest.raises(ValueError, match="seed"):
        suite.load_dense_report(path, held)
    report["seed"] = held.seed
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="sample values"):
        suite.load_dense_report(path, held)


def test_rnea_development_observations_cannot_pass_heldout_contract():
    from agent.domain_packs.pinocchio_rnea.report import evaluate_correctness
    from agent.domain_packs.pinocchio_rnea.spec import model_contract
    dev = suite.split_inputs("rnea_missing_velocity", "development")
    # Synthetic matching backend values test the gate, not robot physics.
    records = [{"input": point,
                **{name: [1., 2.] for name in ("pinocchio_output", "source_output", "analytic_output",
                                              "cppad_output", "codegen_output")},
                **{name: [0.] * 12 for name in ("analytic_jacobian", "finite_difference_jacobian",
                                                "cppad_jacobian", "codegen_jacobian")}}
               for point in dev]
    raw = {"schema_version": 1, "nq": 2, "nv": 2,
           "urdf_sha256": model_contract()["urdf_sha256"], "samples": records}
    assert evaluate_correctness(raw, expected_inputs=dev)["passed"]
    with pytest.raises(ValueError, match="input/order"):
        evaluate_correctness(raw, expected_inputs=suite.split_inputs("rnea_missing_velocity", "heldout"))


@pytest.mark.parametrize("version", ["3.4.0", "9.9.9"])
def test_rnea_direct_build_uses_pinned_exports_and_fresh_candidate(tmp_path, monkeypatch, version):
    ad, pin = suite.dependency_prefixes(tmp_path / "deps")
    (ad / "lib").mkdir(parents=True)
    (ad / "lib/libcppad_lib.so").write_bytes(b"library")
    (pin / "lib/pkgconfig").mkdir(parents=True)
    (pin / "lib/pkgconfig/pinocchio.pc").write_text("Version: 3.4.0")
    monkeypatch.setattr(suite, "capability", lambda *_, **__: {"passed": True})
    builds = []
    def run(command, **kwargs):
        if command[0] == "pkg-config":
            return ProcessResult(0, version if "--modversion" in command else
                                 "-DPINOCCHIO_WITH_URDFDOM -I/include -L/lib -lpinocchio_parsers", "")
        assert command[0] == "test-clang++"
        assert "-DBOOST_MPL_LIMIT_LIST_SIZE=30" in command and "-DNDEBUG" in command
        assert "-DPINOCCHIO_WITH_URDFDOM" in command and "-lpinocchio_parsers" in command
        cwd = kwargs["cwd"]
        builds.append((cwd, (cwd / "candidate.hpp").read_text()))
        (cwd / "candidate.exe").write_bytes(b"compiled")
        return ProcessResult(0, "", "")
    monkeypatch.setattr(suite, "run_bounded", run)
    source = tmp_path / "candidate.hpp"
    for text in ("first candidate", "changed candidate"):
        source.write_text(text)
        result = suite.compile_source(source, "rnea_missing_velocity", tmp_path / "builds",
                                      "test-clang++", dependency_root=tmp_path / "deps")
        assert result["passed"] == (version == "3.4.0")
        if version != "3.4.0":
            assert result["failure"]["category"] == "environment"
    if version == "3.4.0":
        assert len(builds) == 2 and builds[0][0] != builds[1][0]
        assert [text for _, text in builds] == ["first candidate", "changed candidate"]
    else:
        assert not builds
