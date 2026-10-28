"""Real CMake/clangd and separate-image tests. No LLM API calls."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from agent.cpp import configure_build, semantic_query
from agent.containers import run_container
from agent.evaluation import evaluate_repository, receipt_matches
from agent.tools import execute_python
from agent.workspace import HARNESS_ROOT, Workspace, use_workspace


@pytest.fixture
def external_repo(tmp_path):
    root = tmp_path / "candidate"
    shutil.copytree(HARNESS_ROOT / "examples" / "external_cpp", root)
    return root


@pytest.mark.cpp
def test_real_multifile_cpp_build_and_semantics(external_repo, tmp_path):
    image = os.getenv("ROBO_TEST_RUNNER_IMAGE")
    if not image and not all(shutil.which(name) for name in ("clangd", "cmake", "ninja", "g++")):
        pytest.skip("Requires native clangd/CMake/Ninja/g++ or ROBO_TEST_RUNNER_IMAGE")
    workspace = Workspace.create(external_repo, tmp_path / "artifacts",
                                 trusted_local=not bool(image), execution_image=image)
    with use_workspace(workspace):
        built = configure_build("controller")
        assert built["passed"], built
        text = (external_repo / "main.cpp").read_text()
        lines = text.splitlines()
        line = next(i for i, value in enumerate(lines, 1) if "joint_error(target" in value)
        column = lines[line - 1].index("joint_error") + 1
        definition = semantic_query("main.cpp", "definition", line, column)
        assert "controller.cpp" in json.dumps(definition["result"])
        hover = semantic_query("main.cpp", "hover", line, column)
        assert "double" in json.dumps(hover["result"])
        references = semantic_query("main.cpp", "references", line, column)
        assert "main.cpp" in json.dumps(references["result"])
        assert "controller.cpp" in json.dumps(references["result"])
        (external_repo / "controller.cpp").write_text("#include \"controller.hpp\"\nthis is invalid C++;\n")
        assert not configure_build("controller")["passed"]
        diagnostics = semantic_query("controller.cpp", "diagnostics", 2, 1)
        assert any(item.get("severity") == 1 for item in diagnostics["result"])


@pytest.fixture
def private_spec(tmp_path):
    runner, oracle = os.getenv("ROBO_TEST_RUNNER_IMAGE"), os.getenv("ROBO_TEST_ORACLE_IMAGE")
    if not runner or not oracle:
        pytest.skip("Requires two built isolation images; see Phase 7.0 instructions")
    spec = tmp_path / "private-spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "task_id": "public-joint-wrap-demo",
                               "runner_image": runner, "oracle_image": oracle,
                               "runner_command": ["python3", "/opt/robo/runner.py"],
                               "oracle_command": ["python3", "/opt/robo/oracle.py"]}))
    return spec


@pytest.mark.isolation
def test_real_isolated_broken_reference_and_stale_receipt(external_repo, private_spec, tmp_path):
    output = tmp_path / "private-results"
    broken = evaluate_repository(external_repo, private_spec, output)
    assert broken["error"] is None and not broken["passed"], broken
    source = external_repo / "controller.cpp"
    source.write_text(source.read_text().replace("return target - actual;",
                                               "return std::remainder(target - actual, 2.0 * std::acos(-1.0));"))
    reference = evaluate_repository(external_repo, private_spec, output)
    assert reference["passed"], reference
    assert receipt_matches(reference, external_repo, private_spec)
    source.write_text(source.read_text() + "\n// later edit\n")
    assert not receipt_matches(reference, external_repo, private_spec)


@pytest.mark.isolation
def test_real_container_cannot_read_host_or_oracle(external_repo, private_spec, tmp_path, monkeypatch):
    image = json.loads(private_spec.read_text())["runner_image"]
    (external_repo / ".env").write_text("candidate-secret")
    monkeypatch.setenv("ROBO_SECRET_SENTINEL", "host-secret")
    workspace = Workspace.create(external_repo, tmp_path / "artifacts", execution_image=image)
    code = '''import os,json,socket
from pathlib import Path
blocked = False
try: Path('/workspace/main.cpp').write_text('overwrite')
except OSError: blocked = True
print(json.dumps({'oracle':Path('/opt/robo/oracle.py').exists(),
'socket':Path('/var/run/docker.sock').exists(), 'secret':os.getenv('ROBO_SECRET_SENTINEL'),
'env_file':Path('/workspace/.env').exists(), 'write_blocked':blocked}))
'''
    with use_workspace(workspace):
        output = execute_python.invoke({"code": code})
    assert "exit_code: 0" in output, output
    result = json.loads(output.split("--- stdout ---\n")[1])
    assert result == {"oracle": False, "socket": False, "secret": None,
                      "env_file": False, "write_blocked": True}


@pytest.mark.isolation
def test_real_candidate_cannot_forge_verdict(external_repo, private_spec, tmp_path):
    spec = json.loads(private_spec.read_text())
    spec["runner_command"] = ["python3", "-c", "print('{\"passed\": true}')"]
    private_spec.write_text(json.dumps(spec))
    report = evaluate_repository(external_repo, private_spec, tmp_path / "private-results")
    assert not report["passed"] and report["error"] and report["stage"] == "candidate"


@pytest.mark.isolation
def test_real_container_timeout_is_cleaned_up(private_spec, monkeypatch):
    import subprocess
    from agent import containers
    image = json.loads(private_spec.read_text())["runner_image"]
    removed = []
    original = containers.remove_container

    def cleanup(name):
        original(name)
        removed.append(name)

    monkeypatch.setattr(containers, "remove_container", cleanup)
    result = run_container(image, ["python3", "-c", "import time;time.sleep(30)"], [], timeout=2)
    assert result.returncode == 124 and removed
    check = subprocess.run(["docker", "inspect", removed[-1]], capture_output=True)
    assert check.returncode != 0
