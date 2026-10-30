"""Trusted-local 17-case adapter with disjoint development/held-out inputs.

Held-out means separate input populations and operator-only final invocation,
not OS isolation: these public synthetic specifications are not secret oracles.
"""
from __future__ import annotations

import hashlib
import json
import random
import platform
import shutil
import shlex
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.ablation_native import compile_candidate, evaluate_candidate, execution_receipt
from agent.containers import run_bounded
from agent.domain_packs.cpp_robotics.spec import CASE_NAMES, samples
from agent.domain_packs.cpp_robotics.validators import PACK_ROOT as ROBOTICS
from agent.domain_packs.robotics_autodiff_codegen.dense_validation import (
    DenseValidationSpec, deterministic_samples, load_dense_report,
)
from agent.domain_packs.robotics_autodiff_codegen.sparse_validation import evaluate_sparse_report
from agent.failures import with_failure
from agent.workspace import HARNESS_ROOT

SPLIT_VERSION = "independent-inputs-v1"
SEEDS = {"development": 2026102901, "heldout": 2026102902}
PACKS = HARNESS_ROOT / "src/agent/domain_packs"
DENSE_CASES = {"autodiff_hardcoded_scalar", "autodiff_scalar_propagation",
               "autodiff_jacobian_input_order", "autodiff_random_blind_spot"}
SPARSE_CASES = {"sparse_unexpected_dependency", "sparse_missing_dependency"}
RNEA_CASES = {"rnea_acceleration_slice", "rnea_missing_velocity"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def family(case_id: str) -> str:
    if case_id in {"cpp_" + name for name in CASE_NAMES}:
        return "robotics"
    if case_id == "cpp_reference_output":
        return "reference"
    if case_id in DENSE_CASES:
        return "dense"
    if case_id in SPARSE_CASES:
        return "sparse"
    if case_id in RNEA_CASES:
        return "rnea"
    raise ValueError(f"Unsupported ablation case: {case_id}")


def dense_spec(split: str) -> DenseValidationSpec:
    return DenseValidationSpec(seed=SEEDS[split], include_fixed=False)


def split_inputs(case_id: str, split: str) -> list[list[float]]:
    kind = family(case_id)
    seed = SEEDS[split]
    if kind == "robotics":
        return samples(case_id.removeprefix("cpp_"), seed=seed, include_fixed=False)
    if kind in {"dense", "sparse"}:
        return [row["input"] for row in deterministic_samples(dense_spec(split))]  # type: ignore[misc]
    rng = random.Random(seed)
    bounds = (2.5, 2.5, 4., 4., 5., 5.) if kind == "rnea" else (3., 3., 2.)
    return [[rng.uniform(-bound, bound) for bound in bounds] for _ in range(24)]


def split_manifest(case_id: str) -> dict[str, Any]:
    populations = {name: split_inputs(case_id, name) for name in SEEDS}
    if set(map(tuple, populations["development"])) & set(map(tuple, populations["heldout"])):
        raise ValueError("Development and held-out inputs overlap")
    return {"version": SPLIT_VERSION, "case_id": case_id,
            "scope": "public_synthetic_disjoint_inputs_not_secure_hidden",
            "splits": {name: {"count": len(points), "sha256": digest(points)}
                       for name, points in populations.items()}}


def reference_text(case_id: str) -> str:
    kind = family(case_id)
    if kind == "robotics":
        path = ROBOTICS / "harness/reference.hpp"
    elif kind == "rnea":
        path = PACKS / "pinocchio_rnea/harness/reference.hpp"
    else:
        path = HARNESS_ROOT / "benchmarks/ablation_references" / f"{kind}.hpp"
    return path.read_text(encoding="utf-8")


def dependency_prefixes(root: str | Path | None = None) -> tuple[Path, Path]:
    directory = Path(root) if root is not None else HARNESS_ROOT / ".agent_state/domain_deps"
    return (directory / "autodiff_codegen/v1/install", directory / "pinocchio/v1/install")


def capability(case_id: str, *, dependency_root: str | Path | None = None) -> dict[str, Any]:
    kind = family(case_id)
    if kind in {"dense", "sparse", "rnea"}:
        from agent.domain.toolchain import inspect_toolchain
        from agent.domain_packs.robotics_autodiff_codegen.constants import CPPAD_COMMIT, CPPAD_CODEGEN_COMMIT
        from agent.domain_packs.pinocchio_rnea.dependencies import PINOCCHIO_COMMIT
        ad, pin = dependency_prefixes(dependency_root)
        missing = list(inspect_toolchain().missing)
        if platform.system() != "Linux":
            missing.append("linux_codegen_runtime")
        marker_hashes = {}
        required = [(ad, {"cppad_commit": CPPAD_COMMIT, "cppad_codegen_commit": CPPAD_CODEGEN_COMMIT},
                     ["cppad/cppad.hpp", "cppad/cg.hpp", "cppad/configure.hpp", "cppad/cg/configure.hpp"])]
        if kind == "rnea":
            required.append((pin, {"commit": PINOCCHIO_COMMIT},
                             ["pinocchio/multibody/model.hpp", "pinocchio/parsers/urdf.hpp", "pinocchio/codegen/cppadcg.hpp"]))
            if not any(ad.glob("lib*/libcppad_lib.*")):
                missing.append("cppad_runtime_library")
            pc = pin / "lib/pkgconfig/pinocchio.pc"
            if not shutil.which("pkg-config"):
                missing.append("pkg-config")
            if not pc.is_file():
                missing.append("pinocchio_pkg_config")
            else:
                marker_hashes[str(pc)] = hashlib.sha256(pc.read_bytes()).hexdigest()
        for prefix, commits, headers in required:
            marker = prefix.parent / "installed.json"
            try:
                info = json.loads(marker.read_text())
                if not isinstance(info, dict) or any(info.get(key) != value for key, value in commits.items()):
                    missing.append("pinned_commits:" + str(prefix))
                marker_hashes[str(marker)] = hashlib.sha256(marker.read_bytes()).hexdigest()
            except (OSError, ValueError):
                missing.append("install_marker:" + str(prefix))
            missing.extend("header:" + name for name in headers if not (prefix / "include" / name).is_file())
        deps = {"missing": missing, "marker_sha256": marker_hashes,
                "ad_prefix": str(ad), "pinocchio_prefix": str(pin) if kind == "rnea" else None}
        deps["fingerprint"] = digest(deps)
        return with_failure({"passed": not missing, "stage": "capability", "dependencies": deps,
                             "error": "Missing pinned dependencies: " + ", ".join(missing) if missing else None})
    return with_failure({"passed": True, "stage": "capability", "error": None})


def suite_identity(case_ids: list[str], *, dependency_root: str | Path | None = None) -> dict[str, Any]:
    # Freeze the adapter, oracle, harness and dependency contracts before any worker starts.
    paths = [Path(__file__), HARNESS_ROOT / "src/agent/ablation_native.py"]
    for pack in ("cpp_robotics", "robotics_autodiff_codegen", "pinocchio_rnea"):
        paths.extend(p for p in (PACKS / pack).rglob("*") if p.suffix in {".py", ".cpp", ".hpp", ".txt", ".json", ".urdf"})
    paths.extend((HARNESS_ROOT / "benchmarks/ablation_references").glob("*"))
    paths.append(HARNESS_ROOT / "benchmarks/ablation_harness/reference.cpp")
    return {"version": SPLIT_VERSION,
            "build_policy": "fresh harness -O0 -DNDEBUG; RNEA direct compiler with pinned pkg-config; generated-library flags unchanged; no performance gate",
            "files": {p.relative_to(HARNESS_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(set(paths))},
            "capabilities": {case_id: capability(case_id, dependency_root=dependency_root) for case_id in case_ids},
            "split_manifests": {case_id: split_manifest(case_id) for case_id in case_ids}}


def compile_source(source: Path, case_id: str, artifacts: Path, compiler: str,
                   *, dependency_root: str | Path | None = None) -> dict[str, Any]:
    kind = family(case_id)
    if kind == "robotics":
        return compile_candidate(source, case_id.removeprefix("cpp_"), artifacts, compiler)
    cap = capability(case_id, dependency_root=dependency_root)
    if not cap["passed"]:
        return cap
    directory = artifacts / uuid.uuid4().hex
    directory.mkdir(parents=True)
    shutil.copyfile(source, directory / "candidate.hpp")
    if kind == "reference":
        shutil.copyfile(HARNESS_ROOT / "benchmarks/ablation_harness/reference.cpp", directory / "probe.cpp")
        commands = [[compiler, "-std=c++17", "-O0", "probe.cpp", "-o", "candidate.exe"]]
        executable = directory / "candidate.exe"
    elif kind == "rnea":
        from agent.domain_packs.pinocchio_rnea.dependencies import PINOCCHIO_VERSION
        from agent.domain_packs.pinocchio_rnea.spec import MODEL_PATH, model_contract
        ad_prefix, pin_prefix = dependency_prefixes(dependency_root)
        pc = pin_prefix / "lib/pkgconfig/pinocchio.pc"
        configuration: dict[str, Any] = {}
        try:
            version = run_bounded(["pkg-config", "--modversion", str(pc)], timeout=30)
            flags = run_bounded(["pkg-config", "--cflags", "--libs", str(pc)], timeout=30)
            if version.returncode or version.stdout.strip() != PINOCCHIO_VERSION or flags.returncode:
                raise ValueError("Pinned Pinocchio pkg-config version/flags unavailable")
            exported = shlex.split(flags.stdout)
            cppad_library = next(p for p in sorted(ad_prefix.glob("lib*/libcppad_lib.so*")) if p.is_file())
            configuration = {"version": version.stdout.strip(), "flags": exported,
                             "pc_sha256": hashlib.sha256(pc.read_bytes()).hexdigest()}
        except (OSError, ValueError, StopIteration) as exc:
            payload = with_failure({"passed": False, "stage": "capability", "error": str(exc),
                                    "artifact_dir": str(directory)})
            (directory / "build.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
        shutil.copyfile(PACKS / "pinocchio_rnea/harness/main.cpp", directory / "probe.cpp")
        executable = directory / "candidate.exe"
        runtime_paths = [flag[2:] for flag in exported if flag.startswith("-L")]
        runtime_paths.append(str(cppad_library.parent))
        commands = [[compiler, "-std=c++17", "-O0", "-DNDEBUG", "-DCPPAD_DEBUG_AND_RELEASE",
                     "-DBOOST_MPL_LIMIT_LIST_SIZE=30", "-DBOOST_MPL_LIMIT_VECTOR_SIZE=30",
                     '-DWORKSPACE_HEADER="candidate.hpp"', f'-DROBOT_URDF="{MODEL_PATH.as_posix()}"',
                     f'-DURDF_SHA256="{model_contract()["urdf_sha256"]}"',
                     "-isystem", str(ad_prefix / "include"), "probe.cpp", "-o", "candidate.exe",
                     *exported, str(cppad_library), "-ldl",
                     *[arg for path in runtime_paths for arg in ("-Xlinker", "-rpath", "-Xlinker", path)]]]
        (directory / "pkg_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    else:
        from agent.domain.toolchain import inspect_toolchain
        ad_prefix, _ = dependency_prefixes(dependency_root)
        harness = PACKS / "robotics_autodiff_codegen/harness"
        target = "autodiff_codegen_validator"
        definitions = {"WORKSPACE_HEADER": (directory / "candidate.hpp").as_posix(),
                       "AUTODIFF_PREFIX": ad_prefix.as_posix()}
        spec = dense_spec("development")
        definitions.update({key.upper(): str(asdict(spec)[key]) for key in
                            ("output_atol", "output_rtol", "jacobian_atol", "jacobian_rtol")})
        definitions.update(SPARSE_MODEL_CONTRACT="1" if kind == "sparse" else "0",
                           CMAKE_CXX_FLAGS="-DABLATION_RANDOM_ONLY")
        cmake = inspect_toolchain().cmake_path
        assert cmake is not None
        commands = [[cmake, "-S", str(harness), "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG",
                     f"-DCMAKE_CXX_COMPILER={compiler}", *[f"-D{k}={v}" for k, v in definitions.items()]],
                    [cmake, "--build", "build", "--target", target, "--parallel", "1"]]
        executable = directory / "build" / target
    stages = []
    for index, command in enumerate(commands):
        start = time.perf_counter()
        try:
            result = run_bounded(command, cwd=directory, timeout=900 if kind == "rnea" else 240)
            stage = execution_receipt(result, time.perf_counter() - start, executable)
            stage["command"] = command
            stage["stage"] = "configure" if len(commands) > 1 and index == 0 else "build"
            stages.append(stage)
            if result.returncode:
                break
        except OSError as exc:
            stages.append({"stage": "capability", "exit_code": None, "error": str(exc)})
            break
    last = stages[-1]
    payload = with_failure({**last, "passed": last["exit_code"] == 0,
                            "stage": "capability" if last["exit_code"] is None else "build",
                            "error": None if last["exit_code"] == 0 else "C++ build failed",
                            "build_stages": stages, "executable": str(executable), "artifact_dir": str(directory)})
    (directory / "build.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def evaluate_source(source: Path, case_id: str, artifacts: Path, compiler: str,
                    *, split: str, dependency_root: str | Path | None = None) -> dict[str, Any]:
    kind = family(case_id)
    points = split_inputs(case_id, split)
    if kind == "robotics":
        robotics_result = evaluate_candidate(source, case_id.removeprefix("cpp_"), artifacts, compiler, inputs=points)
        return {**robotics_result, "split": split, "inputs_sha256": digest(points)}
    built = compile_source(source, case_id, artifacts, compiler, dependency_root=dependency_root)
    if not built["passed"]:
        return built
    directory = Path(built["artifact_dir"])
    # Generated libraries and observations are fresh for every invocation and split.
    runtime = directory / "runtime"
    runtime.mkdir()
    output = runtime / "observations.json"
    if kind in {"dense", "sparse"}:
        spec = dense_spec(split)
        args = ["--seed", str(spec.seed), "--random-samples", str(spec.random_samples), "--report", output.name]
    else:
        (runtime / "inputs.txt").write_text("\n".join(" ".join(format(x, ".17g") for x in point)
                                                         for point in points) + "\n", encoding="ascii")
        args = ["inputs.txt", output.name]
    execution: dict[str, Any] = {}
    try:
        start = time.perf_counter()
        result = run_bounded([built["executable"], *args], cwd=runtime, timeout=120)
        execution = execution_receipt(result, time.perf_counter() - start, output)
        (runtime / "execution.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
        # Dense harness returns 1 for numerical failures, but all other nonzero exits are runtime failures.
        if result.returncode and not (kind in {"dense", "sparse"} and result.returncode == 1 and output.is_file()):
            payload = {"passed": False, "stage": "runtime", "error": f"Native process exited with code {execution['native_exit_code']}"}
        elif kind in {"dense", "sparse"}:
            dense = load_dense_report(output, dense_spec(split))
            sparse = evaluate_sparse_report(dense, spec=dense_spec(split), contract="triangular" if kind == "sparse" else "coupled")
            passed = dense["passed"] and sparse["passed"] and result.returncode == 0
            payload = {"passed": passed, "stage": "complete" if passed else "derivative_correctness",
                       "sample_count": len(points), "first_failure": dense.get("worst_failure") or sparse.get("worst_failure"),
                       "error": None}
            if not passed:
                payload["stage"] = "dense_correctness" if not dense["passed"] else "sparse_correctness"
        elif kind == "rnea":
            from agent.domain_packs.pinocchio_rnea.report import evaluate_correctness
            payload = {**evaluate_correctness(json.loads(output.read_text()), expected_inputs=points),
                       "stage": "correctness", "error": None}
        else:
            observations = [list(map(float, line.split())) for line in output.read_text().splitlines()]
            if len(observations) != len(points) or any(len(row) != 2 for row in observations):
                raise ValueError("Incomplete or malformed C++ reference observations")
            import math
            failures = [{"sample": i, "input": x, "actual": y, "expected": [x[0]+x[2], x[1]-x[2]]}
                        for i, (x, y) in enumerate(zip(points, observations, strict=True))
                        if not all(math.isfinite(a) and abs(a-b) <= 1e-12
                                   for a, b in zip(y, [x[0]+x[2], x[1]-x[2]], strict=True))]
            payload = {"passed": not failures, "stage": "correctness", "error": None,
                       "sample_count": len(points), "first_failure": failures[0] if failures else None}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        payload = {"passed": False, "stage": "protocol", "error": f"{type(exc).__name__}: {exc}"}
    # Heavy harness stdout can contain the complete report; do not project it to the model.
    payload = with_failure({**payload, **{k: v for k, v in execution.items() if k != "stdout"},
                            "artifact_dir": str(directory), "split": split, "inputs_sha256": digest(points)})
    (directory / "validation.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
