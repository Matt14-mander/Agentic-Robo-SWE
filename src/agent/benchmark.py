"""M1 机器人代码任务评测框架。

核心流程是 ``重置题目 → 确认基线失败 → 运行 Agent → 独立验证 → 记录指标``。
题目和验证器均由仓库维护，Agent 只修改一次性 workspace，保证不同运行之间互不污染。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root

DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "cases.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
DEFAULT_TASK_TIMEOUT_SECONDS = 300.0
DEFAULT_TOOL_BUDGETS = {"easy": 5, "medium": 7, "hard": 9}
ProgressCallback = Callable[[dict[str, Any]], None]


class BenchmarkAttemptTimeout(TimeoutError):
    """Raised when an isolated Agent attempt exceeds its wall-clock budget."""


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    title: str
    category: str
    difficulty: str
    template: str
    workspace: str
    prompt: str
    validator: str
    tags: tuple[str, ...] = ()
    domain_pack: str | None = None
    required_capabilities: tuple[str, ...] = ()
    validator_timeout: int = 30
    artifact_policy: str | None = None
    execution_environment: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkCase":
        required = {
            "id", "title", "category", "difficulty", "template",
            "workspace", "prompt", "validator",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"Benchmark case missing fields: {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            category=str(data["category"]),
            difficulty=str(data["difficulty"]),
            template=str(data["template"]),
            workspace=str(data["workspace"]),
            prompt=str(data["prompt"]),
            validator=str(data["validator"]),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            domain_pack=str(data["domain_pack"]) if data.get("domain_pack") else None,
            required_capabilities=tuple(
                str(item) for item in data.get("required_capabilities", [])
            ),
            validator_timeout=int(data.get("validator_timeout", 30)),
            artifact_policy=(
                str(data["artifact_policy"]) if data.get("artifact_policy") else None
            ),
            execution_environment=(
                str(data["execution_environment"])
                if data.get("execution_environment")
                else None
            ),
        )


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    details: list[str]
    duration_seconds: float
    error: str | None = None


@dataclass
class BenchmarkResult:
    case_id: str
    title: str
    category: str
    difficulty: str
    repeat_index: int
    strategy: str
    success: bool
    baseline_passed: bool
    validation_details: list[str]
    duration_seconds: float
    validation_seconds: float
    loop_steps: int
    tool_calls: int
    tool_budget: int
    within_tool_budget: bool
    tool_calls_by_name: dict[str, int]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    workspace_modified: bool
    workspace: str
    provider: str
    executor_backend: str
    domain_pack: str | None
    domain_pack_version: str | None
    capability_check: dict[str, bool]
    toolchain_fingerprint: str | None
    final_suggestion: str | None
    attempts: int
    repair_attempts_used: int
    official_validator_called: bool
    official_validator_calls: int
    official_validator_attempts: int
    validator_compliance_rate: float
    claimed_success: bool
    false_positive: bool
    timed_out: bool
    timeout_attempts: int
    attempt_history: list[dict[str, Any]]
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(manifest: str | Path = DEFAULT_MANIFEST) -> list[BenchmarkCase]:
    path = Path(manifest)
    if not path.is_absolute():
        path = resolve_within_root(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in {1, 2}:
        raise ValueError(f"Unsupported benchmark schema_version: {data.get('schema_version')!r}")
    cases = [BenchmarkCase.from_dict(item) for item in data.get("cases", [])]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark case ids must be unique")
    if not cases:
        raise ValueError("Benchmark manifest contains no cases")
    for case in cases:
        resolve_within_root(case.template)
        resolve_within_root(case.workspace)
        if case.validator_timeout < 1:
            raise ValueError(f"Validator timeout must be positive: {case.id}")
        if case.domain_pack:
            from agent.domain.registry import get_domain_registry

            get_domain_registry().get(case.domain_pack)
    return cases


def select_cases(cases: Iterable[BenchmarkCase], selected_ids: Iterable[str]) -> list[BenchmarkCase]:
    cases = list(cases)
    selected = list(selected_ids)
    if not selected:
        return cases
    by_id = {case.id: case for case in cases}
    unknown = sorted(set(selected) - by_id.keys())
    if unknown:
        raise ValueError(f"Unknown benchmark case(s): {', '.join(unknown)}")
    return [by_id[case_id] for case_id in selected]


def prepare_workspace(case: BenchmarkCase) -> Path:
    template = resolve_within_root(case.template)
    workspace = resolve_within_root(case.workspace)
    if not template.is_file():
        raise FileNotFoundError(f"Benchmark template not found: {case.template}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, workspace)
    return workspace


def validate_case(case: BenchmarkCase, timeout: int | None = None) -> ValidationResult:
    started = time.perf_counter()
    command = [
        sys.executable,
        "-m",
        "benchmarks.validators",
        case.validator,
        case.workspace,
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or case.validator_timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ValidationResult(
            passed=False,
            details=[],
            duration_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        payload = json.loads(proc.stdout.strip())
        details = [str(item) for item in payload.get("details", [])]
        error = payload.get("error")
        passed = bool(payload.get("passed")) and proc.returncode == 0
    except (json.JSONDecodeError, AttributeError) as exc:
        details = [proc.stdout.strip()] if proc.stdout.strip() else []
        error = f"Invalid validator output: {exc}; stderr={proc.stderr.strip()}"
        passed = False

    return ValidationResult(
        passed=passed,
        details=details,
        duration_seconds=time.perf_counter() - started,
        error=str(error) if error else None,
    )


def build_task(case: BenchmarkCase, tool_budget: int | None = None) -> str:
    validation_code = (
        "from benchmarks.validators import validate\n"
        f"print(validate({case.validator!r}, {case.workspace!r}))"
    )
    budget_line = f"本任务工具调用预算为 {tool_budget} 次。\n" if tool_budget is not None else ""
    domain_line = f"领域能力包：{case.domain_pack}\n" if case.domain_pack else ""
    return (
        f"机器人代码评测任务 [{case.id}]：{case.title}\n\n"
        f"{domain_line}"
        f"目标文件：{case.workspace}\n"
        f"问题描述：{case.prompt}\n\n"
        "只修改目标文件。直接读取目标并实施最小修复，不要单独编写复现脚本。修复后必须使用 execute_python "
        "运行下面的验证代码；只有返回 passed=True 才算完成：\n\n"
        f"```python\n{validation_code}\n```\n\n"
        f"{budget_line}"
        "最后给出问题、修改和验证证据的简短总结。"
    )


def build_repair_task(case: BenchmarkCase, validation: ValidationResult, attempt: int) -> str:
    """Create deterministic feedback for a follow-up repair attempt."""
    validation_code = (
        "from benchmarks.validators import validate\n"
        f"print(validate({case.validator!r}, {case.workspace!r}))"
    )
    evidence = "; ".join(validation.details) or validation.error or "validator returned passed=False"
    return (
        f"Benchmark repair attempt {attempt} for [{case.id}].\n\n"
        f"The previous attempt failed the official validator: {evidence}\n"
        f"The existing workspace is {case.workspace}; do not reset it. Read it directly, fix the "
        "remaining defect, and run this exact validator with execute_python:\n\n"
        f"```python\n{validation_code}\n```\n\n"
        "Only report completion after it returns passed=True."
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _message_metrics(
    messages: Iterable[Any], case: BenchmarkCase | None = None
) -> tuple[int, dict[str, int], int, int, int, int]:
    tool_names: Counter[str] = Counter()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    official_validator_calls = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name", "unknown") if isinstance(call, dict) else "unknown"
            tool_names[str(name)] += 1
            if isinstance(call, dict) and name == "execute_python" and case is not None:
                args = call.get("args") or {}
                code = args.get("code", "") if isinstance(args, dict) else ""
                if "benchmarks.validators" in str(code) and case.validator in str(code):
                    official_validator_calls += 1

        usage = getattr(message, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return (
        sum(tool_names.values()),
        dict(sorted(tool_names.items())),
        input_tokens,
        output_tokens,
        total_tokens,
        official_validator_calls,
    )


def _claims_success(suggestion: str | None) -> bool:
    if not suggestion:
        return False
    normalized = suggestion.casefold()
    markers = (
        "验证通过", "修复完成", "所有检查通过", "已完成修复",
        "verified", "all checks passed", "fix is complete", "successfully fixed",
    )
    return any(marker in normalized for marker in markers)


def _emit_progress(callback: ProgressCallback | None, event: str, **payload: Any) -> None:
    if callback is not None:
        callback({"event": event, **payload})


def _invoke_graph_subprocess(
    state: dict[str, Any],
    config: dict[str, Any],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    """Run one graph attempt in a child process that can be killed on Windows."""
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".attempt-", dir=DEFAULT_RESULTS_DIR) as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "request.json"
        output_path = temp_path / "result.pkl"
        input_path.write_text(
            json.dumps({"state": state, "config": config}, ensure_ascii=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "agent.benchmark_worker",
            str(input_path),
            str(output_path),
        ]
        try:
            process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_label = f"{timeout_seconds:g}s" if timeout_seconds is not None else "limit"
            raise BenchmarkAttemptTimeout(
                f"Agent attempt exceeded {timeout_label} and was terminated"
            ) from exc

        if not output_path.is_file():
            stderr = process.stderr.strip()
            raise RuntimeError(
                f"Benchmark worker exited with code {process.returncode}"
                + (f": {stderr}" if stderr else " without a result")
            )
        status, payload = pickle.loads(output_path.read_bytes())
        if status == "error":
            raise RuntimeError(str(payload))
        if not isinstance(payload, dict):
            raise TypeError(f"Benchmark worker returned {type(payload).__name__}, expected dict")
        return payload


def run_case(
    case: BenchmarkCase,
    *,
    max_loop_steps: int = 12,
    repair_attempts: int = 0,
    strategy: str = "focused",
    repeat_index: int = 1,
    tool_budget: int | None = None,
    task_timeout_seconds: float | None = DEFAULT_TASK_TIMEOUT_SECONDS,
    progress: ProgressCallback | None = None,
    invoke: Callable[..., dict[str, Any]] | None = None,
) -> BenchmarkResult:
    if strategy not in {"focused", "general"}:
        raise ValueError(f"Unknown benchmark strategy: {strategy}")
    if repair_attempts < 0:
        raise ValueError("repair_attempts must be non-negative")
    if task_timeout_seconds is not None and task_timeout_seconds <= 0:
        raise ValueError("task_timeout_seconds must be positive or None")
    resolved_tool_budget = tool_budget or DEFAULT_TOOL_BUDGETS.get(case.difficulty, 7)
    if resolved_tool_budget < 1:
        raise ValueError("tool_budget must be positive or None")

    workspace = prepare_workspace(case)
    original_digest = _file_digest(workspace)
    baseline = validate_case(case)
    started = time.perf_counter()
    validation = baseline
    attempt_history: list[dict[str, Any]] = []
    aggregate_tools: Counter[str] = Counter()
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_loop_steps = 0
    official_validator_calls = 0
    official_validator_attempts = 0
    suggestion: str | None = None
    errors: list[str] = []
    task = build_task(case, resolved_tool_budget)
    domain_version, capability_check, toolchain_fingerprint = domain_case_metadata(case)

    for attempt in range(1, repair_attempts + 2):
        _emit_progress(
            progress,
            "attempt_start",
            case_id=case.id,
            repeat_index=repeat_index,
            attempt=attempt,
            max_attempts=repair_attempts + 1,
            timeout_seconds=task_timeout_seconds,
        )
        attempt_started = time.perf_counter()
        result: dict[str, Any] = {}
        run_error: str | None = None
        timed_out = False
        state = {
            "task": task,
            "loop_step": 0,
            "max_loop_steps": max_loop_steps,
            "benchmark_mode": True,
            "benchmark_strategy": strategy,
            "benchmark_validator": case.validator,
            "benchmark_workspace": case.workspace,
            "benchmark_validator_calls": 0,
            "benchmark_validation_passed": False,
            "benchmark_tool_budget": resolved_tool_budget,
            "benchmark_context_rounds": 3,
            "domain_packs": (case.domain_pack,) if case.domain_pack else (),
            "domain_capabilities": capability_check,
            "domain_toolchain_fingerprint": toolchain_fingerprint,
        }
        config = {"recursion_limit": max(50, max_loop_steps * 4)}
        try:
            if invoke is None:
                result = _invoke_graph_subprocess(state, config, task_timeout_seconds)
            else:
                result = invoke(state, config=config)
        except BenchmarkAttemptTimeout as exc:
            timed_out = True
            run_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # One failed attempt must not abort the suite.
            run_error = f"{type(exc).__name__}: {exc}"

        validation = validate_case(case)
        messages = result.get("messages") or []
        metrics = _message_metrics(messages, case)
        attempt_tools, by_name, in_tokens, out_tokens, tokens, validator_calls = metrics
        gate_validator_calls = int(result.get("benchmark_validator_calls") or 0)
        if gate_validator_calls:
            validator_calls += gate_validator_calls
            attempt_tools += gate_validator_calls
            by_name["execute_python"] = by_name.get("execute_python", 0) + gate_validator_calls
        aggregate_tools.update(by_name)
        total_input_tokens += in_tokens
        total_output_tokens += out_tokens
        total_tokens += tokens
        total_loop_steps += int(result.get("loop_step") or 0)
        official_validator_calls += validator_calls
        official_validator_attempts += int(validator_calls > 0)

        raw_suggestion = result.get("suggestion")
        suggestion = str(raw_suggestion) if raw_suggestion is not None else None
        claimed_success = _claims_success(suggestion)
        attempt_history.append({
            "attempt": attempt,
            "is_repair": attempt > 1,
            "validation_passed": validation.passed,
            "validation_details": validation.details,
            "validation_error": validation.error,
            "loop_steps": int(result.get("loop_step") or 0),
            "tool_calls": attempt_tools,
            "tool_calls_by_name": by_name,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": tokens,
            "official_validator_called": validator_calls > 0,
            "official_validator_calls": validator_calls,
            "validator_gate_calls": gate_validator_calls,
            "claimed_success": claimed_success,
            "false_positive": claimed_success and not validation.passed,
            "timed_out": timed_out,
            "duration_seconds": round(time.perf_counter() - attempt_started, 4),
            "error": run_error,
        })
        _emit_progress(
            progress,
            "attempt_complete",
            case_id=case.id,
            repeat_index=repeat_index,
            attempt=attempt,
            passed=validation.passed,
            timed_out=timed_out,
            duration_seconds=attempt_history[-1]["duration_seconds"],
            error=run_error or validation.error,
        )
        if run_error:
            errors.append(f"attempt {attempt}: {run_error}")
        if validation.passed:
            break
        if attempt <= repair_attempts:
            task = build_repair_task(case, validation, attempt + 1)

    duration = time.perf_counter() - started
    if validation.error:
        errors.append(validation.error)
    if baseline.passed:
        errors.append("Invalid benchmark fixture: baseline already passes")

    attempts = len(attempt_history)

    return BenchmarkResult(
        case_id=case.id,
        title=case.title,
        category=case.category,
        difficulty=case.difficulty,
        repeat_index=repeat_index,
        strategy=strategy,
        success=validation.passed and not baseline.passed,
        baseline_passed=baseline.passed,
        validation_details=validation.details,
        duration_seconds=round(duration, 4),
        validation_seconds=round(validation.duration_seconds, 4),
        loop_steps=total_loop_steps,
        tool_calls=sum(aggregate_tools.values()),
        tool_budget=resolved_tool_budget,
        within_tool_budget=sum(aggregate_tools.values()) <= resolved_tool_budget,
        tool_calls_by_name=dict(sorted(aggregate_tools.items())),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        workspace_modified=_file_digest(workspace) != original_digest,
        workspace=relpath_for_display(workspace),
        provider=os.getenv("MODEL_PROVIDER", "anthropic"),
        executor_backend=os.getenv("EXECUTOR_BACKEND", "local"),
        domain_pack=case.domain_pack,
        domain_pack_version=domain_version,
        capability_check=capability_check,
        toolchain_fingerprint=toolchain_fingerprint,
        final_suggestion=suggestion,
        attempts=attempts,
        repair_attempts_used=max(0, attempts - 1),
        official_validator_called=official_validator_calls > 0,
        official_validator_calls=official_validator_calls,
        official_validator_attempts=official_validator_attempts,
        validator_compliance_rate=round(official_validator_attempts / attempts, 4),
        claimed_success=any(item["claimed_success"] for item in attempt_history),
        false_positive=any(item["false_positive"] for item in attempt_history),
        timed_out=any(item["timed_out"] for item in attempt_history),
        timeout_attempts=sum(item["timed_out"] for item in attempt_history),
        attempt_history=attempt_history,
        error="; ".join(errors) if errors else None,
    )


def domain_case_metadata(
    case: BenchmarkCase,
) -> tuple[str | None, dict[str, bool], str | None]:
    if not case.domain_pack:
        return None, {}, None
    from agent.domain.registry import get_domain_registry

    pack = get_domain_registry().get(case.domain_pack)
    payload = dict(pack.capability_probe()) if pack.capability_probe else {}
    missing_value = payload.get("missing", ())
    missing = (
        {str(item) for item in missing_value}
        if isinstance(missing_value, (list, tuple, set))
        else set()
    )
    capabilities = {
        capability: capability not in missing for capability in case.required_capabilities
    }
    if case.execution_environment:
        capabilities["execution_environment"] = (
            os.getenv("EXECUTOR_BACKEND", "local") == case.execution_environment
        )
    fingerprint = payload.get("fingerprint")
    return pack.version, capabilities, str(fingerprint) if fingerprint else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without exposing a partially-written checkpoint to readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _build_report(
    *,
    run_id: str,
    created_at: str,
    cases: list[BenchmarkCase],
    results: list[BenchmarkResult],
    max_loop_steps: int,
    repair_attempts: int,
    repeats: int,
    strategy: str,
    tool_budget: int | None,
    task_timeout_seconds: float | None,
    status: str,
    fatal_error: str | None = None,
) -> dict[str, Any]:
    planned_runs = len(cases) * repeats
    passed = sum(result.success for result in results)
    false_positives = sum(result.false_positive for result in results)
    total_attempts = sum(result.attempts for result in results)
    compliant_attempts = sum(result.official_validator_attempts for result in results)
    per_case: dict[str, Any] = {}
    for case in cases:
        case_results = [result for result in results if result.case_id == case.id]
        if not case_results:
            continue
        case_passed = sum(result.success for result in case_results)
        case_attempts = sum(result.attempts for result in case_results)
        per_case[case.id] = {
            "domain_pack": case.domain_pack,
            "required_capabilities": list(case.required_capabilities),
            "runs": len(case_results),
            "passed": case_passed,
            "success_rate": round(case_passed / len(case_results), 4),
            "average_tool_calls": round(
                sum(item.tool_calls for item in case_results) / len(case_results), 2
            ),
            "average_tokens": round(
                sum(item.total_tokens for item in case_results) / len(case_results), 2
            ),
            "average_duration_seconds": round(
                sum(item.duration_seconds for item in case_results) / len(case_results), 4
            ),
            "false_positives": sum(item.false_positive for item in case_results),
            "timeouts": sum(item.timed_out for item in case_results),
            "within_tool_budget": sum(item.within_tool_budget for item in case_results),
            "validator_compliance_rate": round(
                sum(item.official_validator_attempts for item in case_results) / case_attempts, 4
            ),
        }

    completed_runs = len(results)
    return {
        "schema_version": 4,
        "run_id": run_id,
        "status": status,
        "created_at": created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "provider": os.getenv("MODEL_PROVIDER", "anthropic"),
        "executor_backend": os.getenv("EXECUTOR_BACKEND", "local"),
        "domain_packs": sorted({case.domain_pack for case in cases if case.domain_pack}),
        "max_loop_steps": max_loop_steps,
        "repair_attempts": repair_attempts,
        "repeats": repeats,
        "strategy": strategy,
        "tool_budget_override": tool_budget,
        "task_timeout_seconds": task_timeout_seconds,
        "case_count": planned_runs,
        "planned_runs": planned_runs,
        "completed_runs": completed_runs,
        "remaining_runs": planned_runs - completed_runs,
        "passed": passed,
        "failed": completed_runs - passed,
        "success_rate": round(passed / completed_runs, 4) if completed_runs else 0.0,
        "total_duration_seconds": round(sum(item.duration_seconds for item in results), 4),
        "total_tool_calls": sum(item.tool_calls for item in results),
        "total_tokens": sum(item.total_tokens for item in results),
        "average_duration_seconds": round(
            sum(item.duration_seconds for item in results) / completed_runs, 4
        ) if completed_runs else 0.0,
        "average_tool_calls": round(
            sum(item.tool_calls for item in results) / completed_runs, 2
        ) if completed_runs else 0.0,
        "average_tokens": round(
            sum(item.total_tokens for item in results) / completed_runs, 2
        ) if completed_runs else 0.0,
        "official_validator_compliance_rate": round(
            compliant_attempts / total_attempts, 4
        ) if total_attempts else 0.0,
        "false_positives": false_positives,
        "false_positive_rate": round(false_positives / completed_runs, 4)
        if completed_runs else 0.0,
        "timeouts": sum(item.timed_out for item in results),
        "timeout_attempts": sum(item.timeout_attempts for item in results),
        "within_tool_budget": sum(item.within_tool_budget for item in results),
        "tool_budget_compliance_rate": round(
            sum(item.within_tool_budget for item in results) / completed_runs, 4
        ) if completed_runs else 0.0,
        "fatal_error": fatal_error,
        "per_case": per_case,
        "results": [result.to_dict() for result in results],
    }


def run_suite(
    cases: Iterable[BenchmarkCase],
    *,
    max_loop_steps: int = 12,
    repair_attempts: int = 0,
    repeats: int = 1,
    strategy: str = "focused",
    tool_budget: int | None = None,
    task_timeout_seconds: float | None = DEFAULT_TASK_TIMEOUT_SECONDS,
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    invoke: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    cases = list(cases)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_dir) if output_dir else DEFAULT_RESULTS_DIR / run_id
    destination = resolve_within_root(str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    case_results_dir = destination / "cases"
    case_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "checkpoint.json"
    report_path = destination / "report.json"
    created_at = datetime.now(UTC).isoformat()
    planned_runs = len(cases) * repeats
    results: list[BenchmarkResult] = []

    def make_report(status: str, fatal_error: str | None = None) -> dict[str, Any]:
        return _build_report(
            run_id=run_id,
            created_at=created_at,
            cases=cases,
            results=results,
            max_loop_steps=max_loop_steps,
            repair_attempts=repair_attempts,
            repeats=repeats,
            strategy=strategy,
            tool_budget=tool_budget,
            task_timeout_seconds=task_timeout_seconds,
            status=status,
            fatal_error=fatal_error,
        )

    _write_json_atomic(checkpoint_path, make_report("running"))
    _emit_progress(
        progress,
        "suite_start",
        run_id=run_id,
        planned_runs=planned_runs,
        output_dir=str(destination),
        checkpoint=str(checkpoint_path),
    )
    try:
        for repeat_index in range(1, repeats + 1):
            for case in cases:
                position = len(results) + 1
                _emit_progress(
                    progress,
                    "case_start",
                    case_id=case.id,
                    repeat_index=repeat_index,
                    repeats=repeats,
                    position=position,
                    planned_runs=planned_runs,
                )
                result = run_case(
                    case,
                    max_loop_steps=max_loop_steps,
                    repair_attempts=repair_attempts,
                    strategy=strategy,
                    repeat_index=repeat_index,
                    tool_budget=tool_budget,
                    task_timeout_seconds=task_timeout_seconds,
                    progress=progress,
                    invoke=invoke,
                )
                results.append(result)
                case_path = case_results_dir / f"repeat-{repeat_index:02d}__{case.id}.json"
                _write_json_atomic(case_path, result.to_dict())
                _write_json_atomic(checkpoint_path, make_report("running"))
                _emit_progress(
                    progress,
                    "case_complete",
                    case_id=case.id,
                    repeat_index=repeat_index,
                    position=position,
                    planned_runs=planned_runs,
                    passed=result.success,
                    timed_out=result.timed_out,
                    duration_seconds=result.duration_seconds,
                    tool_calls=result.tool_calls,
                    tool_budget=result.tool_budget,
                    within_tool_budget=result.within_tool_budget,
                    total_tokens=result.total_tokens,
                    result_path=str(case_path),
                    checkpoint=str(checkpoint_path),
                )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(checkpoint_path, make_report("interrupted", error))
        _emit_progress(
            progress,
            "suite_interrupted",
            completed_runs=len(results),
            planned_runs=planned_runs,
            error=error,
            checkpoint=str(checkpoint_path),
        )
        raise

    report = make_report("completed")
    _write_json_atomic(checkpoint_path, report)
    _write_json_atomic(report_path, report)
    _emit_progress(
        progress,
        "suite_complete",
        completed_runs=len(results),
        planned_runs=planned_runs,
        report=str(report_path),
    )
    return report, report_path
