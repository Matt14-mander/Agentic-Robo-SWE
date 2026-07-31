"""M1 机器人代码任务评测框架。

核心流程是 ``重置题目 → 确认基线失败 → 运行 Agent → 独立验证 → 记录指标``。
题目和验证器均由仓库维护，Agent 只修改一次性 workspace，保证不同运行之间互不污染。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from agent.tools._paths import PROJECT_ROOT, relpath_for_display, resolve_within_root

DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "cases.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


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
    success: bool
    baseline_passed: bool
    validation_details: list[str]
    duration_seconds: float
    validation_seconds: float
    loop_steps: int
    tool_calls: int
    tool_calls_by_name: dict[str, int]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    workspace_modified: bool
    workspace: str
    provider: str
    executor_backend: str
    final_suggestion: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(manifest: str | Path = DEFAULT_MANIFEST) -> list[BenchmarkCase]:
    path = Path(manifest)
    if not path.is_absolute():
        path = resolve_within_root(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
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


def validate_case(case: BenchmarkCase, timeout: int = 30) -> ValidationResult:
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
            timeout=timeout,
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


def build_task(case: BenchmarkCase) -> str:
    validation_code = (
        "from benchmarks.validators import validate\n"
        f"print(validate({case.validator!r}, {case.workspace!r}))"
    )
    return (
        f"机器人代码评测任务 [{case.id}]：{case.title}\n\n"
        f"目标文件：{case.workspace}\n"
        f"问题描述：{case.prompt}\n\n"
        "只修改目标文件。先复现问题，再实施最小修复。修复后必须使用 execute_python "
        "运行下面的验证代码；只有返回 passed=True 才算完成：\n\n"
        f"```python\n{validation_code}\n```\n\n"
        "最后给出问题、修改和验证证据的简短总结。"
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _message_metrics(messages: Iterable[Any]) -> tuple[int, dict[str, int], int, int, int]:
    tool_names: Counter[str] = Counter()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name", "unknown") if isinstance(call, dict) else "unknown"
            tool_names[str(name)] += 1

        usage = getattr(message, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return sum(tool_names.values()), dict(sorted(tool_names.items())), input_tokens, output_tokens, total_tokens


def run_case(
    case: BenchmarkCase,
    *,
    max_loop_steps: int = 12,
    invoke: Callable[..., dict[str, Any]] | None = None,
) -> BenchmarkResult:
    workspace = prepare_workspace(case)
    original_digest = _file_digest(workspace)
    baseline = validate_case(case)
    started = time.perf_counter()
    result: dict[str, Any] = {}
    run_error: str | None = None

    try:
        if invoke is None:
            from agent.graph import graph

            invoke = graph.invoke
        result = invoke(
            {
                "task": build_task(case),
                "loop_step": 0,
                "max_loop_steps": max_loop_steps,
            },
            config={"recursion_limit": max(50, max_loop_steps * 4)},
        )
    except Exception as exc:  # 评测单题失败不能中断整套 benchmark
        run_error = f"{type(exc).__name__}: {exc}"

    duration = time.perf_counter() - started
    validation = validate_case(case)
    messages = result.get("messages") or []
    tool_calls, by_name, input_tokens, output_tokens, total_tokens = _message_metrics(messages)
    suggestion = result.get("suggestion")
    if suggestion is not None and not isinstance(suggestion, str):
        suggestion = str(suggestion)

    errors = [error for error in (run_error, validation.error) if error]
    if baseline.passed:
        errors.append("Invalid benchmark fixture: baseline already passes")

    return BenchmarkResult(
        case_id=case.id,
        title=case.title,
        category=case.category,
        difficulty=case.difficulty,
        success=validation.passed and not baseline.passed and not run_error,
        baseline_passed=baseline.passed,
        validation_details=validation.details,
        duration_seconds=round(duration, 4),
        validation_seconds=round(validation.duration_seconds, 4),
        loop_steps=int(result.get("loop_step") or 0),
        tool_calls=tool_calls,
        tool_calls_by_name=by_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        workspace_modified=_file_digest(workspace) != original_digest,
        workspace=relpath_for_display(workspace),
        provider=os.getenv("MODEL_PROVIDER", "anthropic"),
        executor_backend=os.getenv("EXECUTOR_BACKEND", "local"),
        final_suggestion=suggestion,
        error="; ".join(errors) if errors else None,
    )


def run_suite(
    cases: Iterable[BenchmarkCase],
    *,
    max_loop_steps: int = 12,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    cases = list(cases)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results = [run_case(case, max_loop_steps=max_loop_steps) for case in cases]
    passed = sum(result.success for result in results)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "provider": os.getenv("MODEL_PROVIDER", "anthropic"),
        "executor_backend": os.getenv("EXECUTOR_BACKEND", "local"),
        "max_loop_steps": max_loop_steps,
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "success_rate": round(passed / len(results), 4) if results else 0.0,
        "total_duration_seconds": round(sum(item.duration_seconds for item in results), 4),
        "total_tool_calls": sum(item.tool_calls for item in results),
        "total_tokens": sum(item.total_tokens for item in results),
        "results": [result.to_dict() for result in results],
    }

    destination = Path(output_dir) if output_dir else DEFAULT_RESULTS_DIR / run_id
    destination = resolve_within_root(str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report, report_path
