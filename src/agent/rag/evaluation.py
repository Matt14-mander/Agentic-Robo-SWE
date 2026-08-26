"""Deterministic evaluation for source-code retrieval quality."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from agent.rag.index import SearchStrategy
from agent.rag.models import SearchResult
from agent.tools._paths import PROJECT_ROOT, resolve_within_root

DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "rag_cases.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "rag_results"
ProgressCallback = Callable[[dict[str, Any]], None]


class SearchIndex(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        path_prefix: str | None = None,
        strategy: SearchStrategy = "hybrid",
        use_cache: bool = True,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RetrievalTarget:
    path: str
    symbol: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalTarget":
        path = str(data.get("path", "")).replace("\\", "/").strip()
        if not path:
            raise ValueError("RAG evaluation target requires a non-empty path")
        symbol = data.get("symbol")
        return cls(path=path, symbol=str(symbol) if symbol is not None else None)


@dataclass(frozen=True)
class RagEvalCase:
    id: str
    query: str
    targets: tuple[RetrievalTarget, ...]
    path_prefix: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagEvalCase":
        case_id = str(data.get("id", "")).strip()
        query = str(data.get("query", "")).strip()
        targets = tuple(RetrievalTarget.from_dict(item) for item in data.get("targets", []))
        if not case_id or not query or not targets:
            raise ValueError("RAG evaluation case requires id, query and at least one target")
        prefix = data.get("path_prefix")
        return cls(
            id=case_id,
            query=query,
            targets=targets,
            path_prefix=str(prefix).replace("\\", "/") if prefix is not None else None,
            tags=tuple(str(tag) for tag in data.get("tags", [])),
        )


@dataclass(frozen=True)
class RagCaseResult:
    case_id: str
    query: str
    targets: tuple[RetrievalTarget, ...]
    latency_ms: float
    retrieved: tuple[SearchResult, ...]
    first_relevant_rank: int | None
    matched_target_count: int
    path_hit: bool
    symbol_hit: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reciprocal_rank"] = (
            round(1.0 / self.first_relevant_rank, 6) if self.first_relevant_rank else 0.0
        )
        return payload


def load_rag_cases(manifest: str | Path = DEFAULT_MANIFEST) -> list[RagEvalCase]:
    path = Path(manifest)
    if not path.is_absolute():
        path = resolve_within_root(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported RAG evaluation schema_version: {data.get('schema_version')!r}")
    cases = [RagEvalCase.from_dict(item) for item in data.get("cases", [])]
    ids = [case.id for case in cases]
    if not cases:
        raise ValueError("RAG evaluation manifest contains no cases")
    if len(ids) != len(set(ids)):
        raise ValueError("RAG evaluation case ids must be unique")
    return cases


def select_rag_cases(
    cases: Iterable[RagEvalCase], selected_ids: Iterable[str]
) -> list[RagEvalCase]:
    cases = list(cases)
    selected = list(selected_ids)
    if not selected:
        return cases
    by_id = {case.id: case for case in cases}
    unknown = sorted(set(selected) - by_id.keys())
    if unknown:
        raise ValueError(f"Unknown RAG evaluation case(s): {', '.join(unknown)}")
    return [by_id[case_id] for case_id in selected]


def _symbol_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(f"{expected}#part")


def _target_matches(result: SearchResult, target: RetrievalTarget) -> bool:
    return result.path.replace("\\", "/") == target.path and (
        target.symbol is None or _symbol_matches(result.symbol, target.symbol)
    )


def evaluate_case(
    index: SearchIndex,
    case: RagEvalCase,
    *,
    top_k: int,
    strategy: SearchStrategy = "hybrid",
) -> RagCaseResult:
    started = time.perf_counter()
    retrieved = tuple(index.search(
        case.query,
        top_k=top_k,
        path_prefix=case.path_prefix,
        strategy=strategy,
        use_cache=False,
    ))
    latency_ms = round((time.perf_counter() - started) * 1000, 3)

    relevant_ranks = [
        rank
        for rank, result in enumerate(retrieved, start=1)
        if any(_target_matches(result, target) for target in case.targets)
    ]
    matched_targets = {
        target
        for target in case.targets
        if any(_target_matches(result, target) for result in retrieved)
    }
    expected_paths = {target.path for target in case.targets}
    path_hit = any(result.path.replace("\\", "/") in expected_paths for result in retrieved)
    symbol_targets = [target for target in case.targets if target.symbol is not None]
    symbol_hit = any(target in matched_targets for target in symbol_targets) if symbol_targets else path_hit
    return RagCaseResult(
        case_id=case.id,
        query=case.query,
        targets=case.targets,
        latency_ms=latency_ms,
        retrieved=retrieved,
        first_relevant_rank=min(relevant_ranks) if relevant_ranks else None,
        matched_target_count=len(matched_targets),
        path_hit=path_hit,
        symbol_hit=symbol_hit,
    )


def _metrics_at_k(results: list[RagCaseResult], k: int) -> dict[str, float]:
    if not results:
        return {
            "hit_rate": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "path_hit_rate": 0.0,
            "symbol_hit_rate": 0.0,
        }
    case_count = len(results)
    hits = 0
    recall = 0.0
    reciprocal_ranks = 0.0
    path_hits = 0
    symbol_hits = 0
    for result in results:
        limited = result.retrieved[:k]
        matched = {
            target
            for target in result.targets
            if any(_target_matches(item, target) for item in limited)
        }
        ranks = [
            rank
            for rank, item in enumerate(limited, start=1)
            if any(_target_matches(item, target) for target in result.targets)
        ]
        expected_paths = {target.path for target in result.targets}
        hits += bool(matched)
        recall += len(matched) / len(result.targets)
        reciprocal_ranks += 1.0 / min(ranks) if ranks else 0.0
        path_hits += any(item.path.replace("\\", "/") in expected_paths for item in limited)
        symbol_targets = [target for target in result.targets if target.symbol is not None]
        symbol_hits += (
            any(target in matched for target in symbol_targets)
            if symbol_targets
            else any(item.path.replace("\\", "/") in expected_paths for item in limited)
        )
    return {
        "hit_rate": round(hits / case_count, 4),
        "recall": round(recall / case_count, 4),
        "mrr": round(reciprocal_ranks / case_count, 4),
        "path_hit_rate": round(path_hits / case_count, 4),
        "symbol_hit_rate": round(symbol_hits / case_count, 4),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_rag_evaluation(
    index: SearchIndex,
    cases: Iterable[RagEvalCase],
    *,
    top_ks: Iterable[int] = (1, 3, 5),
    strategy: SearchStrategy = "hybrid",
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], Path]:
    cases = list(cases)
    if not cases:
        raise ValueError("RAG evaluation requires at least one case")
    normalized_ks = tuple(sorted({int(k) for k in top_ks}))
    if not normalized_ks or normalized_ks[0] < 1 or normalized_ks[-1] > 20:
        raise ValueError("top_ks must contain values between 1 and 20")
    if strategy not in {"vector", "lexical", "hybrid"}:
        raise ValueError(f"Unsupported search strategy: {strategy}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_dir) if output_dir else DEFAULT_RESULTS_DIR / run_id
    destination = resolve_within_root(str(destination))
    cases_dir = destination / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    max_k = normalized_ks[-1]
    results: list[RagCaseResult] = []
    started = time.perf_counter()

    if progress:
        progress({"event": "suite_start", "case_count": len(cases), "output_dir": str(destination)})
    for position, case in enumerate(cases, start=1):
        if progress:
            progress({"event": "case_start", "position": position, "case_count": len(cases), "case_id": case.id})
        result = evaluate_case(index, case, top_k=max_k, strategy=strategy)
        results.append(result)
        case_path = cases_dir / f"{case.id}.json"
        _write_json_atomic(case_path, result.to_dict())
        if progress:
            progress({
                "event": "case_complete",
                "position": position,
                "case_count": len(cases),
                "case_id": case.id,
                "first_relevant_rank": result.first_relevant_rank,
                "latency_ms": result.latency_ms,
                "result_path": str(case_path),
            })

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(results),
        "strategy": strategy,
        "retrieval_cache_enabled": False,
        "top_ks": list(normalized_ks),
        "max_top_k": max_k,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "average_query_latency_ms": round(
            sum(result.latency_ms for result in results) / len(results), 3
        ),
        "metrics": {f"at_{k}": _metrics_at_k(results, k) for k in normalized_ks},
        "results": [result.to_dict() for result in results],
    }
    report_path = destination / "report.json"
    _write_json_atomic(report_path, report)
    return report, report_path
