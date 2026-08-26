"""Phase 3.1 deterministic retrieval evaluation tests."""

from __future__ import annotations

import json

import pytest

from agent.rag.evaluation import (
    RagEvalCase,
    RetrievalTarget,
    evaluate_case,
    load_rag_cases,
    run_rag_evaluation,
    select_rag_cases,
)
from agent.rag.models import SearchResult


def _result(path: str, symbol: str, score: float = 0.8) -> SearchResult:
    return SearchResult(
        path=path,
        symbol=symbol,
        kind="function",
        start_line=1,
        end_line=5,
        content=f"def {symbol}():\n    pass\n",
        score=score,
    )


class FakeIndex:
    def __init__(self, results: dict[str, list[SearchResult]]) -> None:
        self.results = results

    def search(self, query, *, top_k, path_prefix, strategy="hybrid"):
        values = self.results[query]
        if path_prefix:
            values = [item for item in values if item.path.startswith(path_prefix)]
        return values[:top_k]


def test_load_default_rag_manifest_and_select_cases():
    cases = load_rag_cases()
    selected = select_rag_cases(cases, ["official_validator_gate"])

    assert len(cases) >= 10
    assert selected[0].targets[0].symbol == "benchmark_validate"
    with pytest.raises(ValueError, match="Unknown RAG"):
        select_rag_cases(cases, ["missing"])


def test_evaluate_case_accepts_split_chunk_symbol_suffix():
    case = RagEvalCase(
        id="sync",
        query="incremental sync",
        targets=(RetrievalTarget("src/agent/rag/index.py", "CodeIndex.sync"),),
    )
    index = FakeIndex({
        case.query: [
            _result("src/agent/config.py", "get_chat_model"),
            _result("src/agent/rag/index.py", "CodeIndex.sync#part2"),
        ]
    })

    result = evaluate_case(index, case, top_k=5, strategy="vector")

    assert result.first_relevant_rank == 2
    assert result.matched_target_count == 1
    assert result.path_hit and result.symbol_hit
    assert result.to_dict()["reciprocal_rank"] == 0.5


def test_run_rag_evaluation_computes_metrics_and_writes_per_case(tmp_path):
    cases = [
        RagEvalCase(
            id="first",
            query="first query",
            targets=(RetrievalTarget("src/target.py", "target"),),
        ),
        RagEvalCase(
            id="miss",
            query="miss query",
            targets=(RetrievalTarget("src/missing.py", "missing"),),
        ),
    ]
    index = FakeIndex({
        "first query": [
            _result("src/noise.py", "noise"),
            _result("src/target.py", "target"),
        ],
        "miss query": [_result("src/noise.py", "noise")],
    })
    output = tmp_path.name
    # Evaluation outputs are intentionally restricted to the repository; use its test workspace.
    from agent.tools._paths import PROJECT_ROOT

    destination = PROJECT_ROOT / "benchmarks" / "workspaces" / output
    try:
        report, report_path = run_rag_evaluation(
            index, cases, top_ks=[1, 3], output_dir=destination
        )
        loaded = json.loads(report_path.read_text(encoding="utf-8"))

        assert report["metrics"]["at_1"]["hit_rate"] == 0.0
        assert report["metrics"]["at_3"]["hit_rate"] == 0.5
        assert report["metrics"]["at_3"]["mrr"] == 0.25
        assert loaded["schema_version"] == 1
        assert loaded["strategy"] == "hybrid"
        assert (destination / "cases" / "first.json").exists()
        assert (destination / "cases" / "miss.json").exists()
    finally:
        import shutil

        shutil.rmtree(destination, ignore_errors=True)


def test_run_rag_evaluation_rejects_invalid_top_k():
    case = RagEvalCase(
        id="case",
        query="query",
        targets=(RetrievalTarget("src/file.py"),),
    )
    with pytest.raises(ValueError, match="between 1 and 20"):
        run_rag_evaluation(FakeIndex({"query": []}), [case], top_ks=[0])
