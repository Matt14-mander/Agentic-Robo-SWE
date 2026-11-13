# Independent C++ retrieval v1

This is a controlled, public retrieval fixture authored for this project. It is
separate from Harness source, repair benchmarks and the four-query C++ smoke.
The corpus contains 20 self-contained C++17 headers: 8 development targets,
8 heldout targets and 4 similar distractors. The headers pass a combined
`g++ -std=c++17 -Wall -Wextra -pedantic -fsyntax-only` check.

`dev.json` and `heldout.json` each contain 16 queries: one identifier query and
one conceptual query per target. Conceptual queries include English and Chinese.
Target files and queries are disjoint across splits; both splits retrieve from
the same full corpus. This is a query/target split, not two separate indexes.

Only `corpus/` is indexed. Query labels, this README, reports and index artifacts
remain outside it. The runner validates split separation and records hashes of
every corpus file and both manifests. v1 was frozen before its first model run;
no queries, targets, fusion weights or defaults were adjusted after observing
development or heldout results.

```bash
# Offline CI protocol check, explicitly excludes semantic inference.
uv run python scripts/run_cpp_retrieval_eval.py --split dev --hash-only

# Five-arm comparisons, requiring the pinned local semantic model.
uv run python scripts/run_cpp_retrieval_eval.py --split dev
uv run python scripts/run_cpp_retrieval_eval.py --split heldout
```

The heldout labels are public and inspectable: this is a reproducible regression
set, not a private blind evaluation. Reusing it to tune retrieval requires a new
heldout set. Metrics measure target-file retrieval; they do not measure C++ AST
resolution, compilation correctness, repair success or large-repository scaling.
The small corpus, short headers, identifier-heavy query mix and Chinese queries
against English code are material limits on generalization.
