# FinAgent Eval v2

`eval_v2` is a clean evaluation program. Historical datasets under `eval/` are
retained for audit only and are not inputs to v2 scores.

The evaluation order is deliberately one-way:

1. Author questions from a declared coverage matrix.
2. Derive facts from SQLite rows, PDF pages/chunks, or frozen Web captures.
3. Review and freeze Gold records before any FinAgent run.
4. Record dataset and system hashes.
5. Tune retrieval only on `calibration`.
6. Freeze the operating point, then run `blind_test` and `challenge` once.
7. Compute deterministic metrics and perform claim-level human review.

The planned corpus contains 240 cases: 60 calibration, 120 blind-test, and 60
challenge cases. The first implementation milestone is the independently
verified 60-case calibration set. A record is not runnable while its
`gold_status` is `draft` or `needs_review`.

## Directories

- `spec/`: protocol, coverage, metric definitions, and review rubric.
- `schemas/`: machine-readable dataset contracts.
- `data/`: versioned case manifests and immutable Gold records.
- `scripts/`: validators, deterministic scorers, runners, and report builders.
- `runs/`: generated runtime artifacts; never treated as Gold.
- `reports/`: generated tables, curves, and human adjudication reports.

## Non-negotiable controls

- Gold is never inferred from the Agent answer.
- A semantic reference answer alone is insufficient: every factual claim must
  link to a versioned evidence object.
- Calibration results may change parameters; blind-test and challenge results
  may not.
- Web stable and Web live results are reported separately.
- Runtime `success` is telemetry, not end-to-end correctness.
- Invalid or incomplete runs stay in the primary denominator and score zero.

## v0.1 calibration baseline

The newly authored 60-case suite has now completed a real E2E run and a
single-reviewer human pass. Four additional no-answer RAG cases support
threshold replay. See `reports/BASELINE_REPORT_v0.1.md`. Strict E2E success is
45%. K=3 with a reranker threshold near 0.678 is only a provisional candidate
and must not change production until it passes the frozen blind test.
