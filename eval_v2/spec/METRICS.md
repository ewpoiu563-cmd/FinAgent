# Metrics v0.1

## Intent, requirements, and routing

- Intent accuracy; per-class precision, recall, F1; macro/micro F1; confusion
  matrix.
- Requirement precision/recall/F1 over Gold obligations.
- Entity, temporal-scope, source-constraint, and dependency accuracy.
- Tool-necessity accuracy.
- Exact route-set match and route-set precision/recall/F1.
- Required-source recall, forbidden-source violation rate, unnecessary-tool
  call rate, fallback compliance, and clarification-decision accuracy.

## SQL

- Parse/syntax validity, executable rate, read-only safety rate.
- Execution accuracy and typed result-set exact match.
- Row and column precision/recall/F1.
- Entity grounding, temporal filter, aggregation, grouping, ordering, limit,
  null handling, and empty-result behavior accuracy.
- Numeric accuracy with Gold tolerances and unit/scale accuracy.

SQL string exact match is diagnostic only; execution semantics and returned
facts are primary.

## RAG

Let `G_q` be the set of relevant evidence groups for query `q`, and `R_q@K`
the top-K retrieved units. A unit may satisfy one or more evidence groups.

- `Hit@K(q) = 1` when any unit in `R_q@K` matches any group in `G_q`.
- `Recall@K(q) = covered relevant groups / |G_q|`.
- `Precision@K(q) = relevant returned units / K_actual`.
- `RR@K(q) = 1/rank` of the first relevant unit, or zero.
- `MRR@K` is mean reciprocal rank across queries.
- `DCG@K = sum((2^rel_i - 1) / log2(i+1))` using graded relevance 0-3.
- `nDCG@K = DCG@K / ideal_DCG@K`, defined as zero when no relevant Gold
  evidence exists and reported separately for no-answer queries.

Also report MAP@K, context precision/recall, document-scope accuracy,
cross-document contamination, candidate Recall@K before reranking, reranker
gain, evidence-selection precision/recall, and no-answer false-positive rate.

Precision-Recall curves are produced from the calibrated final score. The
report must state whether aggregation is micro or macro and must not interpret
RRF scores as probabilities.

## Answer and citations

- Claim correctness and strict required-assertion accuracy.
- Completeness/requirement coverage.
- Groundedness and unsupported-claim rate.
- Citation entailment, citation precision, and citation completeness.
- Faithfulness to SQL rows, PDF text, or Web captures.
- Numeric, unit, currency, and temporal consistency.
- Conflict-handling and correct-refusal rates.

## Web

- Source authority, fact freshness, publisher attribution, URL validity,
  search/fetch success, and source-replacement success.
- Correctness, completeness, groundedness, citation entailment, citation
  precision/completeness, and faithfulness to captured text.
- Stable-Web and live-Web metrics are never pooled into one headline score.

## End to end and operations

- Strict E2E success rate, partial-success rate, critical-error rate,
  hallucination rate, and expected-failure/refusal success.
- Slice results by route, difficulty, entity, temporal mode, single/multi-turn,
  and answerable/unanswerable.
- Trace completeness, run consistency, latency P50/P95, model/tool calls,
  tokens, and estimated cost.
- Bernoulli rates include Wilson 95% intervals; model comparisons use paired
  bootstrap intervals or McNemar tests on the same frozen cases.

