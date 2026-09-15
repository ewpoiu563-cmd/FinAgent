# Evaluation protocol v0.1

## Scope

FinAgent is evaluated at seven boundaries: intent/requirement recognition,
tool necessity and routing, SQL execution, RAG retrieval and evidence
selection, Web research, answer synthesis/citation, and end-to-end behavior.

The primary unit is a user case. A case contains one or more independently
scored requirements. Difficulty (`daily`, `hard`, `boundary`, `long_tail`) is
an orthogonal tag rather than a substitute for route or capability.

## Dataset plan

| Split | Cases | May tune on it? | Headline use |
|---|---:|---|---|
| calibration | 60 | yes | diagnostic only |
| blind_test | 120 | no | primary |
| challenge | 60 | no | separate robustness headline |

Difficulty totals across all 240 cases are 84 daily, 72 hard, 48 boundary,
and 36 long-tail. Scenario-family targets are 24 direct/local-compute, 42 SQL,
42 RAG, 30 Web, 42 hybrid, 30 multi-turn/dependency, and 30
failure/refusal/security cases.

The RAG views must expose at least 120 independently phrased retrieval queries
with relevance judgments at retrieval-unit level. Queries may be requirements
from RAG, hybrid, and multi-turn cases; they are not necessarily 120 extra E2E
cases.

## Gold authoring

Gold is created before execution and has two reviewers for blind-test and
challenge cases. The author records exact evidence identity and independently
decidable assertions.

- SQL: execute a reviewed read-only oracle query against the declared database
  hash; preserve typed rows and ordering semantics.
- RAG: inspect the original PDF and indexed retrieval-unit text; record document
  id, page, retrieval-unit id, evidence group, and graded relevance (0-3).
- Stable Web: preserve URL, publisher, publication/fact dates, capture time,
  content hash, and the excerpt supporting each claim.
- Live Web: record the same runtime metadata but report separately because the
  source and ranking can drift.
- Direct/local compute: record the derivation or deterministic calculation.

Gold statuses are `draft`, `needs_review`, `verified`, and `frozen`. Only
`frozen` cases can enter a scored run. Corrections create a new dataset version;
they do not rewrite a published score.

## Run policy

- One run per case is used for the primary result.
- A stratified 60-case stability sample receives two additional runs.
- No best-of-N selection is permitted; repetitions are averaged and stability
  is reported independently.
- Record model/configuration, database/index/corpus hashes, reference time,
  session id, run id, answer, complete trace, latency, calls, and token usage.
- A missing answer or incomplete trace is retained and scores zero for strict
  E2E success.

## Precision-recall operating point

Candidate generation and reranking are evaluated separately. Current
production uses candidate K=20 and final K=5; v2 treats those as a baseline,
not an assumed optimum.

On calibration only, sweep candidate K in `{10,20,30,50}`, final K in
`{1,3,5,8,10}`, and reranker thresholds chosen from the observed score
distribution. Because Dense, BM25, RRF, and reranker scores are not the same
probability space, thresholds are calibrated per final scorer/version and are
never transferred blindly between score types.

Selection is constrained optimization:

1. Require macro evidence-group Recall to meet the declared floor (initially
   0.95 overall and 0.90 in every difficulty slice).
2. Reject settings whose no-answer false-positive rate exceeds 0.05.
3. Among feasible settings, maximize F-beta with beta=2, which weights recall
   four times as strongly as precision.
4. If multiple settings are statistically tied, choose the one with smaller
   final K, fewer context tokens, and lower P95 latency.

The report includes Precision-Recall curves, K-vs-Recall, K-vs-Precision,
K-vs-MRR, K-vs-nDCG, threshold-vs-no-answer false-positive rate, and
quality/cost Pareto plots. The selected point and its rationale are frozen
before blind-test execution.

## Human assessment

Answer claims are rated 0-3 for correctness, completeness, groundedness,
citation entailment, citation completeness, relevance, temporal consistency,
and numerical/unit consistency. Strict E2E is binary and passes only if every
required requirement, route/source constraint, critical fact, evidence rule,
and citation rule passes with no critical error.

The primary assistant may prepare and provisionally grade Gold because the
workflow is explicitly requested, but 10-20% of cases and every disputed or
critical-error case should receive owner review. Reviewer disagreements are
preserved rather than silently resolved.

