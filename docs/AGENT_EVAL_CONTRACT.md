# Phase 5 — Agent Eval Contract

> **Archived as of 2026-09-12:** Future evaluation is manual, per the project owner's decision. This historical contract is not the current acceptance workflow. See [Manual Evaluation](MANUAL_EVALUATION.md). Existing evaluator code and artifacts are retained for reference; no new automated scoring is required.

> Status: design only. This contract defines the data and scoring boundary for
> Phase 5. It does not authorize an evaluator implementation, changes to
> production orchestration, changes to the Phase 4 trace schema, or changes to
> the frozen Phase 2 retrieval system.

## 1. Purpose and non-goals

Phase 5 evaluates the end-to-end Agent: decomposition, temporal interpretation,
tool necessity, source selection, execution, evidence use, synthesis, and
user-visible outcome. It is not a replacement for the frozen Phase 2
retrieval-component evaluations or the Phase 4 observability regression suite.

`AggregateMetrics.success_rate` from Phase 4 is **not Agent accuracy**. It is
runtime business-outcome telemetry: among filtered runs with a known outcome,
the share for which `business_outcome == "success"`. A runtime may report
success while selecting an unnecessary source, answering a Gold assertion
incorrectly, using unacceptable evidence, or failing a freshness constraint.
Conversely, an expected clarification or controlled failure can be correct for
a Phase 5 case even though it is not a runtime `success`.

The evaluator must be read-only with respect to the production path. It may
read an immutable case set, run artifacts, raw trace JSONL, and RunSummary
records; it must never silently alter source routing, retrieval data, trace
events, or their schemas in order to obtain a score.

## 2. Evaluation model and unit of analysis

The hierarchy is deliberately requirement-first, matching `TaskPlan`.

| Unit | Meaning | Identifier | Scoring role |
|---|---|---|---|
| **case** | One versioned user scenario and its immutable Gold record. It may contain one or more atomic requirements. | `case_id` | Benchmark sampling, stratification, and case-level result. |
| **requirement** | One independently evaluable user obligation in a case, including its temporal scope, expected necessity mode, source constraints, answer assertions, and evidence rules. | stable `requirement_id` within `case_id` | Primary unit for routing, source, correctness, and coverage scoring. |
| **run** | One execution attempt of one case under an explicitly recorded system/configuration snapshot. A run is bound to exactly one `case_id`, one `run_id`, and one reference date. | `run_id` | Observation unit for latency, calls, retries, fallback, tokens, and a case attempt result. |

A multi-requirement case is not flattened into a single source label. It is
scored per requirement and then aggregated using Gold requirement weights
(default weight `1`). A synthesis-only task (for example `SYNTHESIS`) is not a
new user requirement; it is evaluated through the case-level assertions and
the requirements it combines.

Each benchmark report must state the run policy: normally one deterministic
attempt per case; if repeated attempts are permitted, report both per-run
metrics and case-level results using a predeclared reducer (default: mean of
run-level scores; no best-of-* selection). A run without a complete trace or
required output artifact is retained as an invalid/incomplete run, reported
separately, and scored `0` in the primary accuracy denominator. A diagnostic
evaluable-only view may be published, but it cannot replace the primary score.

## 3. Gold record schema

Gold is versioned, reviewable data, not instructions passed to the Agent. The
following fields are required at the case level unless noted otherwise. A
requirement inherits a case-level value only when its own value is absent.

```yaml
case_id: "phase5-sql-001"                 # unique, immutable once published
question: "..."                           # exact user input
reference_date: "2026-09-10"              # ISO-8601 evaluation time anchor
category: "sql"                           # one primary category below
tags: ["historical", "fund", "single_requirement"]
requirements:
  - requirement_id: "R1"
    question: "..."                       # atomic obligation after human Gold decomposition
    required: true
    weight: 1
    depends_on: []                         # stable requirement_ids in this case
    expected_tool_necessity_mode: "tool_required"
    web_evaluation_mode: "not_applicable" # not_applicable, stable, live
    acceptable_source_sets:             # every entry is one complete, minimal legal plan
      - ["sql"]
    required_sources: ["sql"]
    forbidden_sources: ["web"]
    fallback_expectation:
      mode: "forbidden"                 # forbidden, optional, required
      from_source: null                  # required for optional/required path rules
      to_source: null
      trigger: null
    freshness_requirement:
      mode: "historical"                  # one of none, historical, current, bounded_live
      as_of: "2024-12-31"                 # optional when mode is none
      max_age_days: null                   # required for bounded_live when applicable
    expected_answer_assertions: []
    acceptable_evidence: []
    expected_requirement_outcome: "success" # success, insufficient, source_failure,
                                              # generation_failure, provider_content_block,
                                              # clarification_needed
expected_tool_necessity_mode: "tool_required" # optional shorthand only for uniform requirements
acceptable_source_sets:                         # optional case default
  - ["sql"]
required_sources: ["sql"]                     # optional case default
forbidden_sources: ["web"]                     # optional case default
freshness_requirement: {mode: "historical", as_of: "2024-12-31"}
expected_answer_assertions: []                  # optional case-level/synthesis assertions
acceptable_evidence: []                         # optional case-level evidence rules
expected_case_outcome: "success"               # success, partial_success, source_failure,
                                                 # generation_failure, provider_content_block,
                                                 # clarification_needed
```

### 3.1 Required field semantics

- `case_id` is stable and unique. Corrections create a new Gold-set version;
  previously published results retain the version they used.
- `question` is supplied verbatim to the Agent. `reference_date` is injected
  only through the normal evaluation configuration and must agree with the
  Gold record; it is not a substitute for the user question.
- `category` is the primary stratum; `tags` support overlapping slices such as
  `current`, `historical`, `multi_requirement`, `fallback_expected`, and
  `clarification_expected`.
- `requirements` is the authoritative decomposition. Every requirement has a
  stable `requirement_id`, `required` flag, and positive `weight`. Optional
  requirements must not affect the primary success or coverage denominator.
- `depends_on` is a duplicate-free list of stable `requirement_id` values in
  the same case; it defaults to `[]`. A requirement cannot depend on itself or
  on an unknown requirement. It expresses a Gold execution dependency, not a
  new user obligation. The evaluator must report a dependency-caused
  noncompletion/failure separately from the dependent requirement's own source
  or execution failure; it must not silently reclassify one as the other.
- `expected_tool_necessity_mode` is one of the current policy values:
  `no_tool`, `tool_required`, or `exploratory`. `no_tool` permits direct model
  reasoning or deterministic local computation but forbids external SQL, RAG,
  and Web evidence unless the Gold requirement explicitly says otherwise.
  `exploratory` remains a valid ToolNecessityPolicy schema value. Current
  production `TaskPlanner` does not implement complete exploratory execution
  semantics: it routes the decision to the legacy exploration path instead of
  producing a fully requirement-scoped execution plan. Benchmark v1 therefore
  excludes Gold requirements whose expected mode is `exploratory` from
  headline accuracy denominators and reports them as a separate experimental
  slice. An `exploratory` prediction for a headline `no_tool` or
  `tool_required` requirement is still incorrect.
- `acceptable_source_sets`, `required_sources`, and `forbidden_sources` contain
  canonical source names: `direct`, `local_compute`, `sql`, `rag`, and `web`.
  `acceptable_source_sets` is a non-empty list of duplicate-free source sets;
  each set is one complete, minimal, legal **planned** source set. A requirement
  with equally valid SQL or Web plans is represented as `[["sql"], ["web"]]`,
  not as a flattened `acceptable_sources: ["sql", "web"]`. Thus
  `["sql", "web"]` is rejected unless that exact compound set is explicitly
  listed. `required_sources` may be retained to name sources required in every
  legal plan and must be a subset of every acceptable source set. No source
  may be both forbidden and in any acceptable set. `direct` and
  `local_compute` are source labels when present in the planned task and must
  be listed by Gold; they are not silently ignored as presentation details.
  `web` is not made legal merely because a runtime fallback exists.
- `web_evaluation_mode` is a requirement-level protocol selector with values
  `not_applicable`, `stable`, or `live`. It defaults to `not_applicable` for a
  non-Web requirement. A Web requirement must explicitly declare `stable` or
  `live`; the evaluator must not infer the stable/live scoring protocol from
  tags.
- `freshness_requirement` records the truth-time constraint independently of
  the run date. `none` permits stable knowledge; `historical` fixes an
  `as_of` date or bounded historical interval and does **not** imply current
  lookup; `current` requires evidence current at the reference date;
  `bounded_live` additionally sets a maximum allowed evidence age. A current
  answer based on a newer date is not automatically valid if it changes the
  question's stated time scope.
- `expected_answer_assertions` is a list of typed, independently decidable
  claims, not a single reference prose string. Each assertion declares a
  validation method and severity. Typical forms are `numeric` (target,
  absolute/relative tolerance, unit), `set_or_rows` (typed rows and key),
  `claim` (required propositions and forbidden contradictions), `relation`
  (comparison/ranking/arithmetic invariant), `citation` (claim-to-evidence
  linkage), and `absence_or_refusal` (what must not be claimed). The exact
  validator is selected by category in section 6.
- A record marked `gold_status: "draft_unverified"` may use `null` for
  `expected_answer_assertions` and `acceptable_evidence` only as an explicit
  non-scoreable TODO pending human verification. Published Gold must replace
  those placeholders with Contract-compliant lists.
- `acceptable_evidence` contains one or more evidence rules, not just URLs.
  A rule identifies allowed `source`, stable identity (for example SQL table
  and snapshot id, RAG document id/retrieval unit id, or Web URL/capture id),
  optional publisher, publication/as-of range, and which assertion ids it may
  support. Rules may mark evidence as `required`. Evidence not meeting a rule
  is unacceptable even if its prose appears persuasive.
- `expected_requirement_outcome` is the terminal outcome for one Gold
  requirement and aligns with `TaskStatus`: `success`, `insufficient`,
  `source_failure`, `generation_failure`, `provider_content_block`, or
  `clarification_needed`. `partial_success` is never valid here: it is a
  case/run aggregation result, not an atomic requirement outcome.
- A `draft_unverified` zero-row/failure placeholder whose concrete entity or
  temporal combination has not yet been verified may set both requirement- and
  case-level expected outcomes to `null`. It is not a runnable or scoreable
  Gold case; a published case must use the enumerated outcomes above.
- `expected_case_outcome` is the expected final user-visible run status. It
  may be `success`, `partial_success`, `source_failure`,
  `generation_failure`, `provider_content_block`, or
  `clarification_needed`. A case-level value is not a shortcut around
  requirement outcomes: all required requirements retain their own expected
  requirement outcome. Fallback expectations are evaluated separately and
  must not be encoded by changing this field.

Gold should also carry non-scoring provenance metadata in practice: `gold_set`
version, author/reviewer, creation/review dates, fixture snapshot IDs, and an
optional `notes` field. These fields improve auditability but do not change a
published score.

### 3.2 Fallback expectation and compliance

`fallback_expectation` is a requirement-level execution constraint, separate
from `expected_requirement_outcome` and `expected_case_outcome`:

```yaml
fallback_expectation:
  mode: required                         # forbidden | optional | required
  from_source: sql
  to_source: web
  trigger: zero_result                   # canonical, reviewed trigger predicate
```

`mode: forbidden` means no fallback transition is allowed for the requirement;
`from_source`, `to_source`, and `trigger` may be null to forbid all fallback,
or may scope the prohibition to one path. `optional` permits, but does not
require, one matching transition. `required` requires at least one matching
transition and its trigger. `from_source`, `to_source`, and `trigger` are
mandatory for scoped `optional`/`required` paths. Multiple permitted paths are
represented as a reviewed list of expectation objects, not an implicit broad
allowlist.

A transition matches only when its raw `fallback.dispatched` event has the
same runtime task mapping, `from_source`, and `to_source` (using event fields
or the documented attributes fallback). Its `trigger` must be supported by the
preceding source-result artifact and the event's reason/original-status fields
when available. For example, `zero_result` requires evidence that the primary
source returned no usable rows/results and entered the declared insufficient
path; it cannot be inferred merely from a later Web call. If current Phase 4
events and preserved source-result artifacts cannot establish the trigger, the
fallback trigger check is `not_evaluable` and scores `0` in the primary
benchmark under the existing missing-artifact rule. A matching fallback is an
authorized execution addition to the planned source set, not a planned-source
set mismatch; any actual extra source without such an authorized matching path
fails actual-source compliance. Required, forbidden, and optional fallback
compliance are reported separately; fallback is never accepted just because
the final case outcome happened to be correct.

## 4. Required evaluation categories

Every case has exactly one primary category and may have cross-cutting tags.

| Category | Scope and expected evaluation focus |
|---|---|
| **no-tool / local compute** | Stable reasoning, calculations, transformations, or a deterministic local fixture. Verify the decision not to obtain external evidence and the mathematical/data transformation result. |
| **SQL** | Questions answered from the versioned structured finance database. Verify entity grounding, temporal scope, row/aggregate semantics, and database-snapshot evidence. |
| **RAG** | Questions whose authoritative evidence is in the frozen local document corpus. Verify supported claims and document/retrieval-unit provenance; do not reopen Phase 2 retrieval tuning. |
| **Web** | Questions needing public, current, or explicitly Web-sourced information. Verify source provenance, truth time, citation support, and reproducibility class. |
| **Hybrid / compound requirements** | Multiple atomic requirements or a requirement legitimately requiring multiple sources. Score each requirement/source separately, then test the synthesized relations and completion semantics. |
| **failure / fallback / clarification** | Invalid temporal anchors, unavailable or zero-result sources, controlled recovery, provider/content failure, and cases that must request clarification rather than fabricate an answer. |

Category balance must be reported by case and by required requirement count.
Hybrid cases must not be relabeled as a single-source success merely because
one sibling requirement completed. Fallback cases must distinguish a Gold-
permitted recovery source from an unplanned or forbidden source.

## 5. Run artifact and alignment contract

An eventual evaluator must create a separate, versioned evaluation result per
run. This is an evaluator artifact, not a Phase 4 schema extension. At minimum
it links `case_id`, `run_id`, Gold-set version, system/configuration snapshot,
the exact input/reference date, final status and answer, normalized planned
requirements/necessity decisions/source plan, normalized evidence, and paths
or immutable identifiers for raw trace and RunSummary.

The run artifact must persist a normalized/serialized `TaskPlan` requirement
artifact before execution-result aggregation. Each runtime `TaskPlan.tasks[]`
entry must retain at least: runtime task id, atomic question, serialized tool
necessity decision, and planned source (`TaskPlan.tasks[].source`). It should
also retain `required`, dependencies, and planned task status for audit. The
evaluation result must separately retain the terminal task status/result (or
the task lifecycle events from which it was derived) for requirement-outcome
scoring. These are evaluator-owned capture data; persisting them does not alter
the production `TaskPlan` or Phase 4 event schema.

Alignment rules:

1. Map every required Gold requirement to zero, one, or more planned runtime
   TaskRequirements in the normalized TaskPlan artifact. Mapping uses a stable
   origin/id when available; otherwise it is an evaluator-owned, reviewed
   mapping based on atomic-question scope. A mapping to at least one planned
   TaskRequirement counts for Planning Recall, not as a claim of exact
   decomposition equivalence. An unmapped required Gold requirement is failed
   for accuracy/coverage, never ignored.
2. Read requirement-level planned source from the mapped
   `TaskPlan.tasks[].source` values; the planned source set for a Gold
   requirement is the de-duplicated set across its mapped non-synthesis tasks.
   Read the requirement-level necessity decision from the same serialized task
   artifact, cross-checking `planning.tool_necessity.decided` when present.
   Aggregate `planning.source_plan.completed` is a case-level consistency
   audit only: it cannot reliably assign a source to each requirement.
   Read actual sources and fallbacks from task-mapped source/fallback lifecycle
   events.
3. If the serialized TaskPlan artifact and aggregate planning event disagree
   about shared case-level planning facts (for example the TaskPlan source-plan
   primary/required sources), record `plan_artifact_conflict` with both raw
   values and do not silently overwrite either. The conflict makes the
   affected requirement-level plan score `0` unless a predeclared, immutable
   Gold fixture explicitly declares the aggregate field non-applicable; it
   remains visible in all reports.
4. Treat planned requirement source sets and task-mapped actual sources as
   distinct facts. A planned hybrid is not a fallback, and an actual fallback
   does not rewrite the original plan.
5. Match evidence to assertions using its stable identity and source/time
   metadata. Trace IDs alone establish that an event occurred; they do not by
   themselves prove a factual claim. Missing answer/evidence artifacts make
   the affected Gold checks `not_evaluable`; for the primary benchmark they
   score `0`, while the report also exposes their count and reason.
6. A required clarification/failure is correct only if its status, required
   clarifying/refusal assertions, and prohibited-claim checks pass. It is not
   correct simply because execution stopped.

## 6. Metric definitions and calculation rules

All primary accuracy metrics use the declared evaluation slice and report:
numerator, denominator, rate, invalid/not-evaluable count, category breakdown,
and a 95% confidence interval when the sample size supports it. A scheduled
run/requirement with missing required evaluation artifacts remains in the
primary denominator and scores `0`; evaluable-only rates are diagnostic only.
The default aggregate is a micro average over required requirements; report
macro case averages alongside it to prevent multi-requirement cases from
dominating. The default 95% CI for every binary/proportion-style accuracy rate
is the two-sided Wilson interval. For an assertion-weighted point score, also
report its strict binary requirement-level counterpart and use that counterpart
for the default Wilson interval; do not present a Wilson interval for an
arbitrary fractional weight sum as if it were an independent Bernoulli count.

Benchmark v1 headline accuracy includes only Gold requirements expected to be
`no_tool` or `tool_required`; expected `exploratory` requirements remain in
the full confusion matrix and a separately labeled exploratory slice. They are
not silently discarded from the case inventory or operational telemetry.
A case is headline-eligible for TSR only when all of its required Gold
requirements are headline-eligible; mixed cases remain in the full report and
must publish their exploratory composition.

| Metric | Unit and formula | Source of truth |
|---|---|---|
| **Task Success Rate (TSR)** | Per run, `task_success = 1` iff final case outcome matches `expected_case_outcome` and every required Gold requirement is fully satisfied: required answer/evidence assertions, source-set, freshness, requirement outcome, and fallback checks pass, with no fatal forbidden assertion. `TSR = sum(task_success) / headline eligible runs`. Expected clarification/failure can score 1 under its own Gold rule. | Must be newly computed by Gold evaluator from final artifact, requirement results, and evidence; trace only supplies observations. |
| **Gold Requirement Planning Recall (GRPR)** | `mapped required Gold requirements / all required Gold requirements`. A Gold requirement is mapped when it maps to at least one serialized planned TaskRequirement; no exact task-count or text match is required. This detects decomposition omissions independently of source/correctness. | Gold evaluator recalculates from Gold plus normalized TaskPlan requirement artifact. |
| **Tool Necessity Accuracy (TNA)** | For every headline eligible required requirement, exact match of planned necessity mode to Gold expected mode. `TNA = correctly classified requirements / headline eligible required requirements`. Report a three-class confusion matrix for `no_tool`, `tool_required`, `exploratory` and a separate exploratory slice; do not binary-collapse the full diagnostic result. | Gold evaluator recalculates from Gold plus normalized TaskPlan requirement artifact, cross-checked against `planning.tool_necessity.decided`. |
| **Source Selection Accuracy (SSA)** | A requirement is correct iff its mapped **planned** source set is exactly a member of Gold `acceptable_source_sets`, contains every `required_sources` member, and contains no `forbidden_sources` member. Set membership is exact: an unnecessary planned extra source fails unless the expanded set is explicitly listed. Report actual-source compliance and planned-vs-actual divergence separately: actual extras are valid only when explained by a matching Gold-permitted/required fallback expectation. `SSA = correct planned requirements / headline eligible required requirements`. | Gold evaluator recalculates from Gold plus normalized TaskPlan requirement artifact and task-mapped raw trace. Aggregate source-plan events are consistency audit only, not per-requirement source truth. |
| **Answer Correctness (AC)** | Assertion-weighted score: `sum(passed required assertion weights) / sum(required assertion weights)` for answer-eligible requirements. A requirement is answer-correct only when all of its mandatory assertions pass; report this strict rate in addition to the weighted score. | Must be newly computed by category-specific Gold validators. |
| **Requirement Coverage (Gold RC)** | `sum(weight of headline required Gold requirements fully satisfied) / sum(weight of headline required Gold requirements)`. A requirement is fully satisfied only after its answer, evidence, source-set, freshness, expected-requirement-outcome, and fallback checks pass. | Must be newly computed by Gold evaluator. Do not substitute Phase 4 `requirement_coverage`. |
| **Average Tool Calls** | `sum(tool_call_count over included runs) / number of included runs`, including zero-call runs. Report tool-name/source breakdown. | Phase 4 `RunSummary.tool_call_count`; reconstruct from terminal `tool.call.completed|failed` raw events only if summary is unavailable. |
| **LLM Retry Rate** | Primary: `runs with llm_retry_count > 0 / trace-complete included runs`. Also report retry intensity `sum(llm_retry_count) / sum(llm_attempt_count)` and logical-call retry rate `sum(retried_logical_llm_call_count) / sum(logical_llm_call_count)` when known. This metric covers LLM retries only; it does not claim to measure source/tool retries. | Phase 4 RunSummary; raw terminal LLM trace events are the fallback. Missing legacy distribution is reported as unknown, not inferred. |
| **Fallback Rate** | Primary: `runs with fallback_count > 0 / trace-complete included runs`. Also report `sum(fallback_count) / included runs` and transition counts. Gold-permitted and Gold-unexpected fallback rates are separate slices. | Phase 4 `fallback_count`/`fallback_transitions`, or raw `fallback.dispatched` events. |
| **Latency** | Run wall-clock distribution (`n`, mean, p50, p95, max) over completed runs with non-null `duration_ms`; report missing/incomplete count. Do not sum LLM/tool latency as wall-clock. | Phase 4 `RunSummary.duration_ms`; raw `run.completed.duration_ms` fallback. |
| **Token usage** | Sum and per-run mean of provider input/output/total tokens over calls with known usage; report `token_usage_known_call_count`, unknown call count, and known-usage coverage. Never interpret unknown as zero. | Phase 4 RunSummary provider-token fields; raw terminal LLM events fallback. |

`RunSummary.requirement_coverage` remains useful runtime telemetry: it is
`completed_task_count / required_task_count` based on task lifecycle events.
It answers whether the orchestrator marked its own required tasks complete,
not whether those tasks match Gold requirements, are factually correct, or use
valid evidence. It may be reported next to Gold RC, never under the same name
or denominator.

### 6.1 Category-specific Answer Correctness validation

One exact-string comparison is prohibited as a universal correctness metric.
Free text is normalized only enough to extract/verify typed assertions; its
surface wording is not the target.

- **No-tool / local compute:** validate a parsed number, unit, formula,
  structured value, or deterministic fixture result against a canonical
  computation. Use declared numeric tolerances and invariant checks (for
  example sums, ordering, and dimensional units). Also confirm no external
  source was used when forbidden.
- **SQL:** run the reviewed canonical query or fixture oracle against the
  pinned database snapshot and compare typed rows/keys/aggregates, with
  numeric tolerances only where Gold permits them. Validate entity, period,
  filters, units, and SQL evidence identity (snapshot/table/row or query
  result). Natural-language text may vary; a correct-looking number for the
  wrong entity or time is incorrect.
- **RAG:** validate required and prohibited claims against Gold-identified
  document passages/retrieval-unit IDs, plus claim-to-evidence links. A
  semantic entailment rubric may be used only with stored reviewer decisions
  or a deterministic cited-passage protocol; it must check that citations
  support the stated claim, not merely overlap vocabulary. Phase 2 recall or
  rank metrics are reported separately and are not answer correctness.
- **Web:** validate atomic claims against a recorded acceptable URL/capture,
  publisher and truth-time rule, and citation linkage. Match dates, values,
  and named entities semantically/structurally rather than by full answer
  text. The validator must reject stale evidence for a current claim and
  evidence outside the source allowlist.
- **Hybrid / compound:** first apply the validator appropriate to every
  required atomic requirement, then validate cross-source synthesis
  assertions (for example comparison, reconciliation, attribution, and
  source-qualified conclusion). A hybrid answer is not correct if a fluent
  synthesis omits or contradicts an atomic requirement.
- **Failure / fallback / clarification:** validate the expected requirement
  and case outcomes separately, absence of unsupported substantive claims,
  required missing-information question or explanation, and fallback
  compliance (path, trigger, and permitted actual extra source). Do not use an
  error-string equality test as the correctness proxy.

## 7. Web time drift and reproducibility

Web facts change, pages disappear, and search rankings vary. Every Web result
must therefore record at least evaluation time, URL, publisher/domain,
retrieval/access time, published/updated time when available, and immutable
content identity (capture ID and content hash, or an equivalent archived
snapshot). Search result ordering is not Gold evidence unless captured.

The requirement-level `web_evaluation_mode` selects the protocol below. Tags
may support reporting slices but must never select a stable or live scoring
protocol.

Two distinct Web protocols are required:

1. **Stable Web benchmark.** Gold points to approved archived/captured Web
   artifacts with a fixed capture date and hash. The Agent should be supplied
   the reproducible fixture or a controlled replay endpoint. Correctness is
   evaluated against those artifacts, so scores are comparable across code
   versions. This is the primary benchmark used for release comparisons.
2. **Live Web evaluation.** The Agent accesses the live Web at a declared run
   timestamp. Gold defines claim type, allowed publishers, temporal window,
   and validation procedure rather than assuming a fixed string/value forever.
   Store the retrieved pages/citations where policy permits, plus hashes and
   access times. Results are labeled with the run window and are not directly
   comparable to stable benchmark scores; a changed source is data drift, not
   automatically an Agent regression.

For `current`/`bounded_live` cases, the evaluator checks the claim against the
reference date and declared staleness window. For historical Web cases, it
checks historical truth time, not the freshness of a modern page that merely
describes history. Any unavailable capture, unverifiable timestamp, blocked
page, or source contradiction is recorded as `not_evaluable` or failed under a
predeclared policy; it is never repaired by changing Gold during scoring.

## 8. Phase 4 data reuse boundary

| Available directly from Phase 4 summaries/raw trace | Requires Gold evaluator recomputation |
|---|---|
| run identity/completeness, business outcome, run duration, event count | Task Success Rate and expected-outcome correctness |
| aggregate planned primary/required source, actual source lifecycle, fallback count and transitions | requirement-level source scoring from serialized `TaskPlan.tasks[].source`; Tool Necessity Accuracy; Gold Requirement Planning Recall; Gold requirement mapping; plan-artifact conflict detection |
| task lifecycle completion and runtime `requirement_coverage` | Gold Requirement Coverage and whether a completed task actually satisfied the user requirement |
| LLM/tool/SQL counts, retries, latency sums, provider token sums and known/unknown coverage | Answer Correctness, claim/evidence support, freshness compliance, prohibited-claim checks |
| synthesis/dependency failure status | requirement-vs-case outcome validation, fallback-trigger compliance, and hybrid synthesis correctness |

Raw trace is authoritative for event-level reconstruction; RunSummary is the
versioned derived convenience layer. Neither contains the Gold truth oracle,
semantic answer assertions, complete evidence content, Web snapshots, or a
case-to-requirement correctness judgment. Those are intentionally Phase 5
evaluator responsibilities.

## 9. Reporting and acceptance requirements

Each evaluation report must identify: Gold-set version and hash, category/tag
distribution, system/configuration and data snapshot IDs, reference-date/run
window, run policy, trace-completeness rate, every metric's numerator and
denominator, invalid/not-evaluable counts, and the distinction between stable
and live Web slices. It must separately report `plan_artifact_conflict`, Gold
Requirement Planning Recall, planned-source SSA, actual-source compliance,
fallback compliance by mode/path, and the exploratory slice. Binary accuracy
rates use Wilson 95% CIs by default. It must publish Phase 4 runtime success
telemetry only with its runtime label and never as `accuracy`, `Task Success
Rate`, or Answer Correctness.

Before implementation starts, the first reviewed Gold set must include cases
from all six categories, at least one multi-requirement hybrid, one permitted
fallback, one forbidden-fallback/source case, one historical and one current
temporal constraint, and one clarification/failure case. Phase 5 code may be
implemented only after this contract and its initial Gold examples are
reviewed and frozen for the first benchmark version.
