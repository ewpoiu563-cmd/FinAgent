# Calibration Gold review log v0.1

Review date: 2026-09-13

## Direct/local-compute cases

Eight cases were derived from explicit arithmetic definitions or stable
financial terminology. Numeric results were independently recomputed from the
expressions stored with each assertion. Status: verified for calibration.

## SQL cases

Sixteen questions and oracle queries were authored without reading a FinAgent
answer. Queries were checked for read-only form, table grain, entity, date,
ordering, aggregation, null/zero-row semantics, and requested columns. They
were then executed through a read-only SQLite URI against the recorded
`finance.db` SHA-256. Materialized rows are the standard answers. Status:
verified for calibration, pending owner spot-check before benchmark freeze.

One drafting defect was caught during materialization: the scale-change table
uses `报告期期初基金总份额`, not `报告期初基金总份额`. The seed was corrected
before any Agent run. This correction is part of Gold authoring, not post-run
adaptation.

## RAG cases

Sixteen new questions were authored across the four indexed prospectuses.
Every primary evidence page was rendered from the source PDF and visually
inspected. Indexed unit ids, physical PDF pages, printed pages, evidence text,
and PDF/text hashes are materialized separately.

Visual inspection caught evidence split across physical chunks for the
Xinlitai confidentiality controls, Yaxia material-cost sentence, Yaxia human
resources plan, and Sinocera sensitivity figures. These additional units were
added before retrieval execution. Answer Gold is verified for calibration.
Retrieval judgments remain `seed_verified_needs_pooling`: Precision and nDCG
must not be published until pooled top candidates are manually judged, because
unjudged-but-relevant chunks would otherwise be counted as false positives.

## Web cases

Six cases use official PBOC, CSRC, or SSE pages. Three are stable historical
facts and three are facts verified at the 2026-09-13 reference date. Expected
claims, publisher, source URL, publication/capture date, and supporting excerpt
are recorded. Live-Web cases must be refreshed or marked drifted on later
dates; they are never pooled with stable-Web headline metrics.

## Compound/dialog/failure cases

Fourteen cases combine independently verified Gold records or assert controlled
failure/clarification behavior. Cross-references are validated by the manifest
builder. Status: verified for calibration.

