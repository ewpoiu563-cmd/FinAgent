"""Evaluate local BM25 retrieval against the unchanged RAG Gold Set.

Run from the repository root:

    conda run -n finagent python -m eval.evaluate_bm25_retrieval

The evaluator requests raw Top 5 results and applies the same source-unit
relevance boundary as dense evaluation: parent_chunk_id or chunk_id.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.bm25_retriever import BM25Retriever, BM25RetrievalResult


DEFAULT_GOLD_PATH = ROOT / "eval" / "rag_gold.json"
DEFAULT_SOURCE_CHUNKS_PATH = ROOT / "outputs" / "chunks" / "doc1_v2.jsonl"
DEFAULT_INDEX_DIR = ROOT / "outputs" / "index" / "full_doc1"
DEFAULT_REPORT_PATH = ROOT / "BM25_RETRIEVAL_EVAL_REPORT.md"
TOP_K = 5
REQUIRED_QUERY_TYPES = {
    "company_business",
    "operating_model",
    "risks",
    "customers_suppliers",
    "financial_operating_metrics",
}
METRICS = ("Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR@5")


def retrieval_unit_id(chunk: Mapping[str, Any]) -> str:
    """Return the source retrieval unit shared by a chunk and its children."""

    unit_id = chunk.get("parent_chunk_id") or chunk.get("chunk_id")
    if not isinstance(unit_id, str) or not unit_id.strip():
        raise ValueError("retrieved chunk has no valid chunk_id/retrieval unit id")
    return unit_id


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _pages(value: Any, path: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(page, int) or isinstance(page, bool) or page <= 0
            for page in value
        )
    ):
        raise ValueError(f"{path} must be a non-empty list of positive integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{path} must not contain duplicate pages")
    return value


def load_jsonl_by_chunk_id(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid source JSON at {path}:{line_number}") from error
            if not isinstance(chunk, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            chunk_id = _nonempty_string(
                chunk.get("chunk_id"), f"{path}:{line_number}.chunk_id"
            )
            if chunk_id in chunks:
                raise ValueError(f"duplicate source chunk_id: {chunk_id}")
            chunks[chunk_id] = chunk
    return chunks


def _without_whitespace(text: str) -> str:
    return "".join(text.split())


def _normalized_report_path(path: str | Path) -> str:
    """Normalize report paths so Windows and POSIX separators compare equally."""

    return str(path).replace("\\", "/")


def load_report_metrics(
    report_path: Path, *, gold_path: Path, gold_sha256: str
) -> dict[str, float]:
    """Load verified overall metrics from a same-Gold Dense evaluation report."""

    report = report_path.read_text(encoding="utf-8")
    header = re.search(
        r"- Gold Set: `([^`]+)` \(\d+ queries, SHA-256 `([0-9a-fA-F]{64})`\)",
        report,
    )
    if header is None:
        raise ValueError(f"Dense report has no Gold Set header: {report_path}")
    reported_path, reported_hash = header.groups()
    expected_path = _normalized_report_path(gold_path.resolve().relative_to(ROOT))
    if _normalized_report_path(reported_path) != expected_path:
        raise ValueError(
            "Dense report Gold path does not match evaluation input: "
            f"report={reported_path!r}, input={expected_path!r}"
        )
    if reported_hash.lower() != gold_sha256.lower():
        raise ValueError(
            "Dense report Gold hash does not match evaluation input: "
            f"report={reported_hash}, input={gold_sha256}"
        )

    metrics: dict[str, float] = {}
    for name, value in re.findall(
        r"^\| (Hit@1|Hit@3|Hit@5|Recall@5|MRR@5) \| ([0-9.]+)% \|$",
        report,
        flags=re.MULTILINE,
    ):
        metrics.setdefault(name, float(value) / 100)
    if set(metrics) != set(METRICS):
        raise ValueError(f"Dense report has incomplete overall metrics: {report_path}")
    return metrics


def load_and_validate_gold(
    gold_path: Path, source_chunks_path: Path
) -> tuple[bytes, list[dict[str, Any]]]:
    """Apply the same schema and provenance validation as dense evaluation."""

    original = gold_path.read_bytes()
    try:
        cases = json.loads(original)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Gold JSON: {gold_path}") from error
    if not isinstance(cases, list) or not 15 <= len(cases) <= 20:
        raise ValueError("rag_gold.json must contain 15 to 20 query objects")

    source_chunks = load_jsonl_by_chunk_id(source_chunks_path)
    query_ids: set[str] = set()
    observed_types: set[str] = set()
    for case_index, case in enumerate(cases):
        path = f"gold[{case_index}]"
        if not isinstance(case, dict):
            raise ValueError(f"{path} must be an object")
        query_id = _nonempty_string(case.get("query_id"), f"{path}.query_id")
        if query_id in query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        query_ids.add(query_id)
        _nonempty_string(case.get("question"), f"{path}.question")
        query_type = _nonempty_string(case.get("query_type"), f"{path}.query_type")
        observed_types.add(query_type)
        doc_id = _nonempty_string(case.get("doc_id"), f"{path}.doc_id")

        evidence_list = case.get("gold_evidence")
        if not isinstance(evidence_list, list) or not evidence_list:
            raise ValueError(f"{path}.gold_evidence must be a non-empty list")
        evidence_ids: set[str] = set()
        for evidence_index, evidence in enumerate(evidence_list):
            evidence_path = f"{path}.gold_evidence[{evidence_index}]"
            if not isinstance(evidence, dict):
                raise ValueError(f"{evidence_path} must be an object")
            unit_id = _nonempty_string(
                evidence.get("retrieval_unit_id"),
                f"{evidence_path}.retrieval_unit_id",
            )
            _nonempty_string(
                evidence.get("evidence_group"),
                f"{evidence_path}.evidence_group",
            )
            if unit_id in evidence_ids:
                raise ValueError(f"{query_id} has duplicate Gold unit {unit_id}")
            evidence_ids.add(unit_id)
            pages = _pages(evidence.get("pages"), f"{evidence_path}.pages")
            evidence_text = _nonempty_string(
                evidence.get("evidence_text"), f"{evidence_path}.evidence_text"
            )

            source = source_chunks.get(unit_id)
            if source is None:
                raise ValueError(
                    f"{query_id} Gold unit {unit_id} does not exist in source chunks"
                )
            if source.get("doc_id") != doc_id:
                raise ValueError(
                    f"{query_id} Gold unit {unit_id} doc_id differs from source"
                )
            if source.get("pages") != pages:
                raise ValueError(
                    f"{query_id} Gold unit {unit_id} pages differ from source: "
                    f"gold={pages}, source={source.get('pages')}"
                )
            source_text = source.get("text")
            if not isinstance(source_text, str) or _without_whitespace(
                evidence_text
            ) not in _without_whitespace(source_text):
                raise ValueError(
                    f"{query_id} evidence_text is not a verbatim whitespace-normalized "
                    f"excerpt of source unit {unit_id}"
                )

    missing_types = REQUIRED_QUERY_TYPES - observed_types
    if missing_types:
        raise ValueError(f"Gold Set is missing required query types: {sorted(missing_types)}")
    return original, cases


def score_case(
    case: Mapping[str, Any], results: Sequence[BM25RetrievalResult]
) -> dict[str, Any]:
    """Score one query without filtering or deduplicating ranked results."""

    gold_ids = [item["retrieval_unit_id"] for item in case["gold_evidence"]]
    gold_groups: dict[str, set[str]] = defaultdict(set)
    for evidence in case["gold_evidence"]:
        gold_groups[evidence["evidence_group"]].add(evidence["retrieval_unit_id"])
    gold_set = set(gold_ids)
    retrieved_ids = [retrieval_unit_id(result) for result in results[:TOP_K]]
    retrieved_set = set(retrieved_ids)
    relevant_ranks = [
        rank for rank, unit_id in enumerate(retrieved_ids, start=1) if unit_id in gold_set
    ]
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
    covered_groups = sorted(
        group for group, unit_ids in gold_groups.items() if unit_ids & retrieved_set
    )
    missing_groups = sorted(set(gold_groups) - set(covered_groups))
    return {
        "query_id": case["query_id"],
        "question": case["question"],
        "query_type": case["query_type"],
        "gold_ids": gold_ids,
        "gold_groups": {
            group: sorted(unit_ids) for group, unit_ids in sorted(gold_groups.items())
        },
        "retrieved_top5_ids": retrieved_ids,
        "first_relevant_rank": first_relevant_rank,
        "hit_at_1": bool(first_relevant_rank and first_relevant_rank <= 1),
        "hit_at_3": bool(first_relevant_rank and first_relevant_rank <= 3),
        "hit_at_5": bool(first_relevant_rank and first_relevant_rank <= 5),
        "recall_at_5": len(covered_groups) / len(gold_groups),
        "mrr_at_5": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "covered_evidence_groups": covered_groups,
        "missing_evidence_groups": missing_groups,
        "retrieved": [
            {
                "rank": result["rank"],
                "score": result["score"],
                "retrieval_unit_id": retrieval_unit_id(result),
                "chunk_id": result["chunk_id"],
                "pages": result["pages"],
                "headings": result["headings"],
                "text": result["text"],
            }
            for result in results[:TOP_K]
        ],
    }


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot aggregate an empty evaluation")
    total = len(records)
    return {
        "Hit@1": sum(record["hit_at_1"] for record in records) / total,
        "Hit@3": sum(record["hit_at_3"] for record in records) / total,
        "Hit@5": sum(record["hit_at_5"] for record in records) / total,
        "Recall@5": sum(record["recall_at_5"] for record in records) / total,
        "MRR@5": sum(record["mrr_at_5"] for record in records) / total,
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _ids(ids: Iterable[str]) -> str:
    values = list(ids)
    return "<br>".join(f"`{unit_id}`" for unit_id in values) if values else "—"


def _result_detail_lines(record: Mapping[str, Any], *, heading_level: int = 3) -> list[str]:
    prefix = "#" * heading_level
    lines = [
        f"{prefix} {record['query_id']}",
        "",
        record["question"],
        "",
        f"- Gold units: {_ids(record['gold_ids'])}",
        f"- First relevant rank: {record['first_relevant_rank'] or '未进入 Top5'}",
        f"- Recall@5: {record['recall_at_5']:.2f}",
        "",
        "| Rank | Score | Retrieval unit | Returned chunk | Pages | Headings | Text preview |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for result in record["retrieved"]:
        preview = " ".join(result["text"].split())[:160]
        headings = " > ".join(result["headings"]) or "—"
        lines.append(
            f"| {result['rank']} | {result['score']:.6f} | "
            f"`{result['retrieval_unit_id']}` | `{result['chunk_id']}` | "
            f"{_markdown_cell(result['pages'])} | {_markdown_cell(headings)} | "
            f"{_markdown_cell(preview)} |"
        )
    lines.append("")
    return lines


def write_report(
    report_path: Path,
    *,
    evaluated_at: str,
    gold_path: Path,
    gold_sha256: str,
    dense_report_path: Path,
    dense_metrics: Mapping[str, float],
    source_chunks_path: Path,
    index_dir: Path,
    retriever: BM25Retriever,
    records: Sequence[Mapping[str, Any]],
) -> None:
    metrics = aggregate_metrics(records)
    type_counts = Counter(record["query_type"] for record in records)
    failed = [record for record in records if record["recall_at_5"] == 0]
    partial = [record for record in records if 0 < record["recall_at_5"] < 1]
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["query_type"]].append(record)

    financial_records = by_type["financial_operating_metrics"]
    financial_metrics = aggregate_metrics(financial_records)
    financial_failed = [
        record["query_id"] for record in financial_records if record["recall_at_5"] == 0
    ]

    lines = [
        "# BM25 Retrieval Evaluation Report",
        "",
        "## Evaluation setup",
        "",
        f"- Evaluated at (UTC): `{evaluated_at}`",
        f"- Gold Set: `{gold_path.resolve().relative_to(ROOT).as_posix()}` ({len(records)} queries, "
        f"SHA-256 `{gold_sha256}`)",
        f"- Source chunks: `{source_chunks_path.relative_to(ROOT)}`; Gold IDs, doc IDs, "
        "pages and verbatim evidence excerpts passed the same provenance validation as "
        "Dense Eval.",
        f"- Metadata: `{(index_dir / 'metadata.jsonl').relative_to(ROOT)}` "
        f"({retriever.chunk_count} physical chunks).",
        "- Searchable text: `headings + text`; metadata is loaded without mutation.",
        "- Retrieval policy: `rank_bm25.BM25Okapi` defaults, jieba mixed-language "
        "tokenization, raw `top_k=5`; no k1/b tuning, heading boost, query rewriting, "
        "score threshold, parent dedup, RRF or reranker.",
        "- Relevance boundary: `chunk.get(\"parent_chunk_id\") or chunk[\"chunk_id\"]`, "
        "identical to Dense Eval. Units within an `evidence_group` are OR alternatives; "
        "different groups are complementary. Recall@5 counts covered groups; Hit/MRR "
        "use the first match to any group.",
        f"- Query type counts: {dict(sorted(type_counts.items()))}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {value:.2%} |" for name, value in metrics.items())
    lines.extend(
        [
            "",
            "## Metrics by query type",
            "",
            "| Query type | Queries | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@5 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for query_type, type_records in sorted(by_type.items()):
        type_metrics = aggregate_metrics(type_records)
        lines.append(
            f"| {query_type} | {len(type_records)} | "
            f"{type_metrics['Hit@1']:.2%} | {type_metrics['Hit@3']:.2%} | "
            f"{type_metrics['Hit@5']:.2%} | {type_metrics['Recall@5']:.2%} | "
            f"{type_metrics['MRR@5']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Dense comparison",
            "",
            f"Dense metrics are loaded from the verified same-Gold report: `{dense_report_path}`.",
            "",
            "| Metric | BM25 | Dense | BM25 − Dense |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in METRICS:
        lines.append(
            f"| {name} | {metrics[name]:.2%} | {dense_metrics[name]:.2%} | "
            f"{metrics[name] - dense_metrics[name]:+.2%} |"
        )

    lines.extend(
        [
            "",
            "## Financial operating metrics analysis",
            "",
            f"This category contains {len(financial_records)} queries. BM25 achieved "
            f"Hit@1 {financial_metrics['Hit@1']:.2%}, Hit@3 "
            f"{financial_metrics['Hit@3']:.2%}, Hit@5 "
            f"{financial_metrics['Hit@5']:.2%}, Recall@5 "
            f"{financial_metrics['Recall@5']:.2%}, and MRR@5 "
            f"{financial_metrics['MRR@5']:.2%}.",
            f"Failed financial queries: {', '.join(financial_failed) or 'none'}. "
            "All per-query ranks, Top5 units, failed queries, and partial evidence-group "
            "results below are rendered directly from this evaluation run's `records`; no "
            "prior easy-set analysis text is reused.",
            "",
            "## Per-query summary",
            "",
            "| Query ID | Type | Gold IDs | Retrieved Top5 IDs | First relevant rank | "
            "Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@5 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['query_id']} | {record['query_type']} | "
            f"{_ids(record['gold_ids'])} | {_ids(record['retrieved_top5_ids'])} | "
            f"{record['first_relevant_rank'] or '—'} | {int(record['hit_at_1'])} | "
            f"{int(record['hit_at_3'])} | {int(record['hit_at_5'])} | "
            f"{record['recall_at_5']:.2f} | {record['mrr_at_5']:.2f} |"
        )

    lines.extend(["", "## Per-query Top5", ""])
    for record in records:
        lines.extend(_result_detail_lines(record))

    lines.extend(["## Failed queries", ""])
    if failed:
        lines.append(
            f"{len(failed)} queries retrieved none of their Gold evidence groups in Top5."
        )
        lines.append("")
        for record in failed:
            lines.extend(_result_detail_lines(record))
    else:
        lines.extend(["No query had Recall@5 = 0.", ""])

    lines.extend(["## Partial multi-evidence queries", ""])
    if partial:
        lines.append(
            f"{len(partial)} queries covered at least one but not all required evidence groups."
        )
        lines.append("")
        for record in partial:
            lines.extend(
                [
                    f"### {record['query_id']}",
                    "",
                    f"- Covered groups: {_ids(record['covered_evidence_groups'])}",
                    f"- Missing groups: {_ids(record['missing_evidence_groups'])}",
                    f"- Recall@5: {record['recall_at_5']:.2f}",
                    "",
                ]
            )
    else:
        lines.extend(["No query had 0 < Recall@5 < 1.", ""])

    lines.extend(
        [
            "## Scope",
            "",
            "This is a standalone BM25 baseline. It does not implement or simulate RRF, "
            "a reranker, Agent integration, DenseRetriever changes, FAISS re-indexing, "
            "Gold Set edits, or parent deduplication.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument(
        "--source-chunks", type=Path, default=DEFAULT_SOURCE_CHUNKS_PATH
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--dense-report",
        type=Path,
        default=ROOT / "DENSE_RETRIEVAL_EVAL_REPORT.md",
        help="same-Gold Dense report used for comparison metrics",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate Gold schema/provenance without loading BM25 metadata",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    original_gold, cases = load_and_validate_gold(args.gold, args.source_chunks)
    gold_sha256 = hashlib.sha256(original_gold).hexdigest()
    dense_metrics = load_report_metrics(
        args.dense_report, gold_path=args.gold, gold_sha256=gold_sha256
    )
    print(
        f"Gold validation passed: queries={len(cases)} sha256={gold_sha256}",
        flush=True,
    )
    if args.validate_only:
        return 0

    retriever = BM25Retriever(args.index_dir)
    indexed_units = {retrieval_unit_id(chunk) for chunk in retriever._chunks}
    missing_units = sorted(
        {
            evidence["retrieval_unit_id"]
            for case in cases
            for evidence in case["gold_evidence"]
        }
        - indexed_units
    )
    if missing_units:
        raise ValueError(f"Gold retrieval units are absent from metadata: {missing_units}")

    records: list[dict[str, Any]] = []
    for case in cases:
        results = retriever.retrieve(case["question"], top_k=TOP_K)
        record = score_case(case, results)
        records.append(record)
        print(
            f"{record['query_id']}: first_rank={record['first_relevant_rank']} "
            f"hit@5={record['hit_at_5']} recall@5={record['recall_at_5']:.2f}",
            flush=True,
        )

    if args.gold.read_bytes() != original_gold:
        raise RuntimeError("Gold file changed during evaluation")
    evaluated_at = datetime.now(timezone.utc).isoformat()
    write_report(
        args.report,
        evaluated_at=evaluated_at,
        gold_path=args.gold,
        gold_sha256=gold_sha256,
        dense_report_path=args.dense_report,
        dense_metrics=dense_metrics,
        source_chunks_path=args.source_chunks,
        index_dir=args.index_dir,
        retriever=retriever,
        records=records,
    )
    print(json.dumps(aggregate_metrics(records), ensure_ascii=False, indent=2))
    print(f"Report written to {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
