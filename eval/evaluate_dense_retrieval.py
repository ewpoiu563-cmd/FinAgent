"""Evaluate the persisted dense retriever against the manually curated RAG Gold Set.

Run from the repository root:

    conda run -n finagent python -m eval.evaluate_dense_retrieval

The evaluator always requests the original, unfiltered Top 5 from ``DenseRetriever``.
Fallback children are scored at their source-unit boundary by replacing ``chunk_id``
with ``parent_chunk_id`` when the latter is present.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.dense_retriever import DenseRetriever, DenseRetrievalResult


DEFAULT_GOLD_PATH = ROOT / "eval" / "rag_gold.json"
DEFAULT_SOURCE_CHUNKS_PATH = ROOT / "outputs" / "chunks" / "doc1_v2.jsonl"
DEFAULT_INDEX_DIR = ROOT / "outputs" / "index" / "full_doc1"
DEFAULT_REPORT_PATH = ROOT / "DENSE_RETRIEVAL_EVAL_REPORT.md"
TOP_K = 5
REQUIRED_QUERY_TYPES = {
    "company_business",
    "operating_model",
    "risks",
    "customers_suppliers",
    "financial_operating_metrics",
}


def retrieval_unit_id(chunk: Mapping[str, Any]) -> str:
    """Return the source retrieval unit shared by a chunk and its children."""

    parent_id = chunk.get("parent_chunk_id")
    unit_id = parent_id or chunk.get("chunk_id")
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
            chunk = json.loads(line)
            chunk_id = _nonempty_string(
                chunk.get("chunk_id"), f"{path}:{line_number}.chunk_id"
            )
            if chunk_id in chunks:
                raise ValueError(f"duplicate source chunk_id: {chunk_id}")
            chunks[chunk_id] = chunk
    return chunks


def _without_whitespace(text: str) -> str:
    return "".join(text.split())


def load_and_validate_gold(
    gold_path: Path, source_chunks_path: Path
) -> tuple[bytes, list[dict[str, Any]]]:
    """Validate schema plus human-auditable provenance against source chunks."""

    original = gold_path.read_bytes()
    cases = json.loads(original)
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
    case: Mapping[str, Any], results: Sequence[DenseRetrievalResult]
) -> dict[str, Any]:
    """Score one query while preserving original ranks, including sibling children."""

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
    recall_at_5 = len(covered_groups) / len(gold_groups)

    return {
        "query_id": case["query_id"],
        "question": case["question"],
        "query_type": case["query_type"],
        "gold_ids": gold_ids,
        "gold_groups": {
            group: sorted(unit_ids) for group, unit_ids in sorted(gold_groups.items())
        },
        "covered_evidence_groups": covered_groups,
        "missing_evidence_groups": missing_groups,
        "retrieved_top5_ids": retrieved_ids,
        "first_relevant_rank": first_relevant_rank,
        "hit_at_1": bool(first_relevant_rank and first_relevant_rank <= 1),
        "hit_at_3": bool(first_relevant_rank and first_relevant_rank <= 3),
        "hit_at_5": bool(first_relevant_rank and first_relevant_rank <= 5),
        "recall_at_5": recall_at_5,
        "mrr_at_5": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
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
    return "<br>".join(f"`{unit_id}`" for unit_id in ids)


def _result_detail_lines(record: Mapping[str, Any]) -> list[str]:
    lines = [
        f"### {record['query_id']}",
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
    source_chunks_path: Path,
    index_dir: Path,
    retriever: DenseRetriever,
    records: Sequence[Mapping[str, Any]],
) -> None:
    metrics = aggregate_metrics(records)
    type_counts = Counter(record["query_type"] for record in records)
    failed = [record for record in records if record["recall_at_5"] == 0]
    partial = [record for record in records if 0 < record["recall_at_5"] < 1]
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["query_type"]].append(record)

    lines = [
        "# Dense Retrieval Evaluation Report",
        "",
        "## Evaluation setup",
        "",
        f"- Evaluated at (UTC): `{evaluated_at}`",
        f"- Gold Set: `{gold_path.resolve().relative_to(ROOT).as_posix()}` ({len(records)} queries, "
        f"SHA-256 `{gold_sha256}`)",
        f"- Source chunks: `{source_chunks_path.relative_to(ROOT)}`; Gold IDs, doc IDs, "
        "pages and verbatim evidence excerpts passed provenance validation.",
        f"- Index: `{index_dir.relative_to(ROOT)}`; model "
        f"`{retriever.vector_store.embedding_model}`, dimension "
        f"{retriever.vector_store.dimension}, vectors {retriever.vector_store.vector_count}.",
        "- Retrieval policy: each question is sent once to the unchanged `DenseRetriever` "
        "with `top_k=5`; no query rewriting, filtering or parameter tuning.",
        "- Relevance boundary: `parent_chunk_id` when present, otherwise `chunk_id`. "
        "Sibling fallback children therefore map to one source retrieval unit.",
        "- Gold units within one `evidence_group` are OR alternatives; different "
        "groups are complementary requirements. Recall@5 counts covered groups. "
        "Hit/MRR use the first result matching any Gold group.",
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
            "## Per-query results",
            "",
            "| Query ID | Question | Gold IDs | Retrieved Top5 IDs | First relevant rank | "
            "Hit@1 | Hit@3 | Hit@5 | Recall@5 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        first_rank = record["first_relevant_rank"] or "—"
        lines.append(
            f"| {record['query_id']} | {_markdown_cell(record['question'])} | "
            f"{_ids(record['gold_ids'])} | {_ids(record['retrieved_top5_ids'])} | "
            f"{first_rank} | {int(record['hit_at_1'])} | {int(record['hit_at_3'])} | "
            f"{int(record['hit_at_5'])} | {record['recall_at_5']:.2f} |"
        )

    lines.extend(["", "## Failed queries", ""])
    if failed:
        lines.append(
            f"{len(failed)} queries retrieved none of their Gold evidence groups in Top5. "
            "The unmodified ranked outputs follow for later hybrid/reranker analysis."
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
            lines.extend(_result_detail_lines(record))
    else:
        lines.extend(["No query had 0 < Recall@5 < 1.", ""])

    lines.extend(
        [
            "## Failure-analysis notes",
            "",
            "This report records raw Top5 IDs and normalized retrieval-unit IDs without "
            "changing the retriever. Repeated normalized IDs indicate multiple fallback "
            "children from one source unit occupying separate ranks. Recall@5 counts "
            "distinct `evidence_group` values, with OR semantics within each group; raw "
            "result ranks remain unchanged for Hit/MRR.",
            "The failed and partial sections are the candidate set for later BM25, RRF and "
            "reranker experiments; no such component is used in this evaluation.",
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
        "--validate-only",
        action="store_true",
        help="validate Gold schema/provenance without loading the index or calling embeddings",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    original_gold, cases = load_and_validate_gold(args.gold, args.source_chunks)
    gold_sha256 = hashlib.sha256(original_gold).hexdigest()
    print(
        f"Gold validation passed: queries={len(cases)} sha256={gold_sha256}",
        flush=True,
    )
    if args.validate_only:
        return 0

    load_dotenv(ROOT / ".env")
    retriever = DenseRetriever(args.index_dir)
    indexed_units = {
        retrieval_unit_id(chunk) for chunk in retriever.vector_store._chunks
    }
    missing_units = sorted(
        {
            evidence["retrieval_unit_id"]
            for case in cases
            for evidence in case["gold_evidence"]
        }
        - indexed_units
    )
    if missing_units:
        raise ValueError(f"Gold retrieval units are absent from index: {missing_units}")

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
