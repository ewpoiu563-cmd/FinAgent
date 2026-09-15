"""Evaluate fixed-pool qwen3.7-text-rerank on the frozen multi-document hard set.

This command makes live API calls.  It does not alter the Gold file, embeddings,
index, or Dense/BM25/Hybrid retrievers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate_dense_retrieval import aggregate_metrics, load_and_validate_gold
from eval.evaluate_multi_document_retrieval import annotate, diagnostics
from eval.evaluate_hybrid_retrieval import score_case as score_hybrid_case
from rag.bm25_retriever import BM25Retriever
from rag.dense_retriever import DenseRetriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import RerankerRetriever

CANDIDATE_K = 20
FINAL_K = 5
DEFAULT_GOLD = ROOT / "eval" / "rag_gold_hard.json"
DEFAULT_SOURCE = ROOT / "outputs" / "chunks" / "doc1_v2.jsonl"
DEFAULT_INDEX = ROOT / "outputs" / "index" / "multi_doc_v1"
DEFAULT_REPORT = ROOT / "RERANKER_MULTI_DOC_HARD_EVAL_REPORT.md"
DEFAULT_RECORDS = ROOT / "outputs" / "eval" / "reranker_multi_doc_hard_records.jsonl"


def movement(rrf_rank: int | None, rerank_rank: int | None) -> str:
    """Classify first-relevant-rank movement, including Top5 failures/recovery."""

    if rrf_rank is None and rerank_rank is not None:
        return "recovered"
    if rrf_rank is not None and rerank_rank is None:
        return "failed"
    if rrf_rank is None:
        return "unchanged"
    if rerank_rank < rrf_rank:
        return "improved"
    if rerank_rank > rrf_rank:
        return "degraded"
    return "unchanged"


def build_record(case: Mapping[str, Any], rrf_results: Sequence[Mapping[str, Any]], reranked: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rrf = score_hybrid_case(case, rrf_results)
    record = annotate(case, reranked, hybrid=True)
    record["rrf_first_relevant_rank"] = rrf["first_relevant_rank"]
    record["movement"] = movement(rrf["first_relevant_rank"], record["first_relevant_rank"])
    reranked_top5 = [
        {
            "rank": result["rank"],
            "retrieval_unit_id": result["retrieval_unit_id"],
            "chunk_id": result["chunk_id"],
            "doc_id": result["doc_id"],
            "pages": list(result["pages"]),
            "rerank_score": result["rerank_score"],
        }
        for result in reranked[:FINAL_K]
    ]
    retrieved_set = set(record["retrieved_ids"])
    gold_groups = record["gold_groups"]
    record["covered_evidence_groups"] = sorted(
        group for group, unit_ids in gold_groups.items() if retrieved_set & set(unit_ids)
    )
    record["missing_evidence_groups"] = sorted(
        set(gold_groups) - set(record["covered_evidence_groups"])
    )
    record["reranked_top5_ids"] = [result["retrieval_unit_id"] for result in reranked_top5]
    record["reranked_results"] = reranked_top5
    return record


def write_report(path: Path, *, evaluated_at: str, gold_sha: str, index_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    metrics, diagnostic = aggregate_metrics(records), diagnostics(records)
    counts = Counter(record["movement"] for record in records)
    lines = [
        "# Reranker Multi-document Hard Evaluation Report", "",
        "## Evaluation setup", "",
        f"- Evaluated at (UTC): `{evaluated_at}`",
        f"- Gold Set: `eval/rag_gold_hard.json` ({len(records)} queries, SHA-256 `{gold_sha}`); checked unchanged after retrieval.",
        f"- Index: `{index_dir.resolve().relative_to(ROOT).as_posix()}`.",
        "- Candidate generation (frozen): Dense Top20 + BM25 Top20 → retrieval-unit normalization/dedup → RRF k=60.",
        "- Reranker: `qwen3.7-text-rerank`; every query reranks the RRF candidate pool (maximum 20 units), then emits Final Top5.",
        "- Candidate text: `headings + text`; candidate metadata/provenance are preserved and only the final ordering changes.",
        "- No Gold modification, query rewrite, document embedding, index rebuild, retriever modification, or parameter sweep.", "",
        "## Overall metrics", "", "| Metric | Value |", "|---|---:|",
    ]
    lines.extend(f"| {name} | {value:.2%} |" for name, value in metrics.items())
    lines.extend(["", "## Multi-document diagnostic metrics", "", "| Metric | Value |", "|---|---:|"])
    lines.extend(f"| {name} | {value:.2%} |" for name, value in diagnostic.items())
    lines.extend(["", "## RRF → Reranker first relevant rank movement", "", "| Class | Queries |", "|---|---:|"])
    lines.extend(f"| {name} | {counts[name]} |" for name in ("improved", "unchanged", "degraded", "failed", "recovered"))
    lines.extend(["", "## Per-query results", "", "| Query ID | Question | RRF first relevant rank | Reranker first relevant rank | Movement | Reranked Top5 retrieval unit | Reranked Top5 doc_id | Reranked Top5 page | Rerank score | First target-doc rank | CrossDocTop1Error | Recall@5 / evidence-group coverage |", "|---|---|---:|---:|---|---|---|---|---|---:|---:|---|"])
    for record in records:
        rank = lambda value: "—" if value is None else str(value)
        results = record["reranked_results"]
        ids = "<br>".join(f"`{result['retrieval_unit_id']}`" for result in results)
        docs = "<br>".join(f"`{result['doc_id']}`" for result in results)
        pages = "<br>".join(str(result["pages"]) for result in results)
        scores = "<br>".join(f"{result['rerank_score']:.6f}" for result in results)
        coverage = ", ".join(f"`{group}`" for group in record["covered_evidence_groups"]) or "none"
        question = str(record["question"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {record['query_id']} | {question} | {rank(record['rrf_first_relevant_rank'])} | "
            f"{rank(record['first_relevant_rank'])} | {record['movement']} | {ids} | {docs} | {pages} | {scores} | "
            f"{rank(record['first_target_doc_rank'])} | {int(record['cross_document_top1_error'])} | "
            f"{record['recall_at_5']:.2f} / {coverage} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load completed rerank records without constructing retrievers or calling APIs."""

    if not path.is_file():
        raise FileNotFoundError(f"reranker records checkpoint does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is an empty checkpoint record")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            required = {"query_id", "question", "reranked_results", "reranked_top5_ids", "top5_doc_ids", "first_relevant_rank", "rrf_first_relevant_rank", "movement", "first_target_doc_rank", "cross_document_top1_error", "recall_at_5", "covered_evidence_groups", "hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5", "target_doc_at_1", "target_doc_at_5"}
            missing = required - set(value)
            if missing:
                raise ValueError(f"{path}:{line_number} is missing record fields: {sorted(missing)}")
            records.append(value)
    if not records:
        raise ValueError(f"reranker records checkpoint is empty: {path}")
    return records


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Durably append one completed query before proceeding to the next API call."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--source-chunks", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--report-only", action="store_true", help="Generate Markdown from an existing JSONL checkpoint without any API calls.")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete JSONL checkpoint; only missing query IDs call the reranker API.")
    args = parser.parse_args(argv)
    original_gold, cases = load_and_validate_gold(args.gold, args.source_chunks)
    gold_sha = hashlib.sha256(original_gold).hexdigest()
    if args.report_only:
        records = load_records(args.records)
        if len(records) != len(cases):
            raise ValueError(f"report-only requires {len(cases)} records, got {len(records)}")
        write_report(args.report, evaluated_at=datetime.now(timezone.utc).isoformat(), gold_sha=gold_sha, index_dir=args.index_dir, records=records)
        return 0
    existing = load_records(args.records) if args.resume and args.records.exists() else []
    if args.records.exists() and not args.resume:
        raise FileExistsError(f"reranker records checkpoint already exists: {args.records}; use --report-only or --resume")
    completed_ids = {record["query_id"] for record in existing}
    load_dotenv(ROOT / ".env")
    dense, bm25 = DenseRetriever(args.index_dir), BM25Retriever(args.index_dir)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
    reranker = RerankerRetriever(hybrid)
    records: list[dict[str, Any]] = list(existing)
    for case in cases:
        if case["query_id"] in completed_ids:
            continue
        rrf_candidates = hybrid.retrieve(case["question"], top_k=CANDIDATE_K, candidate_k=CANDIDATE_K)
        reranked = reranker.rerank_candidates(case["question"], rrf_candidates, final_k=FINAL_K)
        record = build_record(case, rrf_candidates[:FINAL_K], reranked)
        records.append(record)
        append_record(args.records, record)
        print(f"{case['query_id']}: RRF={record['rrf_first_relevant_rank']} reranker={record['first_relevant_rank']} {record['movement']}", flush=True)
    if args.gold.read_bytes() != original_gold:
        raise RuntimeError("Gold file changed during evaluation")
    if len(records) != len(cases):
        raise RuntimeError("reranker evaluation ended without one record per Gold query")
    write_report(args.report, evaluated_at=datetime.now(timezone.utc).isoformat(), gold_sha=gold_sha, index_dir=args.index_dir, records=records)
    print(json.dumps({**aggregate_metrics(records), **diagnostics(records), "movement": Counter(record["movement"] for record in records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
