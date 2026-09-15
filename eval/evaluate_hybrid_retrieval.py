"""Evaluate fixed RRF hybrid retrieval against the audited RAG Gold Set.

Run from the repository root:

    conda run -n finagent python -m eval.evaluate_hybrid_retrieval

Dense and BM25 each provide their unchanged Top-20 physical chunks. RRF first
normalizes siblings to ``parent_chunk_id or chunk_id`` and then fuses ranks with
the fixed constant k=60. The raw score spaces are never mixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate_dense_retrieval import load_and_validate_gold
from rag.bm25_retriever import BM25Retriever
from rag.dense_retriever import DenseRetriever
from rag.hybrid_retriever import HybridRetriever, RRF_K, retrieval_unit_id


DEFAULT_GOLD_PATH = ROOT / "eval" / "rag_gold.json"
DEFAULT_SOURCE_CHUNKS_PATH = ROOT / "outputs" / "chunks" / "doc1_v2.jsonl"
DEFAULT_INDEX_DIR = ROOT / "outputs" / "index" / "full_doc1"
DEFAULT_REPORT_PATH = ROOT / "HYBRID_RETRIEVAL_EVAL_REPORT.md"
DENSE_REPORT_PATH = ROOT / "DENSE_RETRIEVAL_EVAL_REPORT.md"
BM25_REPORT_PATH = ROOT / "BM25_RETRIEVAL_EVAL_REPORT.md"
TOP_K = 5
CANDIDATE_K = 20
METRICS = ("Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR@5")


def _reported_gold_hash(report_path: Path, gold_path: Path) -> str:
    """Read the immutable Gold hash recorded by a baseline evaluation report."""

    if not report_path.is_file():
        raise FileNotFoundError(f"baseline report does not exist: {report_path}")
    match = re.search(
        r"Gold Set: `([^`]+)` \(\d+ queries, SHA-256 `([0-9a-fA-F]{64})`\)",
        report_path.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"baseline report has no Gold SHA-256: {report_path}")
    reported_path, reported_hash = match.groups()
    expected_path = str(gold_path.resolve().relative_to(ROOT)).replace("\\", "/")
    if reported_path.replace("\\", "/") != expected_path:
        raise ValueError(
            f"baseline report uses another Gold path: {report_path}: {reported_path!r}"
        )
    return reported_hash.lower()


def validate_baseline_gold_hashes(
    gold_sha256: str, gold_path: Path, dense_report_path: Path, bm25_report_path: Path
) -> None:
    """Refuse comparison when either stored baseline used another Gold Set."""

    baseline_hashes = {
        "Dense": _reported_gold_hash(dense_report_path, gold_path),
        "BM25": _reported_gold_hash(bm25_report_path, gold_path),
    }
    mismatches = {
        name: value for name, value in baseline_hashes.items() if value != gold_sha256
    }
    if mismatches:
        raise RuntimeError(
            "Gold Set changed after baseline evaluation; rerun Dense and BM25 Eval "
            f"before RRF Eval. current={gold_sha256}, baselines={mismatches}"
        )


def score_case(
    case: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score one ranked list at the shared retrieval-unit relevance boundary."""

    gold_ids = [item["retrieval_unit_id"] for item in case["gold_evidence"]]
    gold_groups: dict[str, set[str]] = defaultdict(set)
    for evidence in case["gold_evidence"]:
        gold_groups[evidence["evidence_group"]].add(evidence["retrieval_unit_id"])
    gold_set = set(gold_ids)
    retrieved_ids = [retrieval_unit_id(result) for result in results[:TOP_K]]
    retrieved_set = set(retrieved_ids)
    first_rank = next(
        (rank for rank, unit_id in enumerate(retrieved_ids, 1) if unit_id in gold_set),
        None,
    )
    covered_groups = sorted(
        group for group, unit_ids in gold_groups.items() if unit_ids & retrieved_set
    )
    return {
        "query_id": case["query_id"],
        "question": case["question"],
        "gold_ids": gold_ids,
        "gold_groups": {
            group: sorted(unit_ids) for group, unit_ids in sorted(gold_groups.items())
        },
        "retrieved_ids": retrieved_ids,
        "first_relevant_rank": first_rank,
        "hit_at_1": first_rank is not None and first_rank <= 1,
        "hit_at_3": first_rank is not None and first_rank <= 3,
        "hit_at_5": first_rank is not None and first_rank <= 5,
        "recall_at_5": len(covered_groups) / len(gold_groups),
        "mrr_at_5": 0.0 if first_rank is None else 1.0 / first_rank,
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


def _rank(value: int | None) -> str:
    return str(value) if value is not None else "—"


def _movement(old: int | None, new: int | None) -> str:
    if old is None and new is None:
        return "— → —"
    if old is None:
        return f"— → {new} (recovered)"
    if new is None:
        return f"{old} → — (lost)"
    delta = old - new
    if delta > 0:
        return f"{old} → {new} (↑{delta})"
    if delta < 0:
        return f"{old} → {new} (↓{-delta})"
    return f"{old} → {new} (=)"


def classify_query(
    dense_rank: int | None, bm25_rank: int | None, rrf_rank: int | None
) -> str:
    """Classify Pareto movement against both first-relevant baseline ranks."""

    if rrf_rank is None:
        return "failed"
    baseline_ranks = [
        rank if rank is not None else float("inf")
        for rank in (dense_rank, bm25_rank)
    ]
    if all(rrf_rank <= rank for rank in baseline_ranks) and any(
        rrf_rank < rank for rank in baseline_ranks
    ):
        return "improved"
    if all(rrf_rank >= rank for rank in baseline_ranks) and any(
        rrf_rank > rank for rank in baseline_ranks
    ):
        return "degraded"
    return "unchanged"


def _ids(values: Sequence[str]) -> str:
    return "<br>".join(f"`{value}`" for value in values)


def _groups(groups: Mapping[str, Sequence[str]]) -> str:
    return "<br>".join(
        f"`{group}`: {' OR '.join(f'`{unit_id}`' for unit_id in unit_ids)}"
        for group, unit_ids in groups.items()
    )


def write_report(
    report_path: Path,
    *,
    evaluated_at: str,
    gold_path: Path,
    gold_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    metrics = {
        name: aggregate_metrics([record[name] for record in records])
        for name in ("dense", "bm25", "rrf")
    }
    groups = {name: [] for name in ("improved", "degraded", "unchanged", "failed")}
    for record in records:
        groups[record["classification"]].append(record["query_id"])

    lines = [
        "# Hybrid Retrieval Evaluation Report",
        "",
        "## Evaluation setup",
        "",
        f"- Evaluated at (UTC): `{evaluated_at}`",
        f"- Gold Set: `{gold_path.resolve().relative_to(ROOT).as_posix()}` ({len(records)} queries, SHA-256 `{gold_sha256}`)",
        "- Gold audit: completed; the current evidence-group schema and hash match "
        "both rerun Dense and BM25 baseline reports.",
        "- Shared relevance boundary: `result.get(\"parent_chunk_id\") or "
        "result[\"chunk_id\"]`.",
        f"- Retrieval: unchanged Dense Top{CANDIDATE_K} + unchanged BM25 "
        f"Top{CANDIDATE_K}; retrieval-unit normalization; RRF k={RRF_K}; "
        f"final Top{TOP_K}.",
        "- Gold units within one `evidence_group` are OR alternatives; different groups "
        "are complementary requirements. Recall@5 counts covered groups; Hit/MRR use "
        "the first match to any group.",
        "- Each retriever contributes at most once per retrieval unit. Dense cosine "
        "and BM25 raw scores are not fused.",
        "- No parameter sweep, reranker, answer generation, Agent integration, "
        "re-indexing, or embedding change was used.",
        "",
        "## Overall comparison",
        "",
        "| Metric | Dense | BM25 | RRF |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {metrics['dense'][metric]:.2%} | "
            f"{metrics['bm25'][metric]:.2%} | {metrics['rrf'][metric]:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Per-query comparison",
            "",
            "Movement arrows show changes in first relevant rank; ↑ is better and ↓ is worse.",
            "The exclusive classification is Pareto-based: improved means no worse than "
            "either baseline and strictly better than at least one; degraded is the reverse.",
            "",
            "| Query ID | Gold evidence groups | Dense first relevant rank | BM25 first "
            "relevant rank | RRF first relevant rank | Dense → RRF movement | "
            "BM25 → RRF movement | RRF Top5 units |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for record in records:
        dense_rank = record["dense"]["first_relevant_rank"]
        bm25_rank = record["bm25"]["first_relevant_rank"]
        rrf_rank = record["rrf"]["first_relevant_rank"]
        lines.append(
            f"| {record['query_id']} | {_groups(record['rrf']['gold_groups'])} | "
            f"{_rank(dense_rank)} | {_rank(bm25_rank)} | {_rank(rrf_rank)} | "
            f"{_movement(dense_rank, rrf_rank)} | {_movement(bm25_rank, rrf_rank)} | "
            f"{_ids(record['rrf']['retrieved_ids'])} |"
        )

    rag_020 = next(
        (record for record in records if record["query_id"] == "RAG-020"), None
    )
    if rag_020 is not None:
        lines.extend(
            [
                "",
                "## RAG-020 evidence-group check",
                "",
                f"- Grouping: {_groups(rag_020['rrf']['gold_groups'])}",
                f"- Dense evidence-group Recall@5: {rag_020['dense']['recall_at_5']:.2f}",
                f"- BM25 evidence-group Recall@5: {rag_020['bm25']['recall_at_5']:.2f}",
                f"- RRF evidence-group Recall@5: {rag_020['rrf']['recall_at_5']:.2f}",
            ]
        )

    for title, key in (
        ("Improved queries", "improved"),
        ("Degraded queries", "degraded"),
        ("Unchanged queries", "unchanged"),
        ("Failed queries", "failed"),
    ):
        values = groups[key]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                ", ".join(f"`{value}`" for value in values) if values else "None.",
            ]
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--source-chunks", type=Path, default=DEFAULT_SOURCE_CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dense-report", type=Path, default=DENSE_REPORT_PATH)
    parser.add_argument("--bm25-report", type=Path, default=BM25_REPORT_PATH)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    original_gold, cases = load_and_validate_gold(args.gold, args.source_chunks)
    gold_sha256 = hashlib.sha256(original_gold).hexdigest()
    validate_baseline_gold_hashes(
        gold_sha256, args.gold, args.dense_report, args.bm25_report
    )
    print(
        f"Gold/audit validation passed: queries={len(cases)} sha256={gold_sha256}",
        flush=True,
    )
    if args.validate_only:
        return 0

    load_dotenv(ROOT / ".env")
    dense = DenseRetriever(args.index_dir)
    bm25 = BM25Retriever(args.index_dir)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    records: list[dict[str, Any]] = []
    for case in cases:
        dense_results = dense.retrieve(case["question"], top_k=CANDIDATE_K)
        bm25_results = bm25.retrieve(case["question"], top_k=CANDIDATE_K)
        rrf_results = hybrid.fuse(dense_results, bm25_results, top_k=TOP_K)
        record = {
            "query_id": case["query_id"],
            "dense": score_case(case, dense_results),
            "bm25": score_case(case, bm25_results),
            "rrf": score_case(case, rrf_results),
        }
        record["classification"] = classify_query(
            record["dense"]["first_relevant_rank"],
            record["bm25"]["first_relevant_rank"],
            record["rrf"]["first_relevant_rank"],
        )
        records.append(record)
        print(
            f"{record['query_id']}: dense={record['dense']['first_relevant_rank']} "
            f"bm25={record['bm25']['first_relevant_rank']} "
            f"rrf={record['rrf']['first_relevant_rank']}",
            flush=True,
        )

    if args.gold.read_bytes() != original_gold:
        raise RuntimeError("Gold file changed during evaluation")
    write_report(
        args.report,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        gold_path=args.gold,
        gold_sha256=gold_sha256,
        records=records,
    )
    summary = {
        name: aggregate_metrics([record[name] for record in records])
        for name in ("dense", "bm25", "rrf")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report written to {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
