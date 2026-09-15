"""Run the frozen hard-set retrieval evaluation against a multi-document index.

This module is intentionally an evaluation/reporting adapter.  It does not
change DenseRetriever, BM25Retriever, HybridRetriever, query text, or ranking
parameters.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from eval.evaluate_dense_retrieval import (  # frozen gold/relevance semantics
    aggregate_metrics,
    load_and_validate_gold,
    retrieval_unit_id,
    score_case,
)
from eval.evaluate_hybrid_retrieval import score_case as score_hybrid_case
from rag.bm25_retriever import BM25Retriever
from rag.dense_retriever import DenseRetriever
from rag.hybrid_retriever import HybridRetriever, RRF_K

TOP_K = 5
CANDIDATE_K = 20
DEFAULT_GOLD = ROOT / "eval" / "rag_gold_hard.json"
DEFAULT_SOURCE = ROOT / "outputs" / "chunks" / "doc1_v2.jsonl"
DEFAULT_INDEX = ROOT / "outputs" / "index" / "multi_doc_v1"
METHODS = ("Dense", "BM25", "RRF")
REPORTS = {
    "Dense": ROOT / "DENSE_RETRIEVAL_MULTI_DOC_HARD_EVAL_REPORT.md",
    "BM25": ROOT / "BM25_RETRIEVAL_MULTI_DOC_HARD_EVAL_REPORT.md",
    "RRF": ROOT / "HYBRID_RETRIEVAL_MULTI_DOC_HARD_EVAL_REPORT.md",
}
BASELINE_REPORTS = {
    "Dense": ROOT / "DENSE_RETRIEVAL_HARD_EVAL_REPORT.md",
    "BM25": ROOT / "BM25_RETRIEVAL_HARD_EVAL_REPORT.md",
    "RRF": ROOT / "HYBRID_RETRIEVAL_HARD_EVAL_REPORT.md",
}


def _rank_text(rank: int | None) -> str:
    return str(rank) if rank is not None else "—"


def _ids(values: Sequence[str]) -> str:
    return "<br>".join(f"`{value}`" for value in values)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def annotate(
    case: Mapping[str, Any], results: Sequence[Mapping[str, Any]], *, hybrid: bool = False
) -> dict[str, Any]:
    """Add multi-document diagnostics to the unchanged hard-set scoring record."""

    record = (score_hybrid_case(case, results) if hybrid else score_case(case, results))  # type: ignore[arg-type]
    top5 = list(results[:TOP_K])
    target_doc = case["doc_id"]
    docs = [str(result["doc_id"]) for result in top5]
    record.update(
        {
            "target_doc_id": target_doc,
            "top5_doc_ids": docs,
            "top5_pages": [list(result["pages"]) for result in top5],
            "first_target_doc_rank": next(
                (rank for rank, doc_id in enumerate(docs, 1) if doc_id == target_doc), None
            ),
            "target_doc_at_1": bool(docs and docs[0] == target_doc),
            "target_doc_at_5": target_doc in docs,
            "cross_document_top1_error": bool(docs and docs[0] != target_doc),
        }
    )
    return record


def diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total = len(records)
    return {
        "TargetDoc@1": sum(record["target_doc_at_1"] for record in records) / total,
        "TargetDoc@5": sum(record["target_doc_at_5"] for record in records) / total,
        "CrossDocTop1Error": sum(record["cross_document_top1_error"] for record in records) / total,
    }


def _report(method: str, path: Path, *, evaluated_at: str, gold_sha: str, index_dir: Path,
            records: Sequence[Mapping[str, Any]]) -> None:
    metrics, diagnostic = aggregate_metrics(records), diagnostics(records)
    top1_interference = Counter(
        record["top5_doc_ids"][0] for record in records if record["cross_document_top1_error"]
    )
    lines = [
        f"# {method} Retrieval Multi-document Hard Evaluation Report", "",
        "## Evaluation setup", "",
        f"- Evaluated at (UTC): `{evaluated_at}`",
        f"- Gold Set: `eval/rag_gold_hard.json` ({len(records)} queries, SHA-256 `{gold_sha}`); the file was checked unchanged after retrieval.",
        f"- Index: `{index_dir.resolve().relative_to(ROOT).as_posix()}`.",
        "- Relevance boundary: `parent_chunk_id` when present, otherwise `chunk_id`.",
    ]
    if method == "Dense":
        lines.append("- Frozen retrieval: unchanged Dense Top5; one normal query embedding per query.")
    elif method == "BM25":
        lines.append("- Frozen retrieval: `BM25Okapi` defaults, raw BM25 Top5.")
    else:
        lines.append(f"- Frozen retrieval: Dense Top{CANDIDATE_K} + BM25 Top{CANDIDATE_K}; retrieval-unit normalization; RRF k={RRF_K}; final Top5.")
    lines.extend(["- No query rewrite, document filter, reranker, parameter sweep, document embedding, or index rebuild.", "",
                  "## Overall metrics", "", "| Metric | Value |", "|---|---:|"])
    lines.extend(f"| {name} | {value:.2%} |" for name, value in metrics.items())
    lines.extend(["", "## Multi-document diagnostic metrics", "", "| Metric | Value |", "|---|---:|"])
    lines.extend(f"| {name} | {value:.2%} |" for name, value in diagnostic.items())
    lines.extend(["", "## Per-query results", "",
                  "| Query ID | Top5 retrieval_unit_id | Top5 doc_id | Top5 page | First relevant rank | First target-doc rank | Cross-document Top1 error |",
                  "|---|---|---|---|---:|---:|---:|"])
    for record in records:
        lines.append(
            f"| {record['query_id']} | {_ids(record['retrieved_top5_ids'] if 'retrieved_top5_ids' in record else record['retrieved_ids'])} | "
            f"{_ids(record['top5_doc_ids'])} | {_md(record['top5_pages'])} | "
            f"{_rank_text(record['first_relevant_rank'])} | {_rank_text(record['first_target_doc_rank'])} | "
            f"{int(record['cross_document_top1_error'])} |"
        )
    lines.extend(["", "## Cross-document Top1 interference", ""])
    if top1_interference:
        lines.extend(f"- `{doc_id}`: {count} queries" for doc_id, count in top1_interference.most_common())
    else:
        lines.append("None.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _baseline_metrics(path: Path, method: str) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    values: dict[str, float] = {}
    for metric in ("Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR@5"):
        match = re.search(rf"\| {re.escape(metric)} \| ([0-9.]+)% \|" if method != "RRF"
                          else rf"\| {re.escape(metric)} \| [0-9.]+% \| [0-9.]+% \| ([0-9.]+)% \|", text)
        if not match:
            raise ValueError(f"could not read {metric} from baseline {path}")
        values[metric] = float(match.group(1)) / 100
    return values


def _baseline_ranks(path: Path, method: str) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| RAG-"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        query_id = cells[1]
        raw = cells[5] if method != "RRF" else cells[5]
        ranks[query_id] = None if raw in {"—", ""} else int(raw)
    if len(ranks) != 20:
        raise ValueError(f"could not read all 20 baseline ranks from {path}")
    return ranks


def _comparison(path: Path, records_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    multi_metrics = {method: aggregate_metrics(records) for method, records in records_by_method.items()}
    baseline_metrics = {method: _baseline_metrics(BASELINE_REPORTS[method], method) for method in METHODS}
    lines = ["# Single-document Hard vs Multi-document Hard", "",
             "Baseline reports use the same frozen `eval/rag_gold_hard.json` SHA-256. All changes below switch only the index/metadata source.", "",
             "## Metric comparison", "",
             "| Method | Metric | Single-doc hard | Multi-doc hard | Δ |", "|---|---|---:|---:|---:|"]
    for method in METHODS:
        for metric in multi_metrics[method]:
            old, new = baseline_metrics[method][metric], multi_metrics[method][metric]
            lines.append(f"| {method} | {metric} | {old:.2%} | {new:.2%} | {new-old:+.2%} |")
    lines.extend(["", "## Per-method corpus-expansion impact", ""])
    for method in METHODS:
        old = _baseline_ranks(BASELINE_REPORTS[method], method)
        new = {record["query_id"]: record["first_relevant_rank"] for record in records_by_method[method]}
        newly_failed = [qid for qid in old if old[qid] is not None and new[qid] is None]
        declines = [qid for qid in old if old[qid] is not None and new[qid] is not None and new[qid] - old[qid] >= 2]
        unaffected = [qid for qid in old if old[qid] == new[qid]]
        cross = [record["query_id"] for record in records_by_method[method] if record["cross_document_top1_error"]]
        interference = Counter(record["top5_doc_ids"][0] for record in records_by_method[method] if record["cross_document_top1_error"])
        lines.extend([f"### {method}", "",
                      f"- 新增失败 query: {', '.join(f'`{x}`' for x in newly_failed) or '无'}",
                      f"- rank 明显下降（首个相关 rank 下降 ≥2）: {', '.join(f'`{x}`' for x in declines) or '无'}",
                      f"- cross-document Top1 error query: {', '.join(f'`{x}`' for x in cross) or '无'}",
                      "- 主要 Top1 干扰 doc_id: " + (", ".join(f"`{doc}` ({count})" for doc, count in interference.most_common()) or "无"),
                      f"- corpus 扩大后首个相关 rank 完全不变: {', '.join(f'`{x}`' for x in unaffected) or '无'}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--source-chunks", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--comparison-report", type=Path, default=ROOT / "SINGLE_DOC_VS_MULTI_DOC_HARD_COMPARISON.md")
    args = parser.parse_args(argv)
    original_gold, cases = load_and_validate_gold(args.gold, args.source_chunks)
    gold_sha = hashlib.sha256(original_gold).hexdigest()
    load_dotenv(ROOT / ".env")
    dense, bm25 = DenseRetriever(args.index_dir), BM25Retriever(args.index_dir)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
    outputs: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for case in cases:
        dense20 = dense.retrieve(case["question"], top_k=CANDIDATE_K)
        bm2520 = bm25.retrieve(case["question"], top_k=CANDIDATE_K)
        outputs["Dense"].append(annotate(case, dense20[:TOP_K]))
        outputs["BM25"].append(annotate(case, bm2520[:TOP_K]))
        outputs["RRF"].append(annotate(case, hybrid.fuse(dense20, bm2520, top_k=TOP_K), hybrid=True))
        print(f"{case['query_id']}: Dense={outputs['Dense'][-1]['first_relevant_rank']} BM25={outputs['BM25'][-1]['first_relevant_rank']} RRF={outputs['RRF'][-1]['first_relevant_rank']}", flush=True)
    if args.gold.read_bytes() != original_gold:
        raise RuntimeError("Gold file changed during evaluation")
    now = datetime.now(timezone.utc).isoformat()
    for method in METHODS:
        _report(method, REPORTS[method], evaluated_at=now, gold_sha=gold_sha, index_dir=args.index_dir, records=outputs[method])
    _comparison(args.comparison_report, outputs)
    print(json.dumps({method: {**aggregate_metrics(records), **diagnostics(records)} for method, records in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
