"""Dense-only retrieval baseline for the v2 RAG calibration set.

Evaluation-only helper. It reuses ``HybridRetriever``'s candidate/unit
normalization but feeds RRF only the Dense list, so the produced
``retrieval_unit_id`` space is identical to the hybrid run and the frozen Gold
judgments can be applied unchanged. Output is a score-input file that
``score_retrieval.py`` consumes directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.hybrid_retriever import HybridRetriever  # noqa: E402
from rag.reranker import RerankerRetriever, candidate_text  # noqa: E402


def compact(candidate: Mapping[str, Any], score: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "retrieval_unit_id": candidate.get("retrieval_unit_id"),
        "doc_id": candidate.get("doc_id"),
        "pages": candidate.get("pages", []),
        "score": float(score),
        "rrf_score": candidate.get("rrf_score"),
        "text": candidate.get("text", ""),
    }


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else value["records"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--judgments",
        type=Path,
        required=True,
        help="existing scored input supplying frozen relevance judgments per query_id",
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="also apply the production reranker on top of the Dense candidates",
    )
    args = parser.parse_args()

    judgments = {
        row["query_id"]: row.get("relevant", [])
        for row in load_json(args.judgments)
    }
    cases: list[dict[str, Any]] = []
    for path in args.gold:
        cases.extend(load_json(path))

    retriever = HybridRetriever(args.index)
    reranker = RerankerRetriever(retriever) if args.rerank else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for case in cases:
        query_id = case["case_id"]
        if query_id not in judgments:
            raise ValueError(f"missing frozen judgments for {query_id}")
        scope = case.get("scope_doc_ids") or sorted(
            {item["doc_id"] for item in case.get("evidence", [])}
        )
        started = perf_counter()
        dense = retriever.dense_retriever.retrieve(
            case["question"], top_k=args.top_k, allowed_doc_ids=scope
        )
        # Dense-only list through the same unit normalization used by RRF.
        units = retriever.fuse(dense, [], top_k=args.top_k)
        if reranker is not None:
            scores = reranker.reranker_client.rerank(
                case["question"], [candidate_text(unit) for unit in units]
            )
            ranked = sorted(zip(units, scores), key=lambda item: -float(item[1]))
        else:
            ranked = [(unit, float(unit.get("rrf_score") or 0.0)) for unit in units]
        records.append({
            "query_id": query_id,
            "question": case["question"],
            "difficulty": case["difficulty"],
            "relevant": judgments[query_id],
            "retrieved": [
                compact(unit, score, rank)
                for rank, (unit, score) in enumerate(ranked, 1)
            ],
            "baseline": "dense_rerank" if args.rerank else "dense_only",
            "scope_doc_ids": scope,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        })
        print(f"[{len(records)}/{len(cases)}] {query_id} -> {len(ranked)} units", flush=True)

    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
