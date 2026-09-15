"""Run the current RAG candidate/reranker stack for v2 calibration.

This is an evaluation-only runner. It preserves all 20 reranker scores so K
and threshold sweeps can be replayed without repeated paid calls. It does not
change production's candidate-K=20/final-K=5 behavior.
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

from rag.hybrid_retriever import HybridRetriever
from rag.reranker import RerankerRetriever, candidate_text


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]
    retriever = HybridRetriever(args.index)
    reranker = RerankerRetriever(retriever)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            started = perf_counter()
            record: dict[str, Any] = {
                "query_id": case["case_id"],
                "question": case["question"],
                "difficulty": case["difficulty"],
                "scope_doc_ids": case.get("scope_doc_ids") or sorted({item["doc_id"] for item in case.get("evidence", [])}),
                "run_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                candidates = retriever.retrieve(
                    case["question"],
                    top_k=20,
                    candidate_k=20,
                    allowed_doc_ids=record["scope_doc_ids"],
                )
                scores = reranker.reranker_client.rerank(
                    case["question"], [candidate_text(candidate) for candidate in candidates]
                )
                ranked = sorted(
                    zip(candidates, scores),
                    key=lambda item: -float(item[1]),
                )
                record["success"] = True
                record["retrieved"] = [compact(candidate, score, rank) for rank, (candidate, score) in enumerate(ranked, 1)]
            except Exception as error:
                record.update(success=False, error_type=type(error).__name__, error=str(error), retrieved=[])
            record["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
