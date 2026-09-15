"""Materialize complete zero-relevance pools for verified no-answer queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--judgments-output", type=Path, required=True)
    parser.add_argument("--score-input-output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = {row["case_id"]: row for row in json.loads(args.gold.read_text(encoding="utf-8"))}
    complete, score_records = [], []
    for run in runs:
        case = gold[run["query_id"]]
        if case.get("gold_status") != "verified_no_answer" or case.get("evidence"):
            raise ValueError(f"{run['query_id']}: not a verified empty-relevance case")
        judgments = [{
            "rank": item["rank"], "retrieval_unit_id": item["retrieval_unit_id"],
            "relevance": 0, "evidence_groups": [],
        } for item in run["retrieved"]]
        complete.append({
            "query_id": run["query_id"], "reviewer": "Codex/manual-scope-review",
            "review_date": "2026-09-13", "pool_depth": len(judgments),
            "judgment_basis": "The requested dated fact is outside the bounded source; historical or adjacent chunks cannot entail it.",
            "judgments": judgments,
        })
        score_records.append({
            "query_id": run["query_id"], "question": run["question"],
            "difficulty": run["difficulty"], "relevant": [], "retrieved": run["retrieved"],
        })
    args.judgments_output.parent.mkdir(parents=True, exist_ok=True)
    args.score_input_output.parent.mkdir(parents=True, exist_ok=True)
    args.judgments_output.write_text(json.dumps({"records": complete}, ensure_ascii=False, indent=2), encoding="utf-8")
    args.score_input_output.write_text(json.dumps(score_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"materialized {len(complete)} no-answer pools and {sum(x['pool_depth'] for x in complete)} judgments")


if __name__ == "__main__":
    main()
