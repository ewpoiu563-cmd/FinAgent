"""Expand manually reviewed non-zero overrides into complete Top-N judgments.

The output keeps the immutable runtime ranking separate from human relevance
labels and also emits the compact format consumed by score_retrieval.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(prefix: str, candidates: list[str], query_id: str) -> str:
    matches = [value for value in candidates if value.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"{query_id}: prefix {prefix!r} resolved to {len(matches)} candidates: {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--judgments-output", type=Path, required=True)
    parser.add_argument("--score-input-output", type=Path, required=True)
    args = parser.parse_args()

    runs = load_jsonl(args.run)
    spec = json.loads(args.overrides.read_text(encoding="utf-8"))
    overrides = spec["queries"]
    run_ids = {row["query_id"] for row in runs}
    if run_ids != set(overrides):
        raise ValueError(f"query mismatch: run-only={run_ids-set(overrides)}, overrides-only={set(overrides)-run_ids}")

    complete: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    for run in runs:
        query_id = run["query_id"]
        candidate_ids = [item["retrieval_unit_id"] for item in run["retrieved"]]
        resolved: dict[str, tuple[int, list[str]]] = {}
        unresolved: list[tuple[str, int, list[str]]] = []
        for prefix, value in overrides[query_id].items():
            matches = [candidate_id for candidate_id in candidate_ids if candidate_id.startswith(prefix)]
            grade, groups = int(value[0]), list(value[1])
            if grade not in (1, 2, 3) or not groups:
                raise ValueError(f"{query_id}/{prefix}: invalid grade/groups")
            if len(matches) == 1:
                resolved[matches[0]] = (grade, groups)
            elif not matches:
                # Preserve the Gold evidence group even when the refreshed
                # Top-N pool no longer contains this formerly judged unit.
                # score_retrieval then counts the group as un-recalled rather
                # than silently redefining the query as unanswerable.
                unresolved.append((prefix, grade, groups))
            else:
                raise ValueError(f"{query_id}: prefix {prefix!r} resolved to {len(matches)} candidates: {matches}")

        judgments = []
        relevant = []
        for item in run["retrieved"]:
            unit_id = item["retrieval_unit_id"]
            grade, groups = resolved.get(unit_id, (0, []))
            judgments.append({
                "rank": item["rank"], "retrieval_unit_id": unit_id,
                "relevance": grade, "evidence_groups": groups,
            })
            for group in groups:
                relevant.append({
                    "retrieval_unit_id": unit_id,
                    "relevance": grade,
                    "evidence_group": group,
                })
        for prefix, grade, groups in unresolved:
            for group in groups:
                relevant.append({
                    "retrieval_unit_id": f"missing:{prefix}",
                    "relevance": grade,
                    "evidence_group": group,
                })
        if len(judgments) != len(candidate_ids):
            raise AssertionError(f"{query_id}: incomplete pool")
        complete.append({
            "query_id": query_id,
            "reviewer": spec["reviewer"],
            "review_date": spec["review_date"],
            "pool_depth": len(judgments),
            "judgments": judgments,
        })
        score_records.append({
            "query_id": query_id,
            "question": run["question"],
            "difficulty": run.get("difficulty"),
            "relevant": relevant,
            "retrieved": run["retrieved"],
        })

    args.judgments_output.parent.mkdir(parents=True, exist_ok=True)
    args.score_input_output.parent.mkdir(parents=True, exist_ok=True)
    args.judgments_output.write_text(json.dumps({
        "version": spec["version"], "scale": spec["scale"],
        "pooling": "all candidates from the frozen Top-20 production run",
        "records": complete,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    args.score_input_output.write_text(json.dumps(score_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"materialized {len(complete)} queries and {sum(x['pool_depth'] for x in complete)} judgments")


if __name__ == "__main__":
    main()
