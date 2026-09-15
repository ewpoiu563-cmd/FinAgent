"""Validate and aggregate the locked single-reviewer human judgments."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


DIMS = ("correctness", "completeness", "groundedness", "citation_entailment")


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def suite(case_id: str) -> str:
    for label in ("direct", "sql", "rag", "web", "hybrid", "dialog", "failure"):
        if f"-{label}-" in case_id:
            return label
    return "unknown"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"count": len(rows)}
    for dim in DIMS:
        values = [row[dim] for row in rows if row[dim] is not None]
        result[dim] = {"mean_0_to_4": fmean(values) if values else None, "normalized": fmean(values) / 4 if values else None, "rated_count": len(values)}
    successes = sum(row["e2e_success"] for row in rows)
    result["e2e_success_rate"] = successes / len(rows) if rows else 0.0
    result["e2e_success_count"] = successes
    result["e2e_wilson_95"] = wilson(successes, len(rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=60,
        help="Expected number of unique judgment records (default: 60 for the full calibration set).",
    )
    args = parser.parse_args()
    artifact = json.loads(args.judgments.read_text(encoding="utf-8"))
    rows = artifact["records"]
    objective = {row["case_id"]: row for row in json.loads(args.objective.read_text(encoding="utf-8"))}
    if len(rows) != args.expected_count or len({row["case_id"] for row in rows}) != args.expected_count:
        raise ValueError(f"human evaluation must contain exactly {args.expected_count} unique cases")
    if set(objective) != {row["case_id"] for row in rows}:
        raise ValueError("human/objective case sets differ")
    for row in rows:
        for dim in DIMS:
            value = row.get(dim)
            if value is not None and value not in range(5):
                raise ValueError(f"{row['case_id']}/{dim}: invalid score")
        if row.get("e2e_success") not in (0, 1):
            raise ValueError(f"{row['case_id']}: invalid E2E score")
        row["difficulty"] = objective[row["case_id"]]["difficulty"]
        row["suite"] = suite(row["case_id"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    difficulties: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["suite"]].append(row); difficulties[row["difficulty"]].append(row)
    report = {
        "version": artifact["version"], "reviewer_count": 1,
        "release_caveat": "Single-reviewer calibration; add a blinded second reviewer and adjudicate disagreements before a release gate.",
        "overall": summarize(rows),
        "by_suite": {key: summarize(value) for key, value in sorted(groups.items())},
        "by_difficulty": {key: summarize(value) for key, value in sorted(difficulties.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
