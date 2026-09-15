"""Score retrieval routes against the manual (human) pooled qrels ledger v1.

Metric definitions (fixed before scoring):
  * A case is "answerable" when expected_outcome != "insufficient"; the five
    no-answer boundary cases are reported separately (refusal correctness).
  * Each answerable case has one or more required evidence groups (gN=...).
    A group is covered when at least one of its listed chunks appears in Top-k.
  * Recall@k   = covered groups / required groups.
  * Precision@k= (# retrieved in Top-k that belong to any group) / k.
  * MRR        = 1 / rank of the first group-member chunk in the route ranking.
  * NDCG@k     = DCG@k / IDCG@k with gain 2**grade-1, grade 1..3 from the ledger
                 (group grades and background grades), IDCG taken over the
                 sorted graded set truncated at k.
  * Hit@k      = 1 when any group-member chunk appears in Top-k.

The script only reads frozen inputs plus the ledger and writes new report files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROUTES = ("dense", "dense_rerank", "hybrid_rerank")
DEFAULT_FLAGGED = ("010", "013", "026", "038", "046", "048", "050", "074")


def _parse_ledger(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Fields are separated by " | "; group alternatives are separated by a
        # bare "|" without surrounding spaces.
        parts = [part.strip() for part in re.split(r"\s\|\s", line) if part.strip()]
        case_id = parts[0]
        record: dict[str, Any] = {"llm": None, "groups": [], "bg": {}, "note": ""}
        for part in parts[1:]:
            if part.startswith("llm="):
                record["llm"] = part.split("=", 1)[1].strip()
            elif part.startswith("bg="):
                for item in part.split("=", 1)[1].split(","):
                    item = item.strip()
                    if not item:
                        continue
                    alias, grade = item.split(":")
                    record["bg"][alias.strip()] = int(grade)
            elif part.startswith("g") and "=" in part:
                name, body = part.split("=", 1)
                members: list[tuple[str, int]] = []
                for item in body.split("|"):
                    item = item.strip()
                    if not item:
                        continue
                    alias, grade = item.split(":")
                    members.append((alias.strip(), int(grade)))
                record["groups"].append({"name": name.strip(), "members": members})
            elif part.startswith("note="):
                record["note"] = part.split("=", 1)[1].strip()
        cases[case_id] = record
    return cases


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flagged", default=",".join(DEFAULT_FLAGGED))
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    ledger = _parse_ledger(args.ledger)
    flagged = {f"v2-run002-rag-{suffix}" for suffix in args.flagged.split(",") if suffix}

    missing = [case["case_id"] for case in pool if case["case_id"] not in ledger]
    if missing:
        raise SystemExit(f"ledger missing cases: {missing}")

    detail_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type_clean: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aggregates: dict[str, dict[int, dict[str, list[float]]]] = {
        route: {3: defaultdict(list), 5: defaultdict(list)} for route in ROUTES
    }
    clean_aggregates: dict[str, dict[int, dict[str, list[float]]]] = {
        route: {3: defaultdict(list), 5: defaultdict(list)} for route in ROUTES
    }
    no_answer_rows: list[dict[str, Any]] = []

    for case in pool:
        case_id = case["case_id"]
        entry = ledger[case_id]
        by_alias: dict[str, dict[str, Any]] = {}
        for cand in case["candidates"]:
            alias = cand["retrieval_unit_id"][-6:]
            if alias in by_alias:
                raise SystemExit(f"ambiguous alias {alias} in {case_id}")
            by_alias[alias] = cand

        def resolve(alias: str) -> str:
            if alias not in by_alias:
                raise SystemExit(f"{case_id}: alias {alias} not found in review pool")
            return by_alias[alias]["retrieval_unit_id"]

        groups = [
            {"name": group["name"], "members": {resolve(alias) for alias, _ in group["members"]}}
            for group in entry["groups"]
        ]
        grades: dict[str, int] = {}
        for group in entry["groups"]:
            for alias, grade in group["members"]:
                unit = resolve(alias)
                grades[unit] = max(grades.get(unit, 0), grade)
        for alias, grade in entry["bg"].items():
            unit = resolve(alias)
            grades[unit] = max(grades.get(unit, 0), grade)

        llm_rows.append(
            {
                "case_id": case_id,
                "test_type": case["test_type"],
                "difficulty": case.get("difficulty"),
                "expected_outcome": case.get("expected_outcome") or "success",
                "llm_answer_judgment": entry["llm"],
                "flagged_case": case_id in flagged,
                "manual_note": entry["note"],
            }
        )

        answerable = (case.get("expected_outcome") or "success") != "insufficient"
        row: dict[str, Any] = {
            "case_id": case_id,
            "test_type": case["test_type"],
            "difficulty": case.get("difficulty"),
            "flagged_case": case_id in flagged,
            "required_groups": len(groups),
            "llm_answer_judgment": entry["llm"],
            "question": case["question"],
            "reference_answer": case.get("reference_answer"),
            "manual_note": entry["note"],
        }

        if not answerable:
            top_grades = []
            for route in ROUTES:
                ranked = sorted(
                    (c for c in case["candidates"] if route in (c.get("ranks") or {})),
                    key=lambda c: c["ranks"][route],
                )[:5]
                top_grades.append(max([grades.get(c["retrieval_unit_id"], 0) for c in ranked] or [0]))
            no_answer_rows.append(
                {
                    "case_id": case_id,
                    "question": case["question"],
                    "llm_answer_judgment": entry["llm"],
                    "max_grade_top5_by_route": dict(zip(ROUTES, top_grades)),
                    "manual_note": entry["note"],
                }
            )
            detail_rows.append(row)
            continue

        group_members: set[str] = set()
        for group in groups:
            group_members |= group["members"]
        # NDCG measures the ranking of required evidence (grade >= 2) only;
        # background chunks carry no gain. Cases without any required evidence
        # in the pool therefore get NDCG = 0 instead of a degenerate 1.0.
        all_grades = sorted((grade for grade in grades.values() if grade >= 2), reverse=True)

        for route in ROUTES:
            ranked = sorted(
                (c for c in case["candidates"] if route in (c.get("ranks") or {})),
                key=lambda c: c["ranks"][route],
            )
            ranked_ids = [c["retrieval_unit_id"] for c in ranked]
            first_hit = next((i + 1 for i, unit in enumerate(ranked_ids) if unit in group_members), None)
            for k in (3, 5):
                top = ranked_ids[:k]
                covered = sum(1 for group in groups if group["members"] & set(top))
                recall = covered / len(groups) if groups else 0.0
                precision = sum(1 for unit in top if unit in group_members) / k
                gains = [grade if (grade := grades.get(unit, 0)) >= 2 else 0 for unit in top]
                dcg = _dcg(gains)
                idcg = _dcg(all_grades[:k])
                ndcg = dcg / idcg if idcg else 0.0
                metrics = {
                    "recall": recall,
                    "precision": precision,
                    "ndcg": ndcg,
                    "hit": 1.0 if covered else 0.0,
                }
                for name, value in metrics.items():
                    aggregates[route][k][name].append(value)
                    if case_id not in flagged:
                        clean_aggregates[route][k][name].append(value)
                row[f"{route}_recall@{k}"] = round(recall, 4)
                row[f"{route}_precision@{k}"] = round(precision, 4)
                row[f"{route}_ndcg@{k}"] = round(ndcg, 4)
                row[f"{route}_hit@{k}"] = int(bool(covered))
            rr = 1.0 / first_hit if first_hit else 0.0
            row[f"{route}_mrr"] = round(rr, 4)
            aggregates[route][3]["mrr"].append(rr)
            aggregates[route][5]["mrr"].append(rr)
            if case_id not in flagged:
                clean_aggregates[route][3]["mrr"].append(rr)
                clean_aggregates[route][5]["mrr"].append(rr)

        by_type[case["test_type"]].append(row)
        if case_id not in flagged:
            by_type_clean[case["test_type"]].append(row)
        detail_rows.append(row)

    out_dir = args.output_dir
    (out_dir / "retrieval").mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "version": "testrun002-manual-qrels-v1",
        "reviewer_count": 1,
        "caveat": "Single-reviewer manual pooled qrels over union of Top-5 across three routes.",
        "answerable_cases": sum(1 for r in detail_rows if r.get("required_groups") is not None and "llm_answer_judgment" in r and r["case_id"] not in {x["case_id"] for x in no_answer_rows}),
        "no_answer_cases": len(no_answer_rows),
        "flagged_cases": sorted(flagged),
        "metrics_by_route": {},
        "metrics_by_route_excluding_flagged": {},
        "llm_judgments": {},
        "no_answer_review": no_answer_rows,
    }

    for route in ROUTES:
        for label, source in (("metrics_by_route", aggregates), ("metrics_by_route_excluding_flagged", clean_aggregates)):
            bucket: dict[str, Any] = {}
            for k in (3, 5):
                n = len(source[route][k]["recall"])
                bucket[f"k={k}"] = {
                    "n": n,
                    "recall": round(_mean(source[route][k]["recall"]), 4),
                    "precision": round(_mean(source[route][k]["precision"]), 4),
                    "mrr": round(_mean(source[route][k]["mrr"]), 4),
                    "ndcg": round(_mean(source[route][k]["ndcg"]), 4),
                    "hit": round(_mean(source[route][k]["hit"]), 4),
                }
            summary[label][route] = bucket

    llm_counts: dict[str, int] = defaultdict(int)
    for row in llm_rows:
        llm_counts[row["llm_answer_judgment"] or "unjudged"] += 1
    summary["llm_judgments"] = dict(llm_counts)
    summary["llm_judgments_answerable"] = {}
    answerable_ids = {r["case_id"] for r in detail_rows if r.get("required_groups")} - {
        x["case_id"] for x in no_answer_rows
    }
    for grade in ("correct", "partial", "incorrect"):
        summary["llm_judgments_answerable"][grade] = sum(
            1 for row in llm_rows if row["case_id"] in answerable_ids and row["llm_answer_judgment"] == grade
        )
    summary["llm_judgments_answerable"]["n"] = len(answerable_ids)

    (out_dir / "retrieval" / "manual_retrieval_metrics_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8-sig")
            return
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_dir / "retrieval" / "manual_per_question_results_v1.csv", detail_rows)
    write_csv(out_dir / "retrieval" / "manual_llm_judgments_v1.csv", llm_rows)

    type_rows: list[dict[str, Any]] = []
    for test_type, rows in sorted(by_type.items()):
        entry = {"test_type": test_type, "n": len(rows), "n_excluding_flagged": len(by_type_clean.get(test_type, []))}
        for route in ROUTES:
            for k in (3, 5):
                for metric in ("recall", "precision", "ndcg"):
                    entry[f"{route}_{metric}@{k}"] = round(
                        _mean([row[f"{route}_{metric}@{k}"] for row in rows]), 4
                    )
            entry[f"{route}_mrr"] = round(_mean([row[f"{route}_mrr"] for row in rows]), 4)
        entry["llm_correct"] = sum(1 for row in rows if row["llm_answer_judgment"] == "correct")
        entry["llm_partial"] = sum(1 for row in rows if row["llm_answer_judgment"] == "partial")
        entry["llm_incorrect"] = sum(1 for row in rows if row["llm_answer_judgment"] == "incorrect")
        type_rows.append(entry)
    write_csv(out_dir / "retrieval" / "manual_metrics_by_type_v1.csv", type_rows)
    (out_dir / "retrieval" / "manual_per_question_results_v1.json").write_text(
        json.dumps(detail_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["metrics_by_route"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["metrics_by_route_excluding_flagged"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["llm_judgments"], ensure_ascii=False))


if __name__ == "__main__":
    main()
