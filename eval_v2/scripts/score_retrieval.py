"""Deterministic retrieval metrics and Precision-Recall operating-point search.

Input is JSON or JSONL. Each record follows schemas/retrieval_run.schema.json.
This scorer has no dependency on production retrieval code and never modifies
Gold or runtime artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_KS = (1, 3, 5, 10, 20)


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        records = value if isinstance(value, list) else value.get("records", [])
    if not isinstance(records, list):
        raise ValueError("retrieval input must contain a list of records")
    return records


def _judgments(record: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, set[str]]]:
    grades: dict[str, int] = {}
    groups_by_unit: dict[str, set[str]] = defaultdict(set)
    for item in record.get("relevant", []):
        unit_id = str(item["retrieval_unit_id"])
        grade = int(item["relevance"])
        if grade < 1 or grade > 3:
            raise ValueError(f"relevance must be 1..3 for {record.get('query_id')}")
        grades[unit_id] = max(grades.get(unit_id, 0), grade)
        groups_by_unit[unit_id].add(str(item["evidence_group"]))
    return grades, groups_by_unit


def _dcg(grades: Sequence[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def score_query(record: Mapping[str, Any], k: int, *, threshold: float | None = None) -> dict[str, float]:
    grades, groups_by_unit = _judgments(record)
    gold_groups = {group for groups in groups_by_unit.values() for group in groups}
    retrieved = list(record.get("retrieved", []))
    if threshold is not None:
        retrieved = [item for item in retrieved if float(item["score"]) >= threshold]
    retrieved = retrieved[:k]
    ranked_ids = [str(item["retrieval_unit_id"]) for item in retrieved]
    relevance = [grades.get(unit_id, 0) for unit_id in ranked_ids]
    relevant_flags = [grade > 0 for grade in relevance]
    covered = {group for unit_id in ranked_ids for group in groups_by_unit.get(unit_id, set())}
    first_rank = next((rank for rank, flag in enumerate(relevant_flags, 1) if flag), None)
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    answerable = bool(gold_groups)
    return {
        "answerable": float(answerable),
        "returned": float(len(retrieved)),
        "hit": float(bool(first_rank)) if answerable else 0.0,
        "recall": len(covered) / len(gold_groups) if answerable else 0.0,
        "precision": sum(relevant_flags) / len(retrieved) if retrieved else (1.0 if not answerable else 0.0),
        "rr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": _dcg(relevance) / ideal_dcg if ideal_dcg else 0.0,
        "false_positive": float(not answerable and bool(retrieved)),
    }


def aggregate_at_k(records: Sequence[Mapping[str, Any]], k: int, *, threshold: float | None = None) -> dict[str, float]:
    rows = [score_query(record, k, threshold=threshold) for record in records]
    answerable = [row for row in rows if row["answerable"]]
    no_answer = [row for row in rows if not row["answerable"]]
    mean = lambda key, subset: fmean(row[key] for row in subset) if subset else 0.0
    return {
        "k": k,
        "threshold": threshold,
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "hit": mean("hit", answerable),
        "recall": mean("recall", answerable),
        "precision": mean("precision", answerable),
        "mrr": mean("rr", answerable),
        "ndcg": mean("ndcg", answerable),
        "no_answer_false_positive_rate": mean("false_positive", no_answer),
        "mean_returned": mean("returned", rows),
    }


def aggregate_by_difficulty(
    records: Sequence[Mapping[str, Any]], k: int, *, threshold: float | None = None
) -> list[dict[str, Any]]:
    labels = sorted({str(record.get("difficulty") or "unspecified") for record in records})
    rows = []
    for label in labels:
        subset = [record for record in records if str(record.get("difficulty") or "unspecified") == label]
        row: dict[str, Any] = aggregate_at_k(subset, k, threshold=threshold)
        row["difficulty"] = label
        rows.append(row)
    return rows


def threshold_candidates(records: Sequence[Mapping[str, Any]], points: int = 101) -> list[float]:
    scores = sorted({float(item["score"]) for record in records for item in record.get("retrieved", [])})
    if not scores:
        return []
    # Include both endpoints of the decision path: reject everything and
    # accept everything. Without these sentinels, AUPRC depends on the sampled
    # score range and can omit the zero-recall endpoint.
    epsilon = max(1e-12, abs(scores[-1]) * 1e-12)
    sentinels = [scores[-1] + epsilon, scores[0] - epsilon]
    if len(scores) <= points:
        return sorted(set(scores + sentinels))
    positions = {round(i * (len(scores) - 1) / (points - 1)) for i in range(points)}
    return sorted(set([scores[position] for position in positions] + sentinels))


def precision_recall_curve(records: Sequence[Mapping[str, Any]], k: int, *, points: int = 101) -> list[dict[str, float]]:
    curve = [aggregate_at_k(records, k, threshold=threshold) for threshold in threshold_candidates(records, points)]
    return sorted(curve, key=lambda row: (row["recall"], row["precision"], -float(row["threshold"])))


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def choose_operating_point(
    curve: Iterable[Mapping[str, float]],
    *,
    recall_floor: float = 0.95,
    max_no_answer_fpr: float = 0.05,
    beta: float = 2.0,
) -> dict[str, float] | None:
    feasible = [
        dict(row)
        for row in curve
        if row["recall"] >= recall_floor
        and row["no_answer_false_positive_rate"] <= max_no_answer_fpr
    ]
    if not feasible:
        return None
    for row in feasible:
        row["f_beta"] = f_beta(row["precision"], row["recall"], beta)
    return max(feasible, key=lambda row: (row["f_beta"], row["precision"], -row["mean_returned"], row["threshold"]))


def difficulty_recall_is_feasible(
    records: Sequence[Mapping[str, Any]], k: int, threshold: float, floor: float
) -> bool:
    rows = aggregate_by_difficulty(records, k, threshold=threshold)
    answerable_rows = [row for row in rows if row["answerable_count"]]
    return all(row["recall"] >= floor for row in answerable_rows)


def area_under_pr_curve(curve: Sequence[Mapping[str, float]]) -> float:
    """Trapezoidal macro AUPRC after keeping the best precision per recall."""
    envelope: dict[float, float] = {}
    for row in curve:
        recall, precision = float(row["recall"]), float(row["precision"])
        envelope[recall] = max(envelope.get(recall, 0.0), precision)
    points = sorted(envelope.items())
    if not points:
        return 0.0
    area = 0.0
    prior_recall, prior_precision = (0.0, points[0][1])
    for recall, precision in points:
        area += (recall - prior_recall) * (precision + prior_precision) / 2
        prior_recall, prior_precision = recall, precision
    return area


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--curve-k", type=int, default=20)
    parser.add_argument("--curve-points", type=int, default=101)
    parser.add_argument("--recall-floor", type=float, default=0.95)
    parser.add_argument("--difficulty-recall-floor", type=float, default=0.90)
    parser.add_argument("--max-no-answer-fpr", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=2.0)
    args = parser.parse_args()

    records = _load_records(args.input)
    by_k = [aggregate_at_k(records, k) for k in args.ks]
    curves: list[dict[str, float]] = []
    operating_points = []
    for k in args.ks:
        curve_for_k = precision_recall_curve(records, k, points=args.curve_points)
        curves.extend(curve_for_k)
        selected_for_k = choose_operating_point(
            curve_for_k, recall_floor=args.recall_floor,
            max_no_answer_fpr=args.max_no_answer_fpr, beta=args.beta,
        )
        if selected_for_k and difficulty_recall_is_feasible(
            records, k, float(selected_for_k["threshold"]), args.difficulty_recall_floor
        ):
            selected_for_k["difficulty_recall_floor"] = args.difficulty_recall_floor
            operating_points.append(selected_for_k)
    selected = max(
        operating_points,
        key=lambda row: (row["f_beta"], row["precision"], -row["mean_returned"], -row["k"]),
        default=None,
    )
    curve = [row for row in curves if row["k"] == args.curve_k]
    difficulty_rows = [row for k in args.ks for row in aggregate_by_difficulty(records, k)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_metrics.json").write_text(
        json.dumps({
            "by_k": by_k,
            "pr_auc_by_k": {str(k): area_under_pr_curve([row for row in curves if row["k"] == k]) for k in args.ks},
            "feasible_operating_points_by_k": operating_points,
            "selected_joint_operating_point": selected,
            "selection_is_provisional": len(records) < 60 or sum(not bool(_judgments(record)[0]) for record in records) < 10,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "metrics_by_k.csv", by_k)
    _write_csv(args.output_dir / "precision_recall_curve.csv", curve)
    _write_csv(args.output_dir / "precision_recall_curves_all_k.csv", curves)
    _write_csv(args.output_dir / "metrics_by_difficulty.csv", difficulty_rows)


if __name__ == "__main__":
    main()
