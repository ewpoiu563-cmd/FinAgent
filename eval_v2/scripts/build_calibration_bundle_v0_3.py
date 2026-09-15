"""Assemble the v0.3 calibration metric bundle from frozen artifacts.

Reads the E2E run, its Trace, the objective scores, and the three retrieval
scoring outputs (Dense-only, Dense+rerank, Dense+BM25+RRF+rerank). Writes one
JSON bundle so the report and any downstream summary read identical numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def suite_of(case_id: str) -> str:
    parts = case_id.split("-")
    return parts[2] if len(parts) > 2 else "unknown"


def latency_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_s": round(fmean(values) / 1000, 3),
        "p50_s": round(percentile(values, 0.50) / 1000, 3),
        "p95_s": round(percentile(values, 0.95) / 1000, 3),
        "max_s": round(max(values) / 1000, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--retrieval", type=str, nargs="+", required=True,
                        help="label=path pairs, e.g. dense_only=.../retrieval_metrics.json")
    parser.add_argument("--human", type=Path, help="aggregated human summary JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = load_jsonl(args.run)
    run_case = {row["run_id"]: row["case_id"] for row in runs}
    scores = load_json(args.scores)
    summary = load_json(args.summary)

    web_runs: set[str] = set()
    planned_web: set[str] = set()
    for event in load_jsonl(args.trace):
        if event.get("event_type", "").startswith("tool.call.") and event.get("source") == "web":
            web_runs.add(event["run_id"])
        if event.get("event_type") == "planning.source_plan.completed":
            sources = event.get("attributes", {}).get("required_sources") or []
            if "web" in sources:
                planned_web.add(event["run_id"])

    web_case_ids = {
        run_case[run_id]
        for run_id in (web_runs | planned_web)
        if run_id in run_case
    }
    non_web = [row for row in scores if row["case_id"] not in web_case_ids]
    web_involved = [row for row in scores if row["case_id"] in web_case_ids]

    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_suite[suite_of(row["case_id"])].append(row)

    scalar_rows = [row for row in scores if (row.get("gold_scalar_count") or 0) > 0]
    sql_rows = [row for row in scores if "sql" in (row.get("used_tool_sources") or "")]

    retrieval_blocks: dict[str, Any] = {}
    for item in args.retrieval:
        label, path = item.split("=", 1)
        metrics = load_json(Path(path))
        by_k = {int(row["k"]): row for row in metrics["by_k"]}
        retrieval_blocks[label] = {
            "query_count": metrics["by_k"][0]["query_count"],
            "answerable_count": metrics["by_k"][0]["answerable_count"],
            "no_answer_count": metrics["by_k"][0]["no_answer_count"],
            "by_k": {
                str(k): {
                    "hit": row["hit"], "recall": row["recall"], "precision": row["precision"],
                    "mrr": row["mrr"], "ndcg": row["ndcg"],
                    "no_answer_false_positive_rate": row["no_answer_false_positive_rate"],
                    "mean_returned": row["mean_returned"],
                }
                for k, row in sorted(by_k.items())
            },
            "selected_joint_operating_point": metrics.get("selected_joint_operating_point"),
        }

    def at_k(label: str, k: int, field: str) -> float | None:
        block = retrieval_blocks.get(label, {}).get("by_k", {}).get(str(k))
        return block.get(field) if block else None

    lift: dict[str, Any] = {}
    for k in (1, 3, 5, 10, 20):
        row: dict[str, Any] = {}
        for field in ("recall", "precision", "mrr", "ndcg"):
            dense = at_k("dense_only", k, field)
            hybrid = at_k("hybrid_rerank", k, field)
            if dense is not None and hybrid is not None:
                row[field] = {
                    "dense_only": dense,
                    "hybrid_rerank": hybrid,
                    "absolute_delta": hybrid - dense,
                }
        lift[str(k)] = row

    bundle = {
        "generated_from": {
            "run": str(args.run),
            "trace": str(args.trace),
            "scores": str(args.scores),
        },
        "run_count": len(runs),
        "tool_routing": {
            "route_exact_accuracy": summary["route_exact_accuracy"],
            "route_macro_precision": summary["route_macro_precision"],
            "route_macro_recall": summary["route_macro_recall"],
            "source_boundary_compliance": summary["source_boundary_compliance"],
            "runtime_completion_rate": summary["runtime_completion_rate"],
            "route_exact_by_suite": {
                suite: round(fmean(row["route_exact"] for row in rows), 4)
                for suite, rows in sorted(by_suite.items())
            },
        },
        "sql": {
            "execution_success_rate": summary["sql_execution_success_rate"],
            "read_only_rate": summary["sql_read_only_rate"],
            "gold_scalar_hit_rate": (
                round(fmean(row["gold_scalar_recall"] for row in scalar_rows), 4)
                if scalar_rows else None
            ),
            "gold_scalar_cases": len(scalar_rows),
        },
        "latency": {
            "all": latency_block(scores),
            "non_web": latency_block(non_web),
            "web_involved": latency_block(web_involved),
            "non_web_case_count": len(non_web),
            "web_involved_case_count": len(web_involved),
            "p95_by_suite_s": {
                suite: (latency_block(rows).get("p95_s"))
                for suite, rows in sorted(by_suite.items())
            },
        },
        "retrieval": retrieval_blocks,
        "retrieval_lift_hybrid_vs_dense_only": lift,
    }
    if args.human is not None:
        bundle["human"] = load_json(args.human)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "run_count": bundle["run_count"],
        "route_exact": bundle["tool_routing"]["route_exact_accuracy"],
        "non_web_p95_s": bundle["latency"]["non_web"].get("p95_s"),
        "all_p95_s": bundle["latency"]["all"].get("p95_s"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
