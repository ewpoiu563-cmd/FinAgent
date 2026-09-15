"""Score objective routing, source-boundary, SQL and scalar-answer signals.

Qualitative answer dimensions deliberately remain null for human review; this
script never substitutes string overlap for Correctness or Groundedness.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from urllib.parse import urlparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else value["records"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_sources(case: dict[str, Any], path: Path) -> set[str]:
    route = case.get("expected_route")
    if route is None:
        name = path.name
        route = ["sql"] if name.startswith("sql_") else ["rag"] if name.startswith("rag_") else ["web"]
    return {"direct" if source == "local_compute" else source for source in route}


def normalize(value: Any) -> str:
    if isinstance(value, float):
        value = format(value, ".12g")
    return re.sub(r"[\s,，：:年月日%％元只（）()\[\]\"']", "", str(value)).casefold()


def scalar_values(case: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for row in case.get("gold_rows", []):
        values.extend(value for value in row.values() if value is not None)
    for item in case.get("derived_results", []):
        values.extend(value for key, value in item.items() if key in {"higher_entity", "lower_entity", "difference"})
    for item in case.get("expected_numeric", []):
        values.append(item["value"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gold: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in args.gold:
        for case in load_json(path):
            gold[case["case_id"]] = (case, path)
    runs = load_jsonl(args.runs)
    events_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in load_jsonl(args.trace):
        events_by_run[event["run_id"]].append(event)

    rows = []
    for run in runs:
        case, source_path = gold[run["case_id"]]
        events = events_by_run.get(run["run_id"], [])
        plan = next((e for e in events if e["event_type"] == "planning.source_plan.completed"), None)
        actual_sources = set(plan["attributes"].get("required_sources", [])) if plan else set()
        expected = expected_sources(case, source_path)
        forbidden = set(case.get("forbidden_sources", []))
        tool_events = [e for e in events if e["event_type"] in {"tool.call.completed", "tool.call.failed"}]
        used_sources = {e.get("source") for e in tool_events if e.get("source")}
        sql_calls = [e for e in tool_events if e.get("source") == "sql"]
        sql_outputs = [e.get("attributes", {}).get("output", {}) for e in sql_calls]
        generated = [str(output.get("generated_sql") or "") for output in sql_outputs]
        sql_read_only = [
            bool(re.match(r"^\s*(select|with)\b", statement, re.I))
            and not bool(re.search(r"\b(insert|update|delete|drop|alter|create|replace|truncate)\b", statement, re.I))
            for statement in generated if statement
        ]
        scalars = scalar_values(case)
        normalized_answer = normalize(run.get("answer", ""))
        scalar_hits = [normalize(value) in normalized_answer for value in scalars]
        answer_urls = re.findall(r"https?://[^\s；，)）\]]+", run.get("answer", ""))
        allowed_domains = case.get("allowed_domains", [])
        allowed_url_flags = [
            any((urlparse(url).hostname or "").casefold().endswith(domain.casefold()) for domain in allowed_domains)
            for url in answer_urls
        ] if allowed_domains else []
        route_tp = len(actual_sources & expected)
        precision = route_tp / len(actual_sources) if actual_sources else 0.0
        recall = route_tp / len(expected) if expected else 1.0
        row = {
            "case_id": run["case_id"], "difficulty": run.get("difficulty"),
            "runtime_status": run["status"], "latency_ms": run["latency_ms"],
            "expected_sources": ",".join(sorted(expected)), "actual_sources": ",".join(sorted(actual_sources)),
            "used_tool_sources": ",".join(sorted(used_sources)),
            "route_exact": int(actual_sources == expected),
            "route_precision": precision, "route_recall": recall,
            "source_boundary_ok": int(not bool((actual_sources | used_sources) & forbidden)),
            "answer_citation_count": len(answer_urls),
            "official_domain_citation_rate": fmean(allowed_url_flags) if allowed_url_flags else (0.0 if allowed_domains else None),
            "gold_scalar_count": len(scalars),
            "gold_scalar_recall": sum(scalar_hits) / len(scalar_hits) if scalar_hits else None,
            "sql_call_count": len(sql_calls),
            "sql_execution_success_rate": (
                sum(bool(output.get("success")) for output in sql_outputs) / len(sql_outputs) if sql_outputs else None
            ),
            "sql_read_only_rate": fmean(sql_read_only) if sql_read_only else None,
            "correctness": None, "completeness": None, "groundedness": None,
            "citation_entailment": None, "e2e_human_success": None,
            "answer": run.get("answer", ""),
        }
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "objective_scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "human_review_sheet.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "run_count": len(rows),
        "runtime_completion_rate": fmean(row["runtime_status"] == "completed" for row in rows),
        "route_exact_accuracy": fmean(row["route_exact"] for row in rows),
        "route_macro_precision": fmean(row["route_precision"] for row in rows),
        "route_macro_recall": fmean(row["route_recall"] for row in rows),
        "source_boundary_compliance": fmean(row["source_boundary_ok"] for row in rows),
        "sql_execution_success_rate": fmean(
            row["sql_execution_success_rate"] for row in rows if row["sql_execution_success_rate"] is not None
        ) if any(row["sql_execution_success_rate"] is not None for row in rows) else None,
        "sql_read_only_rate": fmean(row["sql_read_only_rate"] for row in rows if row["sql_read_only_rate"] is not None)
        if any(row["sql_read_only_rate"] is not None for row in rows) else None,
        "qualitative_metrics_status": "pending_human_review",
    }
    (args.output_dir / "objective_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
