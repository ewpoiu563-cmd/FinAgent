"""Deterministically score executed SQL result sets against frozen SQL Gold.

This measures result-set correctness, not whether an answer generator restated
every derived relation.  Values are compared with the Gold tolerance and rows
are matched in declared order or as an unordered multiset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normal(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        # Frozen SQLite dates and runtime presentation may differ only by
        # separators/time suffix; retain all other textual distinctions.
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:\s+00:00:00)?", value):
            return value[:10].replace("-", "")
        return value
    return value


def _same(left: Any, right: Any, tolerance: float) -> bool:
    left, right = _normal(left), _normal(right)
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


_COLUMN_ALIASES = {
    "first_date": "起始日期",
    "first_nav": "起始单位净值",
    "last_date": "结束日期",
    "last_nav": "结束单位净值",
    "growth_rate_pct": "增长率百分比",
    "管理人去重数": "管理人数",
    "缺失数量": "管理费率缺失数",
}


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map deterministic SQL aliases back to their frozen-Gold names."""

    return {_COLUMN_ALIASES.get(column, column): value for column, value in row.items()}


def _required_gold_columns(case: dict[str, Any], rows: list[dict[str, Any]]) -> set[str]:
    """Return answer fields, excluding entity IDs already supplied by the prompt.

    The Gold keeps predicate identifiers so a case is reproducible.  They are
    not required answer fields when the executed SQL correctly omits them from
    its SELECT list.  If an identifier *is* returned, it remains required and
    is checked like every other field.
    """

    columns = {column for row in case["gold_rows"] for column in row}
    actual_columns = {column for row in rows for column in _canonical_row(row)}
    question = case["question"]
    for identifier in ("基金代码", "股票代码"):
        values = {str(row[identifier]) for row in case["gold_rows"] if identifier in row}
        if identifier in columns and identifier not in actual_columns and values and all(value in question for value in values):
            columns.remove(identifier)
    return columns


def _row_matches(
    actual: dict[str, Any], expected: dict[str, Any], required_columns: set[str], tolerance: float
) -> tuple[bool, list[str]]:
    actual = _canonical_row(actual)
    missing = [column for column in required_columns if column not in actual]
    if missing:
        return False, []
    return (
        all(_same(actual[column], expected[column], tolerance) for column in required_columns),
        sorted(required_columns),
    )


def _score_case(case: dict[str, Any], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    semantics = case["answer_semantics"]
    gold_rows = case["gold_rows"]
    output = outputs[-1] if outputs else None
    if not output or not output.get("success"):
        return {"case_id": case["case_id"], "passed": False, "reason": "no_successful_sql_output"}
    rows = output.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return {"case_id": case["case_id"], "passed": False, "reason": "invalid_sql_rows"}
    if semantics["kind"] == "empty_rows":
        passed = len(rows) == 0
        return {"case_id": case["case_id"], "passed": passed, "reason": "empty_rows" if passed else "expected_empty_rows", "actual_row_count": len(rows)}
    tolerance = float(semantics.get("numeric_tolerance", 0.0))
    if len(rows) != len(gold_rows):
        return {"case_id": case["case_id"], "passed": False, "reason": "row_count_mismatch", "expected_row_count": len(gold_rows), "actual_row_count": len(rows)}
    remaining = list(rows)
    required_columns = _required_gold_columns(case, rows)
    matched_columns: set[str] = set()
    for position, expected in enumerate(gold_rows):
        candidates = range(len(remaining)) if not semantics.get("ordered") else (0,)
        match_index = next(
            (index for index in candidates if _row_matches(remaining[index], expected, required_columns, tolerance)[0]),
            None,
        )
        if match_index is None:
            return {"case_id": case["case_id"], "passed": False, "reason": "row_value_mismatch", "gold_row_index": position}
        _, columns = _row_matches(remaining[match_index], expected, required_columns, tolerance)
        matched_columns.update(columns)
        remaining.pop(match_index)
    return {
        "case_id": case["case_id"], "passed": True, "reason": "result_set_match",
        "matched_gold_columns": sorted(matched_columns), "actual_row_count": len(rows),
        "derived_relations_scored": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = _json(args.gold)
    run_to_case = {row["run_id"]: row["case_id"] for row in _jsonl(args.runs)}
    outputs: dict[str, list[dict[str, Any]]] = {}
    for event in _jsonl(args.trace):
        if event.get("event_type") != "tool.call.completed" or event.get("source") != "sql":
            continue
        case_id = run_to_case.get(event.get("run_id"))
        payload = event.get("attributes", {}).get("output")
        if case_id and isinstance(payload, dict):
            outputs.setdefault(case_id, []).append(payload)
    rows = [_score_case(case, outputs.get(case["case_id"], [])) for case in cases]
    passed = sum(row["passed"] for row in rows)
    summary = {
        "metric": "sql_result_set_exact_match_rate",
        "definition": "Executed SQL rows match frozen Gold answer fields, respecting row order and numeric tolerance. Entity IDs supplied in the prompt are not required when correctly omitted from SELECT output.",
        "case_count": len(rows), "passed_count": passed,
        "sql_execution_correctness": passed / len(rows) if rows else 0.0,
        "derived_relations_scored": False,
        "note": "Derived answer-level comparisons/differences are excluded; this metric isolates executed SQL result-set correctness.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sql_execution_correctness_cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "sql_execution_correctness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
