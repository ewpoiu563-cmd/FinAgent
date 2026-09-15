"""Build one reviewable answer key without invoking the system under test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render(case: dict[str, Any]) -> str:
    if "gold_rows" in case:
        parts = ["数据库标准结果：" + json.dumps(case["gold_rows"], ensure_ascii=False, separators=(",", ":"))]
        if case.get("derived_results"):
            parts.append("派生结论：" + json.dumps(case["derived_results"], ensure_ascii=False, separators=(",", ":")))
        if not case["gold_rows"]:
            parts.append("结论：指定条件下无记录；不得改查其他日期或外部来源。")
        return "\n".join(parts)
    claims = list(case.get("expected_claims", []))
    numerics = case.get("expected_numeric", [])
    for item in numerics:
        unit = item.get("unit") or ""
        claims.append(f"{item['expression']} = {item['value']}{unit}")
    if not claims and case.get("expected_outcome"):
        claims.append(f"预期结果状态：{case['expected_outcome']}")
    answer = "；".join(claims) + ("。" if claims else "")
    if case.get("evidence"):
        citations = []
        for item in case["evidence"]:
            if "url" in item:
                citations.append(f"{item.get('publisher', '官方来源')}，{item['url']}")
            else:
                citations.append(f"{item.get('source_file') or item['doc_id']}，PDF第{item['pdf_page']}页，{item['retrieval_unit_id']}")
        answer += "\n证据：" + "；".join(citations) + "。"
    if case.get("required_sources"):
        answer += "\n来源：" + "；".join(case["required_sources"]) + "。"
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        value = json.loads(path.read_text(encoding="utf-8"))
        for case in value if isinstance(value, list) else value["records"]:
            records.append({
                "case_id": case["case_id"],
                "difficulty": case.get("difficulty"),
                "gold_status": case.get("gold_status"),
                "canonical_answer": render(case),
                "expected_route": case.get("expected_route"),
                "expected_outcome": case.get("expected_outcome", "success"),
                "forbidden_sources": case.get("forbidden_sources", []),
                "forbidden_claims": case.get("forbidden_claims", []),
            })
    ids = [row["case_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in answer key")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"built {len(records)} canonical answers")


if __name__ == "__main__":
    main()
