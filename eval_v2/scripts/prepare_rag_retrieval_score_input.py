"""Create deterministic retrieval-score inputs from bound Gold and a run file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    return value if isinstance(value, list) else value["records"]


def relevant(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "retrieval_unit_id": item["retrieval_unit_id"],
            "evidence_group": item["evidence_group"],
            "relevance": item["relevance"],
        }
        for item in case.get("evidence", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = load(args.gold)
    by_id = {case["case_id"]: relevant(case) for case in gold}
    if args.run is None:
        output = [{"query_id": key, "relevant": value} for key, value in by_id.items()]
    else:
        output = []
        for record in load(args.run):
            query_id = record["query_id"]
            if query_id not in by_id:
                raise ValueError(f"run has unknown query_id: {query_id}")
            output.append({**record, "relevant": by_id[query_id]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(output), "answerable": sum(bool(row.get("relevant")) for row in output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
