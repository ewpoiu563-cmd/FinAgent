"""Merge score-input JSON arrays without changing their records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    ids = [row["query_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merged {len(records)} records")


if __name__ == "__main__":
    main()
