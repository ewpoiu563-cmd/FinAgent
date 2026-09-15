"""Materialize independently authored SQL Gold against a versioned SQLite DB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(seed_path: Path, database_path: Path) -> list[dict[str, Any]]:
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(seeds, list):
        raise ValueError("SQL seed must be a JSON list")
    database_hash = sha256(database_path)
    connection = sqlite3.connect(f"file:{database_path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    output = []
    try:
        for seed in seeds:
            statement = str(seed["gold_sql"]).strip()
            if not statement.casefold().startswith(("select", "with")):
                raise ValueError(f"non-read-only Gold SQL: {seed['case_id']}")
            cursor = connection.execute(statement)
            rows = [dict(row) for row in cursor.fetchall()]
            derived_results = []
            for assertion in seed.get("derived_assertions", []):
                if assertion.get("type") == "compare_and_difference":
                    if len(rows) != 2:
                        raise ValueError(f"{seed['case_id']}: comparison requires exactly two rows")
                    value_column = assertion["value_column"]
                    entity_column = assertion["entity_column"]
                    high, low = sorted(rows, key=lambda row: float(row[value_column]), reverse=True)
                    derived_results.append({
                        "type": "compare_and_difference",
                        "higher_entity": high[entity_column],
                        "lower_entity": low[entity_column],
                        "difference": round(float(high[value_column]) - float(low[value_column]), 2),
                        "numeric_tolerance": assertion.get("numeric_tolerance", 0.0),
                    })
            record = dict(seed)
            record.update(
                gold_status="verified",
                gold_rows=rows,
                derived_results=derived_results,
                database_sha256=database_hash,
                materialized_at=datetime.now(timezone.utc).isoformat(),
            )
            output.append(record)
    finally:
        connection.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = materialize(args.seed, args.database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
