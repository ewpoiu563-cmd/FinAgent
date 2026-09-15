"""Persist one production RAG answer per authored Gold case without scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from react_agent import run_react_agent  # noqa: E402


async def run(gold: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        json.loads(line)["case_id"]
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } if output.exists() else set()
    with output.open("a", encoding="utf-8") as handle:
        for position, case in enumerate(gold, 1):
            if case["case_id"] in completed:
                continue
            started = perf_counter()
            row: dict[str, Any] = {
                "case_id": case["case_id"], "question": case["question"],
                "difficulty": case.get("difficulty"), "test_type": case.get("test_type"),
                "scope_doc_ids": case.get("scope_doc_ids", []),
                "reference_answer": case.get("reference_answer"),
                "run_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                row["answer"] = await run_react_agent(
                    case["question"], allowed_doc_ids=tuple(case.get("scope_doc_ids") or ())
                )
                row["success"] = True
            except Exception as error:  # preserve failures as test results
                row.update(success=False, error_type=type(error).__name__, error=str(error), answer=None)
            row["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{position}/{len(gold)}] {case['case_id']} success={row['success']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.gold.read_text(encoding="utf-8"))
    asyncio.run(run(cases, args.output))


if __name__ == "__main__":
    main()
