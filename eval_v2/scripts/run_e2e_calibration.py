"""Run the real FinAgent entrypoint against frozen calibration cases.

The runner is append-only and resumable: a completed case_id is never rerun
unless --rerun is supplied. Runtime answers and traces are not Gold data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        cases.extend(value if isinstance(value, list) else value["records"])
    return cases


def prompt_and_history(case: dict[str, Any]) -> tuple[str, list[dict[str, str]] | None]:
    if "question" in case:
        return case["question"], None
    conversation = case["conversation"]
    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError(f"{case['case_id']}: conversation must end with user")
    return conversation[-1]["content"], conversation[:-1]


async def main_async(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ["TRACE_ENABLED"] = "1"
    os.environ["TRACE_DIR"] = str(args.trace_dir.resolve())
    from agent_loop import run_agent

    cases = load_cases(args.inputs)
    selected = set(args.case_ids or ())
    if selected:
        cases = [case for case in cases if case["case_id"] in selected]
    prior: set[str] = set()
    if args.output.exists() and not args.rerun:
        prior = {
            json.loads(line)["case_id"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for position, case in enumerate(cases, 1):
        case_id = case["case_id"]
        if case_id in prior:
            print(f"[{position}/{len(cases)}] skip {case_id}", flush=True)
            continue
        question, history = prompt_and_history(case)
        run_id = f"evalv2-{case_id}-{uuid.uuid4().hex[:10]}"
        started = time.perf_counter()
        status = "completed"
        answer = ""
        error_type = None
        error_message = None
        try:
            answer = await asyncio.wait_for(
                run_agent(question, session_id=None, run_id=run_id, chat_history=history),
                timeout=args.timeout,
            )
        except asyncio.TimeoutError as error:
            status, error_type, error_message = "timeout", type(error).__name__, str(error)
        except Exception as error:  # preserve failures as eval observations
            status, error_type, error_message = "error", type(error).__name__, str(error)
        record = {
            "case_id": case_id,
            "difficulty": case.get("difficulty"),
            "question": question,
            "run_id": run_id,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "answer": answer,
            "error_type": error_type,
            "error_message": error_message,
        }
        with args.output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{position}/{len(cases)}] {case_id}: {status} ({record['latency_ms']} ms)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--rerun", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
