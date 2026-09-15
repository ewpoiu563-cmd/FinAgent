"""Validate calibration composition and emit a hash-bound dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    cases = []
    files = []
    for path in args.files:
        records = load(path)
        cases.extend(records)
        files.append({"path": path.as_posix(), "sha256": sha256(path), "case_count": len(records)})

    identifiers = [str(case["case_id"]) for case in cases]
    duplicates = sorted(case_id for case_id, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates}")
    if len(cases) != 60:
        raise ValueError(f"calibration v0.1 requires 60 cases, found {len(cases)}")

    known = set(identifiers)
    missing_references = []
    for case in cases:
        if not (case.get("question") or case.get("conversation")):
            raise ValueError(f"case has no question/conversation: {case['case_id']}")
        for reference in case.get("gold_references", []):
            if reference not in known:
                missing_references.append((case["case_id"], reference))
    if missing_references:
        raise ValueError(f"missing Gold references: {missing_references}")

    difficulty = Counter(str(case["difficulty"]) for case in cases)
    expected_difficulty = {"daily": 14, "hard": 20, "boundary": 13, "long_tail": 13}
    if dict(difficulty) != expected_difficulty:
        raise ValueError(f"unexpected difficulty mix: {dict(difficulty)}")

    manifest = {
        "dataset": "finagent-eval-v2-calibration",
        "version": "0.1",
        "split": "calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "difficulty_counts": dict(sorted(difficulty.items())),
        "case_ids": identifiers,
        "source_files": files,
        "policy": {
            "agent_outputs_used_to_author_gold": False,
            "may_tune_on_split": True,
            "rag_precision_ndcg_publishable": False,
            "rag_blocker": "pooled candidate relevance review not completed"
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
