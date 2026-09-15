"""Run the frozen Phase 5 semantic-judge qualification fixture set once."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agent_eval.semantic_judge import (
    SEMANTIC_JUDGE_OUTPUT_SCHEMA_VERSION,
    SEMANTIC_JUDGE_PROMPT_VERSION,
    SemanticJudgeAdapter,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="eval/semantic_judge_qualification_v0.1.json")
    parser.add_argument("--output-dir", default="outputs/eval/semantic_judge_qualification/v0.1-run1")
    parser.add_argument("--report", default="eval/SEMANTIC_JUDGE_QUALIFICATION_REPORT.md")
    args = parser.parse_args()
    from config import LLM_MODEL, call_llm

    fixture_path = Path(args.fixtures)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "judge_log.jsonl"
    adapter = SemanticJudgeAdapter(call_llm, timeout_seconds=60, malformed_retries=1, log_path=log_path)
    results = []
    for fixture in payload["fixtures"]:
        started = time.perf_counter()
        judgment = adapter.judge(
            answer=fixture["answer"], assertion=fixture["assertion"],
            evidence_texts=fixture.get("evidence_texts", []),
            gold_evidence_texts=fixture.get("gold_evidence_texts", []),
        )
        predicted = judgment.passed and not judgment.contradiction_detected
        results.append({
            "id": fixture["id"], "class": fixture["class"], "expected": fixture["expected"],
            "predicted": predicted, "correct": predicted == fixture["expected"],
            "contradiction_detected": judgment.contradiction_detected,
            "reason": judgment.reason, "latency_ms": round((time.perf_counter() - started) * 1000),
            "not_evaluable": "not_evaluable" in judgment.reason,
        })
    total = len(results)
    correct = sum(item["correct"] for item in results)
    false_positive = sum(not item["expected"] and item["predicted"] for item in results)
    false_negative = sum(item["expected"] and not item["predicted"] for item in results)
    parse_failures = sum(item["not_evaluable"] for item in results)
    contradiction_failures = [item["id"] for item in results if "contradiction" in item["class"] and item["predicted"]]
    criteria = payload["pass_criteria"]
    passed = correct / total >= criteria["minimum_overall_accuracy"] and parse_failures <= criteria["maximum_parse_failures"] and not contradiction_failures
    summary = {
        "fixture_set": payload["fixture_set"], "fixture_version": payload["version"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": LLM_MODEL,
        "temperature": 0, "timeout_seconds": 60, "max_malformed_retries": 1,
        "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
        "output_schema_version": SEMANTIC_JUDGE_OUTPUT_SCHEMA_VERSION,
        "total": total, "correct": correct, "accuracy": correct / total,
        "false_positive": false_positive, "false_negative": false_negative,
        "parse_retry_failures": parse_failures, "contradiction_failures": contradiction_failures,
        "latency_ms": {"mean": sum(item["latency_ms"] for item in results) / total, "min": min(item["latency_ms"] for item in results), "max": max(item["latency_ms"] for item in results)},
        "token_usage": "unavailable from SemanticJudgeAdapter return contract",
        "passed": passed, "results": results,
    }
    (output_dir / "qualification_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = "\n".join(f"| {x['id']} | {x['class']} | {'PASS' if x['expected'] else 'FAIL'} | {'PASS' if x['predicted'] else 'FAIL'} | {'yes' if x['correct'] else 'no'} | {x['latency_ms']} |" for x in results)
    report = f"""# Semantic Judge Qualification Report\n\n## Configuration\n\n- Fixture set: `{payload['fixture_set']}@{payload['version']}` (15 fixed, manually pre-labeled fixtures; no benchmark Gold cases)\n- Model: `{LLM_MODEL}`\n- Temperature: `0`\n- Timeout: `60s`; malformed-output retry limit: `1`\n- Prompt: `{SEMANTIC_JUDGE_PROMPT_VERSION}`; schema: `{SEMANTIC_JUDGE_OUTPUT_SCHEMA_VERSION}`\n- Predeclared gate: accuracy >= 90%, zero parse failures, and no contradiction fixture judged PASS\n\n## Result\n\n- Status: **{'PASS' if passed else 'FAIL'}**\n- Correct: {correct}/{total} ({correct/total:.1%})\n- False positives: {false_positive}; false negatives: {false_negative}; parse/retry failures: {parse_failures}\n- Latency: mean {summary['latency_ms']['mean']:.0f} ms, min {summary['latency_ms']['min']} ms, max {summary['latency_ms']['max']} ms\n- Token usage: unavailable from the current judge adapter return contract\n\n## Fixture results\n\n| ID | Class | Expected | Predicted | Correct | Latency ms |\n|---|---|---:|---:|---:|---:|\n{rows}\n\nRaw auditable results and Judge exchanges are persisted under `{output_dir.as_posix()}`.\n"""
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("total", "correct", "accuracy", "false_positive", "false_negative", "parse_retry_failures", "passed")}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
