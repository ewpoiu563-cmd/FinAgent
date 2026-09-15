"""Bounded, auditable adapter for online semantic judgments."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .assertions import SemanticJudgment

SEMANTIC_JUDGE_PROMPT_VERSION = "phase5-semantic-judge-v1"
SEMANTIC_JUDGE_OUTPUT_SCHEMA_VERSION = "1.0"


class SemanticJudgeAdapter:
    def __init__(
        self,
        call: Callable[..., str],
        *,
        timeout_seconds: float = 30,
        malformed_retries: int = 1,
        log_path: str | Path | None = None,
    ) -> None:
        if timeout_seconds <= 0 or malformed_retries < 0:
            raise ValueError("semantic judge bounds are invalid")
        self.call = call
        self.timeout_seconds = timeout_seconds
        self.malformed_retries = malformed_retries
        self.log_path = Path(log_path) if log_path is not None else None
        self._lock = threading.Lock()

    def judge(
        self,
        *,
        answer: str,
        assertion: Mapping[str, Any],
        evidence_texts: Sequence[str],
        gold_evidence_texts: Sequence[str] = (),
    ) -> SemanticJudgment:
        request = {
            "protocol": {
                "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
                "output_schema_version": SEMANTIC_JUDGE_OUTPUT_SCHEMA_VERSION,
                "task": "Decide whether the runtime answer satisfies the atomic Gold assertion.",
                "rubric": [
                    "PASS only if every required proposition is entailed by the runtime answer.",
                    "FAIL if a required proposition is missing, contradicted, assigned to the wrong entity, or contains a wrong number.",
                    "For absence_or_refusal, PASS only for an appropriately scoped insufficiency/refusal with no fabricated prohibited claim.",
                    "Harmless extra wording is allowed, but unsupported material claims are not.",
                    "Evidence excerpts are context only; never treat Gold evidence text as if the runtime answer stated it.",
                    "Return exactly one JSON object matching output_schema, with no markdown or additional text.",
                ],
            },
            "gold_atomic_assertion": dict(assertion),
            "runtime_requirement_answer_fragment": answer,
            "gold_acceptable_evidence_excerpts": list(gold_evidence_texts),
            "observed_evidence_excerpts": list(evidence_texts),
            "output_schema": {"passed": "boolean", "contradiction_detected": "boolean", "reason": "short string"},
        }
        last_reason = "semantic judge failed"
        for attempt in range(1, self.malformed_retries + 2):
            raw = None
            try:
                raw = self._call(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
                judgment = _parse(raw)
                self._log(request, raw, judgment, attempt, None)
                return judgment
            except FutureTimeout:
                last_reason = "semantic judge timeout; not_evaluable"
                self._log(request, None, None, attempt, "timeout")
                break
            except Exception as error:
                last_reason = f"semantic judge malformed/failure: {type(error).__name__}; not_evaluable"
                self._log(request, raw, None, attempt, type(error).__name__)
        return SemanticJudgment(False, False, last_reason)

    def _call(self, prompt: str) -> str:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-judge")
        future = executor.submit(self.call, prompt, temperature=0, timeout=self.timeout_seconds)
        try:
            return future.result(timeout=self.timeout_seconds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _log(self, request, raw, judgment, attempt, error):
        if self.log_path is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt,
            "request": request,
            "raw_response": raw,
            "parsed_response": ({"passed": judgment.passed, "contradiction_detected": judgment.contradiction_detected, "reason": judgment.reason} if judgment else None),
            "error": error,
        }
        try:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError:
            pass


def _parse(raw: str) -> SemanticJudgment:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty judge output")
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, Mapping):
        raise TypeError("judge output is not an object")
    if type(value.get("passed")) is not bool or type(value.get("contradiction_detected")) is not bool:
        raise TypeError("judge booleans are malformed")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise ValueError("judge reason is malformed")
    return SemanticJudgment(value["passed"], value["contradiction_detected"], reason.strip())


__all__ = ["SemanticJudgeAdapter", "SEMANTIC_JUDGE_PROMPT_VERSION", "SEMANTIC_JUDGE_OUTPUT_SCHEMA_VERSION"]
