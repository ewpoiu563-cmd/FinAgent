"""Typed assertion validation with an explicit deterministic/semantic boundary."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Protocol, Sequence

from .models import AssertionResult, ObservedEvidence


@dataclass(frozen=True, slots=True)
class SemanticJudgment:
    passed: bool
    contradiction_detected: bool
    reason: str


class SemanticJudge(Protocol):
    def judge(self, *, answer: str, assertion: Mapping[str, Any], evidence_texts: Sequence[str], gold_evidence_texts: Sequence[str] = ()) -> SemanticJudgment: ...


class CannedSemanticJudge:
    """Offline-only judge fixture; unknown assertions fail closed."""
    def __init__(self, decisions: Mapping[str, bool] | None = None) -> None:
        self.decisions = dict(decisions or {})
    def judge(self, *, answer: str, assertion: Mapping[str, Any], evidence_texts: Sequence[str], gold_evidence_texts: Sequence[str] = ()) -> SemanticJudgment:
        passed = self.decisions.get(assertion["assertion_id"], False)
        return SemanticJudgment(passed, not passed, "canned semantic judgment")


class AssertionValidatorRegistry:
    def __init__(self, semantic_judge: SemanticJudge | None = None) -> None:
        self.semantic_judge = semantic_judge

    def validate(self, assertion: Mapping[str, Any], *, answer: str, observation: Any, evidence: Sequence[ObservedEvidence], gold_evidence_texts: Sequence[str] = ()) -> AssertionResult:
        kind = assertion["type"]
        validator = getattr(self, f"_validate_{kind}")
        if kind in {"claim", "absence_or_refusal"}:
            passed, reason = self._semantic(assertion, answer, evidence, gold_evidence_texts)
        else:
            passed, reason = validator(assertion, answer, observation, evidence)
        return AssertionResult(assertion["assertion_id"], kind, passed, float(assertion.get("weight", 1)), reason)

    def _validate_numeric(self, assertion, answer, observation, evidence):
        value, unit = _observed_value(observation)
        numbers = [value] if isinstance(value, (int, float)) else _numbers(answer)
        target = float(assertion["target"])
        absolute = float(assertion.get("absolute_tolerance") or 0)
        relative = assertion.get("relative_tolerance")
        tolerance = max(absolute, abs(target) * float(relative or 0))
        passed = any(math.isclose(float(number), target, rel_tol=0, abs_tol=tolerance) for number in numbers)
        if unit is not None and unit != assertion.get("unit"):
            passed = False
        if isinstance(observation, Mapping) and assertion.get("value_representation"):
            observed_representation = observation.get("value_representation")
            if observed_representation is not None and observed_representation != assertion["value_representation"]:
                passed = False
        return passed, "numeric target/tolerance/unit matched" if passed else "numeric target, tolerance, or unit mismatch"

    def _validate_date(self, assertion, answer, observation, evidence):
        value, _ = _observed_value(observation)
        candidates = [str(value)] if value is not None else re.findall(r"(?:19|20)\d{2}-\d{2}-\d{2}", answer)
        passed = assertion["target"] in candidates
        if assertion.get("not_after") and passed:
            passed = date.fromisoformat(assertion["target"]) <= date.fromisoformat(assertion["not_after"])
        return passed, "date constraints matched" if passed else "date target/cutoff mismatch"

    def _validate_set_or_rows(self, assertion, answer, observation, evidence):
        value, _ = _observed_value(observation)
        rows = value if isinstance(value, list) else None
        if rows is None:
            return False, "structured rows observation missing"
        expected = assertion["rows"]
        passed = len(rows) == assertion["row_count"] and rows == expected if assertion.get("ordered") else _row_set(rows) == _row_set(expected)
        return passed, "rows matched" if passed else "row count/content/order mismatch"

    def _validate_relation(self, assertion, answer, observation, evidence):
        value, _ = _observed_value(observation)
        if isinstance(value, bool):
            return value, "normalized relation observation passed" if value else "normalized relation observation failed"
        left, right = assertion.get("left_target"), assertion.get("right_target")
        proposition = "".join(assertion.get("required_propositions", []))
        stated = proposition and proposition.rstrip("。") in answer
        relation = assertion.get("relation")
        true_relation = (left > right if relation == "greater_than" else left < right if relation == "less_than" else left == right)
        return bool(stated and true_relation), "relation matched" if stated and true_relation else "relation missing or contradicted"

    def _validate_citation(self, assertion, answer, observation, evidence):
        ids = {identifier for item in evidence for identifier in item.matched_gold_evidence_ids}
        required = set(assertion.get("required_evidence_ids", []))
        linked_observed = set(observation.get("linked_observed_evidence_ids", [])) if isinstance(observation, Mapping) else set()
        answer_linked = any(
            token and str(token) in answer
            for item in evidence
            for token in (item.evidence_id, item.identity.get("url"), item.identity.get("publisher_domain"), item.identity.get("page_title"))
        )
        passed = required.issubset(ids) and bool(linked_observed or answer_linked)
        return passed, "required citation evidence linked" if passed else "required citation evidence missing"

    def _validate_claim(self, assertion, answer, observation, evidence):
        return self._semantic(assertion, answer, evidence, ())

    def _validate_absence_or_refusal(self, assertion, answer, observation, evidence):
        return self._semantic(assertion, answer, evidence, ())

    def _semantic(self, assertion, answer, evidence, gold_evidence_texts):
        if self.semantic_judge is None:
            return False, "semantic judge unavailable"
        judgment = self.semantic_judge.judge(answer=answer, assertion=assertion, evidence_texts=[item.content for item in evidence if item.content], gold_evidence_texts=gold_evidence_texts)
        passed = judgment.passed and not judgment.contradiction_detected
        return passed, judgment.reason


def _observed_value(observation: Any) -> tuple[Any, str | None]:
    if isinstance(observation, Mapping):
        return observation.get("value"), observation.get("unit")
    return observation, None


def _numbers(text: str) -> list[float]:
    values = []
    for item in re.findall(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            values.append(float(item.replace(",", "")))
        except ValueError:
            pass
    return values


def _row_set(rows: list[Mapping[str, Any]]) -> set[tuple[tuple[str, Any], ...]]:
    return {tuple(sorted(row.items())) for row in rows}
