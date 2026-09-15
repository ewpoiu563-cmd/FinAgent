"""Read-only requirement and case scoring."""

from __future__ import annotations

from datetime import date
from dataclasses import replace
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .alignment import DeterministicRequirementAligner, RequirementAligner
from .assertions import AssertionValidatorRegistry
from .models import CaseEvaluation, EvaluationRunArtifact, ObservedEvidence, RequirementEvaluation


class AgentEvaluator:
    def __init__(self, *, aligner: RequirementAligner | None = None, validators: AssertionValidatorRegistry | None = None) -> None:
        self.aligner = aligner or DeterministicRequirementAligner()
        self.validators = validators or AssertionValidatorRegistry()

    def evaluate(self, gold_case: Mapping[str, Any], run: EvaluationRunArtifact) -> CaseEvaluation:
        identity_invalid = []
        if run.case_id != gold_case["case_id"]:
            identity_invalid.append("case_id_mismatch")
        if run.reference_date != gold_case["reference_date"]:
            identity_invalid.append("reference_date_mismatch")
        invalid = tuple((*run.invalid_reasons, *identity_invalid))
        if not run.trace_complete:
            invalid = (*invalid, "trace_incomplete")
        alignments = {item.gold_requirement_id: item for item in self.aligner.align(gold_case, run)}
        task_by_id = {task.runtime_task_id: task for task in run.planned_requirements}
        requirement_results = []
        for gold in gold_case["requirements"]:
            alignment = alignments[gold["requirement_id"]]
            tasks = [task_by_id[tid] for tid in alignment.runtime_task_ids]
            requirement_results.append(self._score_requirement(gold, alignment.status, tasks, run, invalid))
        case_evidence = self._matched_evidence(gold_case.get("acceptable_evidence", []), run.evidence_registry)
        case_gold_texts = _gold_evidence_texts(gold_case.get("acceptable_evidence", []))
        synthesis_results = tuple(self.validators.validate(assertion, answer=run.final_answer or "", observation=run.case_assertion_observations.get(assertion["assertion_id"]), evidence=case_evidence, gold_evidence_texts=case_gold_texts) for assertion in gold_case.get("expected_answer_assertions", []))
        synthesis_correct = all(result.passed for result in synthesis_results)
        required = [result for result, gold in zip(requirement_results, gold_case["requirements"]) if gold.get("required")]
        headline_required = [result for result in required if result.headline_eligible]
        weight_total = sum(result.weight for result in headline_required)
        coverage = sum(result.weight for result in headline_required if result.requirement_fully_satisfied) / weight_total if weight_total else 1.0
        outcome_correct = run.final_status == gold_case["expected_case_outcome"]
        conflict = bool(run.plan_artifact_conflicts)
        headline_eligible = all(result.headline_eligible for result in required)
        fully = headline_eligible and not invalid and not conflict and outcome_correct and coverage == 1.0 and synthesis_correct
        return CaseEvaluation(gold_case["case_id"], run.run_id, not invalid, invalid, tuple(requirement_results), outcome_correct and not invalid, coverage if not invalid else 0.0, synthesis_results, synthesis_correct and not invalid, fully, int(fully), conflict, headline_eligible)

    def _score_requirement(self, gold, alignment_status, tasks, run, invalid):
        mapped = alignment_status.startswith("mapped") and bool(tasks)
        planned = {task.planned_source for task in tasks if task.planned_source}
        modes = {task.tool_necessity_mode for task in tasks}
        planned_correct = mapped and sorted(planned) in [sorted(item) for item in gold["acceptable_source_sets"]]
        planned_correct = planned_correct and set(gold["required_sources"]).issubset(planned) and not set(gold["forbidden_sources"]).intersection(planned)
        if run.plan_artifact_conflicts:
            planned_correct = False
        # An empty task set makes both helpers vacuously true.  Alignment is a
        # prerequisite for evaluating runtime behaviour, so unknown ownership
        # must fail closed instead of inflating compliance metrics.
        fallback_ok = mapped and self._fallback_compliant(gold["fallback_expectation"], tasks)
        actual_ok = mapped and self._actual_sources_compliant(gold, planned, tasks, fallback_ok)
        evidence = self._matched_evidence(gold["acceptable_evidence"], run.evidence_registry, {task.runtime_task_id for task in tasks}, gold.get("web_evaluation_mode", "not_applicable"))
        evidence_ok = self._evidence_complete(gold["acceptable_evidence"], evidence)
        freshness_ok = evidence_ok and self._freshness_correct(gold, evidence, run.reference_date)
        answer = "\n".join(task.answer_fragment or "" for task in tasks)
        observations = {}
        for task in tasks:
            observations.update(task.assertion_observations)
        gold_texts = _gold_evidence_texts(gold["acceptable_evidence"])
        assertions = tuple(self.validators.validate(assertion, answer=answer, observation=observations.get(assertion["assertion_id"]), evidence=evidence, gold_evidence_texts=gold_texts) for assertion in gold["expected_answer_assertions"])
        required_assertions = [item for item, raw in zip(assertions, gold["expected_answer_assertions"]) if raw.get("required")]
        strict = all(item.passed for item in required_assertions)
        total_weight = sum(item.weight for item in required_assertions)
        weighted = sum(item.weight for item in required_assertions if item.passed) / total_weight if total_weight else 1.0
        outcomes = {task.terminal_status for task in tasks}
        outcome_ok = mapped and outcomes == {gold["expected_requirement_outcome"]}
        tool_ok = mapped and modes == {gold["expected_tool_necessity_mode"]}
        dependency_failure = any(task.dependency_failure for task in tasks)
        headline_eligible = gold["expected_tool_necessity_mode"] in {"no_tool", "tool_required"}
        fully = bool(mapped and not invalid and tool_ok and planned_correct and actual_ok and outcome_ok and fallback_ok and strict and evidence_ok and freshness_ok and not dependency_failure and not run.plan_artifact_conflicts)
        reasons = tuple(invalid) + (("plan_artifact_conflict",) if run.plan_artifact_conflicts else ())
        return RequirementEvaluation(gold["requirement_id"], bool(gold.get("required")), float(gold.get("weight", 1)), headline_eligible, alignment_status, tuple(task.runtime_task_id for task in tasks), mapped and not invalid, tool_ok and not invalid, planned_correct and not invalid, actual_ok and not invalid, outcome_ok and not invalid, fallback_ok and not invalid, dependency_failure, assertions, strict and not invalid, weighted if not invalid else 0.0, evidence_ok and not invalid, freshness_ok and not invalid, fully, reasons)

    @staticmethod
    def _fallback_compliant(expectation, tasks):
        transitions = [transition for task in tasks for transition in task.fallback_transitions]
        matching = [
            transition
            for task in tasks
            for transition in task.fallback_transitions
            if transition.from_source == expectation.get("from_source")
            and transition.to_source == expectation.get("to_source")
            and transition.trigger == expectation.get("trigger")
            and _trigger_supported(transition.trigger, task.source_outcomes.get(transition.from_source))
        ]
        mode = expectation["mode"]
        return not transitions if mode == "forbidden" else True if mode == "optional" and (not transitions or len(matching) == len(transitions)) else bool(matching) and len(matching) == len(transitions)

    @staticmethod
    def _actual_sources_compliant(gold, planned, tasks, fallback_ok):
        actual = {source for task in tasks for source in task.actual_sources}
        extras = actual - planned
        permitted = {transition.to_source for task in tasks for transition in task.fallback_transitions} if fallback_ok else set()
        forbidden_actual = set(gold["forbidden_sources"]) - permitted
        return planned.issubset(actual) and not forbidden_actual.intersection(actual) and extras.issubset(permitted)

    @staticmethod
    def _matched_evidence(rules, observed, owners=None, web_evaluation_mode="not_applicable"):
        matched = []
        for item in observed:
            if owners is not None and item.runtime_task_id not in owners:
                continue
            gold_ids = []
            for rule in rules:
                if item.source == rule["source"] and _identity_matches(rule.get("identity", {}), item.identity, item.source, web_evaluation_mode):
                    gold_ids.append(rule["evidence_id"])
            if gold_ids:
                matched.append(replace(item, matched_gold_evidence_ids=tuple(gold_ids)))
        return tuple(matched)

    @staticmethod
    def _evidence_complete(rules, matched):
        ids = {identifier for item in matched for identifier in item.matched_gold_evidence_ids}
        return all(not rule.get("required") or rule["evidence_id"] in ids for rule in rules)

    @staticmethod
    def _freshness_correct(gold, evidence, reference_date):
        requirement = gold["freshness_requirement"]
        mode = requirement["mode"]
        if mode in {"none", "historical"}:
            # Historical as_of describes fact time. Publication may legitimately be later.
            return True
        reference = date.fromisoformat(reference_date)
        for item in evidence:
            fact_date = item.fact_as_of or item.identity.get("as_of") or item.identity.get("reference_date")
            if not fact_date:
                return False
            observed = date.fromisoformat(str(fact_date)[:10])
            if mode == "current" and observed != reference:
                return False
            if mode == "bounded_live" and (reference - observed).days > int(requirement["max_age_days"]):
                return False
        return True


def _identity_matches(expected, observed, source, web_evaluation_mode="not_applicable"):
    if source == "sql":
        return _sql_identity_matches(expected, observed)
    keys = {
        "rag": ("doc_id", "retrieval_unit_id", "pages"),
        "web": ("url", "publisher", "publisher_domain", "publisher_platform", "publication_date", "capture_id", "capture_sha256"),
    }.get(source, ())
    compared = False
    for key in keys:
        if key in expected:
            if source == "web" and web_evaluation_mode == "live" and key in {"capture_id", "capture_sha256"}:
                continue
            if source == "web" and web_evaluation_mode == "live" and key in {"publisher", "publisher_platform", "publication_date"} and key not in observed:
                continue
            compared = True
            left = _canonical_url(expected[key]) if source == "web" and key == "url" else observed.get(key)
            right = _canonical_url(observed.get(key)) if source == "web" and key == "url" else expected[key]
            if right != left:
                return False
    return compared or not expected


def _sql_identity_matches(expected, observed):
    """Match SQL provenance without treating Gold SQL as the implementation.

    Gold query text, parameters, fingerprints, and query artifact IDs describe
    the evaluator-owned oracle.  Production only has to preserve an auditable
    executed statement against the same database snapshot.  Table scope is
    compared when the runtime extractor could identify it reliably.
    """
    expected_database = expected.get("database_sha256")
    observed_database = observed.get("database_sha256")
    if not expected_database or observed_database != expected_database:
        return False

    executed_sql = observed.get("generated_sql") or observed.get("canonical_sql")
    if not isinstance(executed_sql, str) or not executed_sql.strip():
        return False

    expected_tables = {
        value for key in ("table", "joined_table")
        for value in [expected.get(key)] if isinstance(value, str) and value
    }
    observed_tables = {
        value for key in ("table", "joined_table")
        for value in [observed.get(key)] if isinstance(value, str) and value
    }
    if expected_tables and observed_tables and not expected_tables.issubset(observed_tables):
        return False
    return True


def _canonical_url(value):
    if not isinstance(value, str):
        return value
    parts = urlsplit(value)
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/") or "/", parts.query, ""))


def _gold_evidence_texts(rules):
    return tuple(
        text
        for rule in rules
        for text in [rule.get("identity", {}).get("evidence_text")]
        if isinstance(text, str) and text
    )


def _trigger_supported(trigger, source_outcome):
    if trigger == "zero_result":
        return source_outcome in {"zero_result", "insufficient"}
    if trigger in {"source_failure", "generation_failure", "provider_content_block", "insufficient"}:
        return source_outcome == trigger
    return bool(trigger and source_outcome)
