"""Fail-fast loader and static validator for Agent Eval Gold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SOURCES = {"direct", "local_compute", "sql", "rag", "web"}
NECESSITY_MODES = {"no_tool", "tool_required", "exploratory"}
REQUIREMENT_OUTCOMES = {"success", "insufficient", "source_failure", "generation_failure", "provider_content_block", "clarification_needed"}
CASE_OUTCOMES = REQUIREMENT_OUTCOMES | {"partial_success"}
WEB_MODES = {"not_applicable", "stable", "live"}
FALLBACK_MODES = {"forbidden", "optional", "required"}
ASSERTION_TYPES = {"numeric", "set_or_rows", "claim", "absence_or_refusal", "date", "citation", "relation"}


class GoldValidationError(ValueError):
    pass


def load_gold(path: str | Path = "eval/gold/agent_eval_gold_v0.1.json") -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_gold(payload)
    return payload


def validate_gold(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("cases"), list):
        raise GoldValidationError("Gold root must contain a cases list")
    cases = payload["cases"]
    case_ids = [case.get("case_id") for case in cases]
    _unique(case_ids, "case_id")
    if payload.get("case_count") != len(cases):
        raise GoldValidationError("case_count mismatch")
    verified_count = sum(case.get("gold_status") == "verified" for case in cases)
    if payload.get("verified_case_count") != verified_count:
        raise GoldValidationError("verified_case_count mismatch")
    for case in cases:
        _validate_case(case)


def _validate_case(case: Mapping[str, Any]) -> None:
    verified = case.get("gold_status") == "verified"
    for key in ("case_id", "question", "reference_date", "category", "requirements", "expected_case_outcome"):
        if verified and case.get(key) is None:
            raise GoldValidationError(f"{case.get('case_id')}: verified field {key} is null")
    requirements = case.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise GoldValidationError(f"{case.get('case_id')}: requirements must be non-empty")
    requirement_ids = [item.get("requirement_id") for item in requirements]
    _unique(requirement_ids, f"{case.get('case_id')}.requirement_id")
    known = set(requirement_ids)
    graph: dict[str, list[str]] = {}
    case_assertions = case.get("expected_answer_assertions")
    case_evidence = case.get("acceptable_evidence")
    if verified and (case_assertions is None or case_evidence is None):
        raise GoldValidationError(f"{case.get('case_id')}: unresolved case scoring fields")
    all_assertion_ids = {
        assertion.get("assertion_id")
        for owner in [case, *requirements]
        for assertion in (owner.get("expected_answer_assertions") or [])
    }
    _validate_assertions_and_evidence(case_assertions or [], case_evidence or [], case.get("case_id"), all_assertion_ids)
    if case.get("expected_case_outcome") not in CASE_OUTCOMES:
        raise GoldValidationError(f"{case.get('case_id')}: invalid case outcome")
    for requirement in requirements:
        rid = requirement.get("requirement_id")
        if verified and any(requirement.get(key) is None for key in ("question", "required", "depends_on", "expected_tool_necessity_mode", "acceptable_source_sets", "required_sources", "forbidden_sources", "freshness_requirement", "fallback_expectation", "expected_requirement_outcome")):
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: verified required field is null")
        if not isinstance(requirement.get("weight"), (int, float)) or requirement["weight"] <= 0:
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: weight must be positive")
        dependencies = requirement.get("depends_on", [])
        if len(dependencies) != len(set(dependencies)) or not set(dependencies).issubset(known) or rid in dependencies:
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: invalid dependencies")
        graph[rid] = dependencies
        if requirement.get("expected_tool_necessity_mode") not in NECESSITY_MODES:
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: invalid necessity mode")
        _validate_sources(requirement, f"{case.get('case_id')}.{rid}")
        if requirement.get("web_evaluation_mode") not in WEB_MODES:
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: invalid web mode")
        fallback = requirement.get("fallback_expectation")
        if not isinstance(fallback, Mapping) or fallback.get("mode") not in FALLBACK_MODES:
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: invalid fallback")
        if fallback.get("mode") in {"optional", "required"} and not all(fallback.get(key) for key in ("from_source", "to_source", "trigger")):
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: scoped fallback fields required")
        if requirement.get("expected_requirement_outcome") not in REQUIREMENT_OUTCOMES:
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: invalid requirement outcome")
        assertions = requirement.get("expected_answer_assertions")
        evidence = requirement.get("acceptable_evidence")
        if verified and (assertions is None or evidence is None):
            raise GoldValidationError(f"{case.get('case_id')}.{rid}: unresolved TODO/null scoring fields")
        _validate_assertions_and_evidence(assertions or [], evidence or [], f"{case.get('case_id')}.{rid}", all_assertion_ids)
    _assert_acyclic(graph, str(case.get("case_id")))


def _validate_sources(record: Mapping[str, Any], label: str) -> None:
    sets = record.get("acceptable_source_sets")
    if not isinstance(sets, list) or not sets:
        raise GoldValidationError(f"{label}: acceptable_source_sets must be non-empty")
    required = set(record.get("required_sources", []))
    forbidden = set(record.get("forbidden_sources", []))
    if not required.issubset(SOURCES) or not forbidden.issubset(SOURCES):
        raise GoldValidationError(f"{label}: unknown source")
    for source_set in sets:
        if not isinstance(source_set, list) or not source_set or len(source_set) != len(set(source_set)) or not set(source_set).issubset(SOURCES):
            raise GoldValidationError(f"{label}: invalid acceptable source set")
        if not required.issubset(set(source_set)) or forbidden.intersection(source_set):
            raise GoldValidationError(f"{label}: required/forbidden source conflict")


def _validate_assertions_and_evidence(assertions: list[Any], evidence: list[Any], label: str, valid_assertion_ids: set[str] | None = None) -> None:
    if not isinstance(assertions, list) or not isinstance(evidence, list):
        raise GoldValidationError(f"{label}: assertions/evidence must be lists")
    assertion_ids = [item.get("assertion_id") for item in assertions if isinstance(item, Mapping)]
    evidence_ids = [item.get("evidence_id") for item in evidence if isinstance(item, Mapping)]
    _unique(assertion_ids, f"{label}.assertion_id")
    _unique(evidence_ids, f"{label}.evidence_id")
    assertion_set, evidence_set = set(assertion_ids), set(evidence_ids)
    resolvable_assertions = valid_assertion_ids or assertion_set
    for assertion in assertions:
        if assertion.get("type") not in ASSERTION_TYPES:
            raise GoldValidationError(f"{label}: unsupported assertion type")
        if assertion.get("required") and assertion.get("weight", 0) <= 0:
            raise GoldValidationError(f"{label}: required assertion needs positive weight")
        if not set(assertion.get("required_evidence_ids", [])).issubset(evidence_set):
            raise GoldValidationError(f"{label}: unresolved required_evidence_ids")
    for rule in evidence:
        if not set(rule.get("assertion_ids", [])).issubset(resolvable_assertions):
            raise GoldValidationError(f"{label}: evidence references unknown assertion")
        if rule.get("required") and (not rule.get("evidence_id") or not rule.get("source")):
            raise GoldValidationError(f"{label}: required evidence identity missing")
        if rule.get("required") and rule.get("source") in {"sql", "rag", "web"} and not isinstance(rule.get("identity"), Mapping):
            raise GoldValidationError(f"{label}: required external evidence needs stable identity")


def _assert_acyclic(graph: Mapping[str, list[str]], label: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(node: str) -> None:
        if node in visiting:
            raise GoldValidationError(f"{label}: dependency cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)
    for node in graph:
        visit(node)


def _unique(values: list[Any], label: str) -> None:
    if any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
        raise GoldValidationError(f"{label} must be unique non-empty strings")
