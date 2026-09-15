"""Evaluator-owned immutable normalized run artifacts and score models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class FallbackTransition:
    from_source: str
    to_source: str
    trigger: str | None = None
    runtime_task_id: str | None = None


@dataclass(frozen=True, slots=True)
class ObservedEvidence:
    evidence_id: str
    source: str
    identity: Mapping[str, Any]
    content: str | None = None
    runtime_task_id: str | None = None
    fact_as_of: str | None = None
    publication_date: str | None = None
    matched_gold_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannedRequirement:
    runtime_task_id: str
    atomic_question: str
    required: bool
    depends_on: tuple[str, ...]
    tool_necessity_mode: str
    planned_source: str | None
    terminal_status: str
    planned_status: str = "pending"
    answer_fragment: str | None = None
    evidence_ids: tuple[str, ...] = ()
    stable_origin_requirement_ids: tuple[str, ...] = ()
    actual_sources: tuple[str, ...] = ()
    fallback_transitions: tuple[FallbackTransition, ...] = ()
    source_outcomes: Mapping[str, str] = field(default_factory=dict)
    dependency_failure: bool = False
    assertion_observations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationRunArtifact:
    case_id: str
    run_id: str
    reference_date: str
    system_snapshot_id: str
    trace_ref: str | None
    run_summary_ref: str | None
    final_status: str
    final_answer: str | None
    trace_complete: bool
    planned_requirements: tuple[PlannedRequirement, ...]
    evidence_registry: tuple[ObservedEvidence, ...] = ()
    case_assertion_observations: Mapping[str, Any] = field(default_factory=dict)
    plan_artifact_conflicts: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()
    session_id: str | None = None
    system_snapshot: Mapping[str, Any] = field(default_factory=dict)
    serialized_task_plan: Mapping[str, Any] | None = None
    terminal_task_plan: Mapping[str, Any] | None = None
    run_summary: Mapping[str, Any] | None = None
    trace_malformed_line_count: int = 0

    @property
    def valid(self) -> bool:
        return self.trace_complete and not self.invalid_reasons

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRunArtifact":
        """Load the evaluator-owned JSON form without touching production models."""
        tasks = []
        for raw in value.get("planned_requirements", []):
            transitions = tuple(FallbackTransition(**item) for item in raw.get("fallback_transitions", []))
            tasks.append(PlannedRequirement(
                runtime_task_id=raw["runtime_task_id"], atomic_question=raw["atomic_question"],
                required=raw["required"], depends_on=tuple(raw.get("depends_on", [])),
                tool_necessity_mode=raw["tool_necessity_mode"], planned_source=raw.get("planned_source"),
                terminal_status=raw["terminal_status"], planned_status=raw.get("planned_status", "pending"),
                answer_fragment=raw.get("answer_fragment"),
                evidence_ids=tuple(raw.get("evidence_ids", [])),
                stable_origin_requirement_ids=tuple(raw.get("stable_origin_requirement_ids", [])),
                actual_sources=tuple(raw.get("actual_sources", [])), fallback_transitions=transitions,
                source_outcomes=dict(raw.get("source_outcomes", {})),
                dependency_failure=bool(raw.get("dependency_failure", False)),
                assertion_observations=dict(raw.get("assertion_observations", {})),
            ))
        evidence = tuple(ObservedEvidence(
            evidence_id=item["evidence_id"], source=item["source"], identity=dict(item.get("identity", {})),
            content=item.get("content"), runtime_task_id=item.get("runtime_task_id"),
            fact_as_of=item.get("fact_as_of"), publication_date=item.get("publication_date"),
            matched_gold_evidence_ids=tuple(item.get("matched_gold_evidence_ids", [])),
        ) for item in value.get("evidence_registry", []))
        return cls(
            case_id=value["case_id"], run_id=value["run_id"], reference_date=value["reference_date"],
            system_snapshot_id=value["system_snapshot_id"], trace_ref=value.get("trace_ref"),
            run_summary_ref=value.get("run_summary_ref"), final_status=value["final_status"],
            final_answer=value.get("final_answer"), trace_complete=bool(value["trace_complete"]),
            planned_requirements=tuple(tasks), evidence_registry=evidence,
            case_assertion_observations=dict(value.get("case_assertion_observations", {})),
            plan_artifact_conflicts=tuple(value.get("plan_artifact_conflicts", [])),
            invalid_reasons=tuple(value.get("invalid_reasons", [])),
            session_id=value.get("session_id"),
            system_snapshot=dict(value.get("system_snapshot", {})),
            serialized_task_plan=value.get("serialized_task_plan"),
            terminal_task_plan=value.get("terminal_task_plan"),
            run_summary=value.get("run_summary"),
            trace_malformed_line_count=int(value.get("trace_malformed_line_count", 0)),
        )


@dataclass(frozen=True, slots=True)
class Alignment:
    gold_requirement_id: str
    status: str
    runtime_task_ids: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AssertionResult:
    assertion_id: str
    assertion_type: str
    passed: bool
    weight: float
    reason: str


@dataclass(frozen=True, slots=True)
class RequirementEvaluation:
    requirement_id: str
    required: bool
    weight: float
    headline_eligible: bool
    alignment_status: str
    runtime_task_ids: tuple[str, ...]
    planning_recall: bool
    tool_necessity_correct: bool
    planned_source_correct: bool
    actual_source_compliant: bool
    outcome_correct: bool
    fallback_compliant: bool
    dependency_failure: bool
    assertion_results: tuple[AssertionResult, ...]
    answer_correct_strict: bool
    answer_correct_weighted: float
    evidence_correct: bool
    freshness_correct: bool
    requirement_fully_satisfied: bool
    invalid_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    run_id: str
    valid_artifact: bool
    invalid_reasons: tuple[str, ...]
    requirement_results: tuple[RequirementEvaluation, ...]
    expected_outcome_correct: bool
    gold_requirement_coverage: float
    synthesis_assertion_results: tuple[AssertionResult, ...]
    synthesis_correct: bool
    case_fully_satisfied: bool
    tsr_contribution: int
    plan_artifact_conflict: bool
    headline_eligible: bool
