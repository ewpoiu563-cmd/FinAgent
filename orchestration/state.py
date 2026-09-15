"""Mutable execution state for source-aware single and multi-source runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .evidence import EvidenceBundle


_EXECUTABLE_SOURCES = frozenset({"sql", "rag", "web"})


class SourceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    INSUFFICIENT = "insufficient"
    SOURCE_FAILURE = "source_failure"
    GENERATION_FAILURE = "generation_failure"
    PROVIDER_CONTENT_BLOCK = "provider_content_block"


_FAILURE_STATUSES = frozenset(
    {
        SourceStatus.INSUFFICIENT,
        SourceStatus.SOURCE_FAILURE,
        SourceStatus.GENERATION_FAILURE,
        SourceStatus.PROVIDER_CONTENT_BLOCK,
    }
)


@dataclass(frozen=True)
class FallbackRecord:
    fallback_from: str
    fallback_to: str
    reason: str

    def __post_init__(self) -> None:
        if self.fallback_from not in _EXECUTABLE_SOURCES:
            raise ValueError(f"unsupported fallback source: {self.fallback_from}")
        if self.fallback_to not in _EXECUTABLE_SOURCES:
            raise ValueError(f"unsupported fallback target: {self.fallback_to}")
        if not self.reason.strip():
            raise ValueError("fallback reason must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "fallback_from": self.fallback_from,
            "fallback_to": self.fallback_to,
            "reason": self.reason,
        }


@dataclass
class AgentState:
    """Authoritative source state; legacy success booleans are not consulted."""

    required_sources: set[str]
    completed_sources: set[str] = field(default_factory=set)
    failed_sources: set[str] = field(default_factory=set)
    source_attempts: dict[str, int] = field(default_factory=dict)
    evidence_by_source: dict[str, EvidenceBundle] = field(default_factory=dict)
    fallback_records: list[FallbackRecord] = field(default_factory=list)
    source_statuses: dict[str, SourceStatus] = field(default_factory=dict)
    source_errors: dict[str, str] = field(default_factory=dict)
    answers_by_source: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.required_sources = set(self.required_sources)
        unknown = self.required_sources.difference(_EXECUTABLE_SOURCES)
        if unknown:
            raise ValueError(f"unsupported required sources: {sorted(unknown)}")
        if not self.required_sources:
            raise ValueError("required_sources must not be empty")
        self.source_statuses = {
            source: self.source_statuses.get(source, SourceStatus.PENDING)
            for source in self.required_sources
        }
        self._validate_sets()

    @property
    def tracked_sources(self) -> set[str]:
        """Sources participating in this run, including bounded fallbacks."""

        fallback_sources = {
            source
            for record in self.fallback_records
            for source in (record.fallback_from, record.fallback_to)
        }
        return self.required_sources | fallback_sources

    def start_source(self, source: str) -> bool:
        self._require_source(source)
        if source in self.completed_sources:
            return False
        if self.source_statuses[source] is SourceStatus.RUNNING:
            raise RuntimeError(f"source is already running: {source}")
        self.source_attempts[source] = self.source_attempts.get(source, 0) + 1
        self.source_statuses[source] = SourceStatus.RUNNING
        self.failed_sources.discard(source)
        self.source_errors.pop(source, None)
        return True

    def complete_source(
        self,
        source: str,
        evidence: EvidenceBundle,
        *,
        answer: str | None = None,
    ) -> None:
        self._require_running(source)
        self._validate_evidence_source(source, evidence)
        if not evidence.execution_success or not evidence.source_match or not evidence.items:
            raise ValueError("successful source evidence must be non-empty, matched, and execution-successful")
        self.evidence_by_source[source] = evidence
        if answer and answer.strip():
            self.answers_by_source[source] = answer.strip()
        self.completed_sources.add(source)
        self.failed_sources.discard(source)
        self.source_statuses[source] = SourceStatus.SUCCESS
        self.source_errors.pop(source, None)
        self._validate_sets()

    def fail_source(
        self,
        source: str,
        status: SourceStatus | str,
        *,
        evidence: EvidenceBundle | None = None,
        error: str | None = None,
    ) -> None:
        self._require_running(source)
        normalized = SourceStatus(status)
        if normalized not in _FAILURE_STATUSES:
            raise ValueError(f"not a terminal failure status: {normalized.value}")
        if evidence is not None:
            self._validate_evidence_source(source, evidence)
            self.evidence_by_source[source] = evidence
        self.completed_sources.discard(source)
        self.failed_sources.add(source)
        self.source_statuses[source] = normalized
        if error:
            self.source_errors[source] = error
        self._validate_sets()

    def record_fallback(self, fallback_from: str, fallback_to: str, reason: str) -> None:
        """Record fallback without mutating the query's original source requirement."""

        record = FallbackRecord(fallback_from, fallback_to, reason)
        if fallback_from not in self.tracked_sources:
            raise ValueError(f"fallback origin is not tracked: {fallback_from}")
        self.fallback_records.append(record)
        self.source_statuses.setdefault(fallback_to, SourceStatus.PENDING)

    @property
    def ready_for_synthesis(self) -> bool:
        return self.completed_sources.issuperset(self.required_sources)

    def evidence_count_by_source(self) -> dict[str, int]:
        return {
            source: len(self.evidence_by_source[source].items)
            if source in self.evidence_by_source
            else 0
            for source in sorted(self.tracked_sources)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_sources": sorted(self.required_sources),
            "completed_sources": sorted(self.completed_sources),
            "failed_sources": sorted(self.failed_sources),
            "source_attempts": dict(sorted(self.source_attempts.items())),
            "source_statuses": {
                source: status.value for source, status in sorted(self.source_statuses.items())
            },
            "evidence_count_by_source": self.evidence_count_by_source(),
            "fallback_records": [record.to_dict() for record in self.fallback_records],
        }

    def _require_source(self, source: str) -> None:
        if source not in self.tracked_sources:
            raise ValueError(f"source is neither required nor a recorded fallback: {source}")

    def _require_running(self, source: str) -> None:
        self._require_source(source)
        if self.source_statuses[source] is not SourceStatus.RUNNING:
            raise RuntimeError(f"source is not running: {source}")

    @staticmethod
    def _validate_evidence_source(source: str, evidence: EvidenceBundle) -> None:
        if evidence.source_type is None or evidence.source_type.value != source:
            raise ValueError(f"evidence bundle source does not match {source}")
        if any(item.source_type.value != source for item in evidence.items):
            raise ValueError(f"evidence item crossed the {source} source boundary")

    def _validate_sets(self) -> None:
        overlap = self.completed_sources.intersection(self.failed_sources)
        if overlap:
            raise ValueError(f"sources cannot be completed and failed: {sorted(overlap)}")
        outside = (self.completed_sources | self.failed_sources).difference(self.tracked_sources)
        if outside:
            raise ValueError(f"terminal sources are not required: {sorted(outside)}")
