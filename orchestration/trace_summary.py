"""Deterministic per-run summaries derived only from structured trace events."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .trace import TraceEvent


SUMMARY_SCHEMA_VERSION = "1.1"
_PERSIST_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Stable JSON-compatible summary of one run's trace events.

    Latency fields ending in ``_sum`` are sums of terminal-event durations. They
    are not an exclusive wall-clock breakdown and need not equal ``duration_ms``.
    """

    summary_schema_version: str
    run_id: str
    session_id: str | None
    started_at: str | None
    completed_at: str | None
    lifecycle_status: str
    business_outcome: str | None
    duration_ms: int | float | None
    event_count: int
    trace_complete: bool
    requirement_count: int | None
    required_requirement_count: int | None
    planned_primary_source: str | None
    planned_required_sources: tuple[str, ...]
    routing_method: str | None
    dispatch: str | None
    required_task_count: int
    completed_task_count: int
    incomplete_task_count: int
    requirement_coverage: float | None
    actual_sources: tuple[str, ...]
    source_attempt_count_by_source: Mapping[str, int]
    source_success_count_by_source: Mapping[str, int]
    source_failure_count_by_source: Mapping[str, int]
    source_statuses: Mapping[str, tuple[str, ...]]
    fallback_count: int
    fallback_transitions: Mapping[str, int]
    logical_llm_call_count: int
    successful_llm_call_count: int
    failed_llm_call_count: int
    incomplete_llm_call_count: int
    llm_attempt_count: int
    llm_retry_count: int
    retried_logical_llm_call_count: int | None
    llm_latency_ms_sum: int | float
    llm_calls_by_operation: Mapping[str, int]
    provider_reported_input_tokens: int | None
    provider_reported_output_tokens: int | None
    provider_reported_total_tokens: int | None
    token_usage_known_call_count: int
    token_usage_unknown_call_count: int
    tool_call_count: int
    successful_tool_call_count: int
    failed_tool_call_count: int
    incomplete_tool_call_count: int
    tool_latency_ms_sum: int | float
    tool_calls_by_name: Mapping[str, int]
    sql_compilation_count: int
    sql_compilation_failure_count: int
    incomplete_sql_compilation_count: int
    sql_compilation_latency_ms_sum: int | float
    synthesis_status: str | None
    synthesis_duration_ms: int | float | None
    dependency_failure: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation in schema field order."""

        result: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Mapping):
                value = {
                    key: list(item) if isinstance(item, tuple) else item
                    for key, item in value.items()
                }
            result[name] = value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunSummary":
        """Load known fields while tolerating fields from future schemas."""

        if not isinstance(value, Mapping):
            raise TypeError("RunSummary payload must be a mapping")
        known_fields = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in known_fields}
        if "retried_logical_llm_call_count" not in payload:
            retry_count = payload.get("llm_retry_count")
            # A legacy 1.0 summary with no retries is exactly known. If retries
            # occurred, their distribution across logical calls is unknowable.
            payload["retried_logical_llm_call_count"] = (
                0 if retry_count == 0 else None
            )
        for name in ("planned_required_sources", "actual_sources"):
            value_list = payload.get(name)
            if isinstance(value_list, list):
                payload[name] = tuple(value_list)
        statuses = payload.get("source_statuses")
        if isinstance(statuses, Mapping):
            payload["source_statuses"] = {
                key: tuple(item) if isinstance(item, list) else item
                for key, item in statuses.items()
            }
        return cls(**payload)


class TraceSummaryBuilder:
    """Pure aggregation and idempotent JSONL persistence for run summaries."""

    def build(
        self,
        events: Iterable[TraceEvent | Mapping[str, Any]],
        *,
        run_id: str | None = None,
    ) -> RunSummary:
        normalized = [_normalize_event(event) for event in events]
        normalized = [event for event in normalized if event is not None]
        if run_id is None:
            run_ids = list(
                dict.fromkeys(
                    event["run_id"]
                    for event in normalized
                    if _nonempty_string(event.get("run_id")) is not None
                )
            )
            if len(run_ids) != 1:
                raise ValueError("run_id is required unless events contain exactly one run")
            run_id = run_ids[0]
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")

        run_events = [event for event in normalized if event.get("run_id") == run_id]
        if not run_events:
            raise ValueError(f"no trace events found for run_id {run_id!r}")
        return self._aggregate(run_id, run_events)

    def build_all(
        self, events: Iterable[TraceEvent | Mapping[str, Any]]
    ) -> tuple[RunSummary, ...]:
        normalized = [_normalize_event(event) for event in events]
        normalized = [event for event in normalized if event is not None]
        run_ids = list(
            dict.fromkeys(
                event["run_id"]
                for event in normalized
                if _nonempty_string(event.get("run_id")) is not None
            )
        )
        return tuple(
            self._aggregate(
                run_id,
                [event for event in normalized if event.get("run_id") == run_id],
            )
            for run_id in run_ids
        )

    def build_from_jsonl(
        self, path: str | Path, *, run_id: str | None = None
    ) -> RunSummary:
        return self.build(self.read_trace_jsonl(path), run_id=run_id)

    @staticmethod
    def read_trace_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
        """Read valid object lines; malformed/partial lines are ignored."""

        events: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(value, dict):
                    events.append(value)
        return tuple(events)

    @staticmethod
    def persist(summary: RunSummary, path: str | Path) -> None:
        """Upsert one summary by run_id using an atomic full-file rebuild.

        Re-summarizing a run replaces its previous row instead of appending an
        indistinguishable duplicate. Other run summaries retain their order.
        """

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = summary.to_dict()
        with _PERSIST_LOCK:
            existing: list[dict[str, Any]] = []
            if destination.exists():
                with destination.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError("summary JSONL lines must be objects")
                        if value.get("run_id") != summary.run_id:
                            existing.append(value)
            existing.append(payload)
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_name = stream.name
                    for item in existing:
                        stream.write(
                            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        )
                        stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, destination)
            finally:
                if temporary_name and os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def _aggregate(self, run_id: str, events: list[dict[str, Any]]) -> RunSummary:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            by_type.setdefault(str(event.get("event_type") or ""), []).append(event)

        run_started = _first(by_type.get("run.started", ()))
        run_completed = _last(by_type.get("run.completed", ()))
        decomposition = _last(by_type.get("planning.task_decomposition.completed", ()))
        source_plan = _last(by_type.get("planning.source_plan.completed", ()))
        decomposition_attributes = _attributes(decomposition)
        plan_attributes = _attributes(source_plan)

        source_events = [
            event
            for event in events
            if event.get("event_type") in {
                "source.execution.started",
                "source.execution.completed",
                "source.execution.failed",
            }
        ]
        actual_sources = tuple(
            dict.fromkeys(
                source
                for event in source_events
                if (source := _nonempty_string(event.get("source"))) is not None
            )
        )
        source_starts = by_type.get("source.execution.started", [])
        source_completed = by_type.get("source.execution.completed", [])
        source_failed = by_type.get("source.execution.failed", [])
        source_terminals = [
            event
            for event in events
            if event.get("event_type")
            in {"source.execution.completed", "source.execution.failed"}
        ]
        source_attempts = _count_by(source_starts, "source")
        source_successes = _count_by(source_completed, "source")
        source_failures = _count_by(source_failed, "source")
        source_statuses: dict[str, list[str]] = {}
        for event in source_terminals:
            source = _nonempty_string(event.get("source")) or "unknown"
            status = _nonempty_string(event.get("status")) or "unknown"
            source_statuses.setdefault(source, []).append(status)
        source_keys = tuple(
            dict.fromkeys(
                (*actual_sources, *source_attempts, *source_successes, *source_failures)
            )
        )
        source_attempts = {source: source_attempts.get(source, 0) for source in source_keys}
        source_successes = {source: source_successes.get(source, 0) for source in source_keys}
        source_failures = {source: source_failures.get(source, 0) for source in source_keys}
        source_statuses = {
            source: source_statuses.get(source, []) for source in source_keys
        }

        fallback_events = by_type.get("fallback.dispatched", [])
        fallback_transitions: Counter[str] = Counter()
        for event in fallback_events:
            attributes = _attributes(event)
            from_source = (
                _nonempty_string(attributes.get("from_source"))
                or _nonempty_string(event.get("source"))
                or _nonempty_string(attributes.get("from_stage"))
                or "unknown"
            )
            to_source = (
                _nonempty_string(attributes.get("to_source"))
                or _nonempty_string(attributes.get("target"))
                or "unknown"
            )
            fallback_transitions[f"{from_source}->{to_source}"] += 1

        task_events = [
            event
            for event in events
            if event.get("event_type")
            in {
                "task.execution.started",
                "task.execution.completed",
                "task.execution.failed",
            }
        ]
        observed_task_ids = list(
            dict.fromkeys(
                task_id
                for event in task_events
                if (task_id := _nonempty_string(event.get("task_id"))) is not None
            )
        )
        explicit_required_ids = _string_list(plan_attributes.get("required_task_ids"))
        required_task_ids = explicit_required_ids or observed_task_ids
        explicit_required_count = _nonnegative_int(plan_attributes.get("required_task_count"))
        if explicit_required_count is None:
            # Phase 4.1-4.3 traces already recorded the number of atomic
            # execution tasks, but not their required IDs. It is the safest
            # historical fallback and still excludes synthesis tasks.
            explicit_required_count = _nonnegative_int(plan_attributes.get("task_count"))
        if explicit_required_count is None:
            explicit_required_count = len(required_task_ids)
        required_task_count = explicit_required_count
        task_terminal_by_id: dict[str, str] = {}
        for event in task_events:
            if event.get("event_type") not in {
                "task.execution.completed",
                "task.execution.failed",
            }:
                continue
            task_id = _nonempty_string(event.get("task_id"))
            if task_id is not None:
                task_terminal_by_id[task_id] = str(event.get("event_type"))
        completed_task_count = sum(
            task_terminal_by_id.get(task_id) == "task.execution.completed"
            for task_id in required_task_ids
        )
        incomplete_task_count = max(required_task_count - completed_task_count, 0)
        requirement_coverage = (
            completed_task_count / required_task_count if required_task_count else None
        )

        llm_completed = by_type.get("llm.call.completed", [])
        llm_failed = by_type.get("llm.call.failed", [])
        llm_terminals = [
            event
            for event in events
            if event.get("event_type") in {"llm.call.completed", "llm.call.failed"}
        ]
        llm_attempt_count = 0
        llm_retry_count = 0
        retried_logical_llm_call_count = 0
        llm_operations: Counter[str] = Counter()
        known_usage_count = 0
        input_tokens: list[int] = []
        output_tokens: list[int] = []
        total_tokens: list[int] = []
        for event in llm_terminals:
            attributes = _attributes(event)
            attempt_count = _nonnegative_int(attributes.get("attempt_count"))
            if attempt_count is None:
                attempt_count = _nonnegative_int(event.get("attempt")) or 1
            retry_count = _nonnegative_int(attributes.get("retry_count"))
            if retry_count is None:
                retry_count = max(attempt_count - 1, 0)
            llm_attempt_count += attempt_count
            llm_retry_count += retry_count
            if retry_count > 0:
                retried_logical_llm_call_count += 1
            llm_operations[_nonempty_string(attributes.get("operation")) or "unknown"] += 1
            usage = tuple(
                _nonnegative_int(attributes.get(name))
                for name in ("input_tokens", "output_tokens", "total_tokens")
            )
            if all(value is not None for value in usage):
                known_usage_count += 1
            if usage[0] is not None:
                input_tokens.append(usage[0])
            if usage[1] is not None:
                output_tokens.append(usage[1])
            if usage[2] is not None:
                total_tokens.append(usage[2])

        tool_completed = by_type.get("tool.call.completed", [])
        tool_failed = by_type.get("tool.call.failed", [])
        tool_terminals = [
            event
            for event in events
            if event.get("event_type") in {"tool.call.completed", "tool.call.failed"}
        ]
        tool_names = Counter(
            _nonempty_string(_attributes(event).get("tool_name")) or "unknown"
            for event in tool_terminals
        )

        compilation_completed = by_type.get("sql.compilation.completed", [])
        compilation_failed = by_type.get("sql.compilation.failed", [])
        compilation_terminals = [
            event
            for event in events
            if event.get("event_type")
            in {"sql.compilation.completed", "sql.compilation.failed"}
        ]

        synthesis_events = [
            event
            for event in events
            if event.get("event_type")
            in {"synthesis.started", "synthesis.completed", "synthesis.failed"}
        ]
        synthesis_terminals = [
            event
            for event in synthesis_events
            if event.get("event_type") in {"synthesis.completed", "synthesis.failed"}
        ]
        synthesis_status = None
        if synthesis_terminals:
            synthesis_status = _nonempty_string(synthesis_terminals[-1].get("status"))
        elif synthesis_events:
            synthesis_status = "incomplete"

        run_attributes = _attributes(run_completed)
        started_session = run_started.get("session_id") if run_started else events[0].get("session_id")
        return RunSummary(
            summary_schema_version=SUMMARY_SCHEMA_VERSION,
            run_id=run_id,
            session_id=_nonempty_string(started_session),
            started_at=_nonempty_string(run_started.get("timestamp")) if run_started else None,
            completed_at=(
                _nonempty_string(run_completed.get("timestamp")) if run_completed else None
            ),
            lifecycle_status=(
                _nonempty_string(run_completed.get("status")) or "incomplete"
                if run_completed
                else "incomplete"
            ),
            business_outcome=_nonempty_string(run_attributes.get("outcome")),
            duration_ms=_duration(run_completed),
            event_count=len(events),
            trace_complete=run_started is not None and run_completed is not None,
            requirement_count=_nonnegative_int(decomposition_attributes.get("requirement_count")),
            required_requirement_count=_nonnegative_int(
                decomposition_attributes.get("required_requirement_count")
            ),
            planned_primary_source=_nonempty_string(plan_attributes.get("primary_source")),
            planned_required_sources=tuple(_string_list(plan_attributes.get("required_sources"))),
            routing_method=_nonempty_string(plan_attributes.get("routing_method")),
            dispatch=_nonempty_string(plan_attributes.get("dispatch")),
            required_task_count=required_task_count,
            completed_task_count=completed_task_count,
            incomplete_task_count=incomplete_task_count,
            requirement_coverage=requirement_coverage,
            actual_sources=actual_sources,
            source_attempt_count_by_source=source_attempts,
            source_success_count_by_source=source_successes,
            source_failure_count_by_source=source_failures,
            source_statuses={key: tuple(value) for key, value in source_statuses.items()},
            fallback_count=len(fallback_events),
            fallback_transitions=dict(fallback_transitions),
            logical_llm_call_count=len(llm_terminals),
            successful_llm_call_count=len(llm_completed),
            failed_llm_call_count=len(llm_failed),
            incomplete_llm_call_count=_incomplete_count(
                by_type.get("llm.call.started", []), llm_terminals, _llm_key
            ),
            llm_attempt_count=llm_attempt_count,
            llm_retry_count=llm_retry_count,
            retried_logical_llm_call_count=retried_logical_llm_call_count,
            llm_latency_ms_sum=_duration_sum(llm_terminals),
            llm_calls_by_operation=dict(llm_operations),
            provider_reported_input_tokens=sum(input_tokens) if input_tokens else None,
            provider_reported_output_tokens=sum(output_tokens) if output_tokens else None,
            provider_reported_total_tokens=sum(total_tokens) if total_tokens else None,
            token_usage_known_call_count=known_usage_count,
            token_usage_unknown_call_count=len(llm_terminals) - known_usage_count,
            tool_call_count=len(tool_terminals),
            successful_tool_call_count=len(tool_completed),
            failed_tool_call_count=len(tool_failed),
            incomplete_tool_call_count=_incomplete_count(
                by_type.get("tool.call.started", []), tool_terminals, _tool_key
            ),
            tool_latency_ms_sum=_duration_sum(tool_terminals),
            tool_calls_by_name=dict(tool_names),
            sql_compilation_count=len(compilation_terminals),
            sql_compilation_failure_count=len(compilation_failed),
            incomplete_sql_compilation_count=_incomplete_count(
                by_type.get("sql.compilation.started", []),
                compilation_terminals,
                _compilation_key,
            ),
            sql_compilation_latency_ms_sum=_duration_sum(compilation_terminals),
            synthesis_status=synthesis_status,
            synthesis_duration_ms=(
                _duration_sum(synthesis_terminals) if synthesis_terminals else None
            ),
            dependency_failure=any(
                event.get("status") == "dependency_failure"
                or _attributes(event).get("failure_status") == "dependency_failure"
                for event in synthesis_events
            ),
        )


def _normalize_event(event: Any) -> dict[str, Any] | None:
    if isinstance(event, TraceEvent):
        return event.to_dict()
    if isinstance(event, Mapping):
        return dict(event)
    return None


def _attributes(event: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not event:
        return {}
    value = event.get("attributes")
    return value if isinstance(value, Mapping) else {}


def _first(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    return next(iter(events), None)


def _last(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(events)
    return values[-1] if values else None


def _nonempty_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _duration(event: Mapping[str, Any] | None) -> int | float | None:
    if not event:
        return None
    value = event.get("duration_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _duration_sum(events: Iterable[Mapping[str, Any]]) -> int | float:
    return sum(value for event in events if (value := _duration(event)) is not None)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for raw in value if (item := _nonempty_string(raw)) is not None]


def _count_by(events: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(
        Counter(_nonempty_string(event.get(field)) or "unknown" for event in events)
    )


def _llm_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    attributes = _attributes(event)
    return (
        event.get("task_id"),
        event.get("source"),
        attributes.get("operation"),
        attributes.get("model"),
    )


def _tool_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("task_id"),
        event.get("source"),
        _attributes(event).get("tool_name"),
    )


def _compilation_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    attributes = _attributes(event)
    return (event.get("task_id"), event.get("source"), attributes.get("compiler"))


def _incomplete_count(starts, terminals, key) -> int:
    started = Counter(key(event) for event in starts)
    terminal = Counter(key(event) for event in terminals)
    return sum(max(count - terminal.get(item, 0), 0) for item, count in started.items())


__all__ = ["SUMMARY_SCHEMA_VERSION", "RunSummary", "TraceSummaryBuilder"]
