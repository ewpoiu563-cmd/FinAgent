"""Structured, fail-open tracing primitives for orchestration runs."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = "1.0"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _identifier() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Identity shared by every event emitted during one orchestration run."""

    run_id: str = field(default_factory=_identifier)
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id.strip()
        ):
            raise ValueError("session_id must be None or a non-empty string")


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Versioned event envelope persisted by trace sinks.

    Optional fields remain present with a JSON ``null`` value. This keeps the
    envelope shape stable across event types while preserving their optional
    semantics.
    """

    schema_version: str
    timestamp: str
    run_id: str
    session_id: str | None
    event_id: str
    event_type: str
    stage: str
    status: str
    task_id: str | None = None
    source: str | None = None
    attempt: int | None = None
    duration_ms: int | float | None = None
    error_type: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "schema_version",
            "timestamp",
            "run_id",
            "event_id",
            "event_type",
            "stage",
            "status",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.attempt is not None and (
            not isinstance(self.attempt, int)
            or isinstance(self.attempt, bool)
            or self.attempt < 0
        ):
            raise ValueError("attempt must be a non-negative integer or None")
        if self.duration_ms is not None and (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, (int, float))
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be non-negative or None")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("attributes must be a mapping")
        object.__setattr__(self, "attributes", dict(self.attributes))

    @classmethod
    def create(
        cls,
        context: TraceContext,
        *,
        event_type: str,
        stage: str,
        status: str,
        task_id: str | None = None,
        source: str | None = None,
        attempt: int | None = None,
        duration_ms: int | float | None = None,
        error_type: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> "TraceEvent":
        return cls(
            schema_version=TRACE_SCHEMA_VERSION,
            timestamp=_utc_timestamp(),
            run_id=context.run_id,
            session_id=context.session_id,
            event_id=_identifier(),
            event_type=event_type,
            stage=stage,
            status=status,
            task_id=task_id,
            source=source,
            attempt=attempt,
            duration_ms=duration_ms,
            error_type=error_type,
            attributes=attributes or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible envelope in stable key order."""

        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "stage": self.stage,
            "status": self.status,
            "task_id": self.task_id,
            "source": self.source,
            "attempt": self.attempt,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "attributes": dict(self.attributes),
        }


class TraceSink(Protocol):
    def write(self, event: TraceEvent) -> None:
        """Persist one event or raise an exception handled by TraceRecorder."""


class RuntimeArtifactObserver(Protocol):
    """Optional read-only observer outside the versioned TraceEvent schema."""

    def observe(self, kind: str, value: Any, metadata: Mapping[str, Any]) -> None: ...


class JsonlTraceSink:
    """Append one compact JSON object per line to a dedicated trace file."""

    def __init__(
        self,
        path: str | Path,
        *,
        summary_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.summary_path = Path(summary_path) if summary_path is not None else None
        self._lock = threading.Lock()

    def write(self, event: TraceEvent) -> None:
        line = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.write("\n")
            if event.event_type == "run.completed" and self.summary_path is not None:
                # Import lazily to avoid a trace <-> summary module cycle. The
                # recorder remains fail-open if summary derivation/persistence
                # fails after the raw terminal event has been safely appended.
                from .trace_summary import TraceSummaryBuilder

                builder = TraceSummaryBuilder()
                summary = builder.build_from_jsonl(self.path, run_id=event.run_id)
                builder.persist(summary, self.summary_path)


_CURRENT_TRACE_CONTEXT: ContextVar[TraceContext | None] = ContextVar(
    "finagent_trace_context",
    default=None,
)
_CURRENT_TRACE_RECORDER: ContextVar["TraceRecorder | None"] = ContextVar(
    "finagent_trace_recorder",
    default=None,
)
_CURRENT_TRACE_RUN_OUTCOME: ContextVar[str | None] = ContextVar(
    "finagent_trace_run_outcome",
    default=None,
)
_CURRENT_TRACE_TASK_ID: ContextVar[str | None] = ContextVar(
    "finagent_trace_task_id",
    default=None,
)
_CURRENT_TRACE_SOURCE: ContextVar[str | None] = ContextVar(
    "finagent_trace_source",
    default=None,
)
_CURRENT_RUNTIME_ARTIFACT_OBSERVER: ContextVar[RuntimeArtifactObserver | None] = ContextVar(
    "finagent_runtime_artifact_observer",
    default=None,
)


def get_current_trace_context() -> TraceContext | None:
    """Return the context bound to the current thread or async task."""

    return _CURRENT_TRACE_CONTEXT.get()


def get_current_trace_task_id() -> str | None:
    """Return the atomic task currently executing, when one is bound."""

    return _CURRENT_TRACE_TASK_ID.get()


def get_current_trace_source() -> str | None:
    """Return the source currently executing, when one is bound."""

    return _CURRENT_TRACE_SOURCE.get()


@contextmanager
def runtime_artifact_observer(observer: RuntimeArtifactObserver) -> Iterator[None]:
    """Bind evaluation instrumentation without changing production behavior."""

    token = _CURRENT_RUNTIME_ARTIFACT_OBSERVER.set(observer)
    try:
        yield
    finally:
        _CURRENT_RUNTIME_ARTIFACT_OBSERVER.reset(token)


def notify_runtime_artifact(kind: str, value: Any) -> None:
    """Offer a runtime object to a bound observer; failures are always dropped."""

    observer = _CURRENT_RUNTIME_ARTIFACT_OBSERVER.get()
    if observer is None:
        return
    context = get_current_trace_context()
    try:
        observer.observe(
            kind,
            value,
            {
                "run_id": context.run_id if context else None,
                "session_id": context.session_id if context else None,
                "task_id": get_current_trace_task_id(),
                "source": get_current_trace_source(),
            },
        )
    except Exception as error:
        logger.debug("Runtime artifact observer dropped artifact: %s", type(error).__name__)


@contextmanager
def trace_task_scope(task_id: str | None) -> Iterator[None]:
    """Make a task id available to nested policies without changing signatures."""

    token = _CURRENT_TRACE_TASK_ID.set(task_id)
    try:
        yield
    finally:
        _CURRENT_TRACE_TASK_ID.reset(token)


@contextmanager
def trace_execution_scope(
    *,
    task_id: str | None = None,
    source: str | None = None,
) -> Iterator[None]:
    """Bind nested task/source identity without leaking into sibling tasks."""

    task_token = _CURRENT_TRACE_TASK_ID.set(task_id)
    source_token = _CURRENT_TRACE_SOURCE.set(source)
    try:
        yield
    finally:
        _CURRENT_TRACE_SOURCE.reset(source_token)
        _CURRENT_TRACE_TASK_ID.reset(task_token)


class TraceRecorder:
    """Build and persist events without allowing trace failures to escape."""

    def __init__(self, sink: TraceSink | None = None, *, enabled: bool = True) -> None:
        self.sink = sink
        self.enabled = enabled

    @contextmanager
    def bind(self, context: TraceContext) -> Iterator[TraceContext]:
        """Bind a context for nested synchronous and asynchronous work."""

        context_token: Token[TraceContext | None] = _CURRENT_TRACE_CONTEXT.set(context)
        recorder_token: Token[TraceRecorder | None] = _CURRENT_TRACE_RECORDER.set(self)
        try:
            yield context
        finally:
            _CURRENT_TRACE_RECORDER.reset(recorder_token)
            _CURRENT_TRACE_CONTEXT.reset(context_token)

    @contextmanager
    def run(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[TraceContext]:
        """Create an isolated run context and emit its outer lifecycle."""

        context = (
            TraceContext(session_id=session_id)
            if run_id is None
            else TraceContext(run_id=run_id, session_id=session_id)
        )
        started = time.perf_counter()
        outcome_token = _CURRENT_TRACE_RUN_OUTCOME.set(None)
        try:
            with self.bind(context):
                self.record(
                    event_type="run.started",
                    stage="orchestration",
                    status="started",
                    attributes=attributes,
                )
                try:
                    yield context
                except BaseException as error:
                    self.record(
                        event_type="run.completed",
                        stage="orchestration",
                        status="error",
                        duration_ms=round((time.perf_counter() - started) * 1000),
                        error_type=type(error).__name__,
                        attributes={
                            "outcome": _CURRENT_TRACE_RUN_OUTCOME.get()
                            or "unhandled_error"
                        },
                    )
                    raise
                else:
                    self.record(
                        event_type="run.completed",
                        stage="orchestration",
                        status="completed",
                        duration_ms=round((time.perf_counter() - started) * 1000),
                        attributes={
                            "outcome": _CURRENT_TRACE_RUN_OUTCOME.get() or "unknown"
                        },
                    )
        finally:
            _CURRENT_TRACE_RUN_OUTCOME.reset(outcome_token)

    def set_run_outcome(self, outcome: str, *, only_if_unset: bool = False) -> None:
        """Attach a business outcome to the eventual ``run.completed`` event."""

        if isinstance(outcome, str) and outcome.strip():
            if only_if_unset and _CURRENT_TRACE_RUN_OUTCOME.get() is not None:
                return
            _CURRENT_TRACE_RUN_OUTCOME.set(outcome.strip())

    def record(
        self,
        *,
        event_type: str,
        stage: str,
        status: str,
        task_id: str | None = None,
        source: str | None = None,
        attempt: int | None = None,
        duration_ms: int | float | None = None,
        error_type: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceEvent | None:
        context = get_current_trace_context()
        if not self.enabled or self.sink is None or context is None:
            return None
        try:
            event = TraceEvent.create(
                context,
                event_type=event_type,
                stage=stage,
                status=status,
                task_id=task_id if task_id is not None else get_current_trace_task_id(),
                source=source if source is not None else get_current_trace_source(),
                attempt=attempt,
                duration_ms=duration_ms,
                error_type=error_type,
                attributes=attributes,
            )
            self.sink.write(event)
            return event
        except Exception as error:
            # Observability must never become an availability dependency.
            logger.debug("Trace event dropped: %s", type(error).__name__)
            return None


def record_trace_event(**event: Any) -> TraceEvent | None:
    """Record through the recorder bound to the current context, if any."""

    recorder = _CURRENT_TRACE_RECORDER.get()
    if recorder is None:
        return None
    return recorder.record(**event)


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "JsonlTraceSink",
    "TraceContext",
    "TraceEvent",
    "TraceRecorder",
    "TraceSink",
    "RuntimeArtifactObserver",
    "get_current_trace_context",
    "get_current_trace_source",
    "get_current_trace_task_id",
    "notify_runtime_artifact",
    "record_trace_event",
    "runtime_artifact_observer",
    "trace_execution_scope",
    "trace_task_scope",
]
