"""Phase 4.1 structured trace contract and persistence tests."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from orchestration.entrypoint import OrchestrationEntrypoint
from orchestration.trace import (
    TRACE_SCHEMA_VERSION,
    JsonlTraceSink,
    TraceContext,
    TraceEvent,
    TraceRecorder,
    get_current_trace_context,
)


EXPECTED_ENVELOPE_KEYS = [
    "schema_version",
    "timestamp",
    "run_id",
    "session_id",
    "event_id",
    "event_type",
    "stage",
    "status",
    "task_id",
    "source",
    "attempt",
    "duration_ms",
    "error_type",
    "attributes",
]


class CollectingSink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


class FailingSink:
    def __init__(self):
        self.calls = 0

    def write(self, event):
        self.calls += 1
        raise OSError("trace storage unavailable")


def _event(context: TraceContext, event_type: str) -> TraceEvent:
    return TraceEvent.create(
        context,
        event_type=event_type,
        stage="test",
        status="success",
        task_id="T1",
        source="sql",
        attempt=1,
        duration_ms=12,
        attributes={"count": 2},
    )


def test_trace_event_schema_is_stable_and_json_compatible():
    event = _event(TraceContext(run_id="run-1", session_id="session-1"), "tool.completed")

    payload = event.to_dict()

    assert list(payload) == EXPECTED_ENVELOPE_KEYS
    assert payload["schema_version"] == TRACE_SCHEMA_VERSION == "1.0"
    assert payload["run_id"] == "run-1"
    assert payload["session_id"] == "session-1"
    assert payload["attributes"] == {"count": 2}
    assert datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00")).tzinfo
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_jsonl_contains_one_independently_loadable_object_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JsonlTraceSink(path)
    context = TraceContext(run_id="run-jsonl")

    sink.write(_event(context, "first"))
    sink.write(_event(context, "second"))

    lines = path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    assert len(payloads) == 2
    assert [payload["event_type"] for payload in payloads] == ["first", "second"]
    assert all(isinstance(payload, dict) for payload in payloads)


def test_events_in_one_run_share_run_id_and_lifecycle_is_recorded():
    sink = CollectingSink()
    recorder = TraceRecorder(sink)

    with recorder.run(session_id="session-1") as context:
        recorder.record(event_type="plan.created", stage="planning", status="success")
        recorder.record(event_type="task.started", stage="execution", status="started")

    assert [event.event_type for event in sink.events] == [
        "run.started",
        "plan.created",
        "task.started",
        "run.completed",
    ]
    assert {event.run_id for event in sink.events} == {context.run_id}
    assert {event.session_id for event in sink.events} == {"session-1"}
    assert len({event.event_id for event in sink.events}) == len(sink.events)
    assert get_current_trace_context() is None


def test_two_concurrent_async_runs_keep_run_ids_isolated():
    sink = CollectingSink()
    recorder = TraceRecorder(sink)

    async def run_one(name):
        with recorder.run(attributes={"name": name}) as context:
            await asyncio.sleep(0)
            assert get_current_trace_context() is context
            recorder.record(
                event_type="checkpoint",
                stage="execution",
                status="success",
                attributes={"name": name},
            )
            await asyncio.sleep(0)
            assert get_current_trace_context() is context
            return context.run_id

    async def run_both():
        return await asyncio.gather(run_one("a"), run_one("b"))

    run_ids = asyncio.run(run_both())

    assert len(set(run_ids)) == 2
    checkpoints = [event for event in sink.events if event.event_type == "checkpoint"]
    assert len(checkpoints) == 2
    assert {event.run_id for event in checkpoints} == set(run_ids)
    assert get_current_trace_context() is None


def test_trace_sink_failure_does_not_break_orchestration_business_logic():
    sink = FailingSink()
    recorder = TraceRecorder(sink)
    planner = Mock()
    planner.plan.side_effect = RuntimeError("force existing legacy fallback")
    legacy = AsyncMock(return_value="business answer")
    entrypoint = OrchestrationEntrypoint(
        planner,
        Mock(),
        legacy,
        trace_recorder=recorder,
    )

    answer = asyncio.run(entrypoint.run("question"))

    assert answer == "business answer"
    legacy.assert_awaited_once_with("question")
    assert sink.calls == 3


def test_trace_recorder_accepts_caller_run_id():
    recorder = TraceRecorder(enabled=False)

    with recorder.run(session_id="session-1", run_id="agui-run-1") as context:
        assert context.session_id == "session-1"
        assert context.run_id == "agui-run-1"


def test_jsonl_sink_appends_without_overwriting_history(tmp_path):
    path = tmp_path / "history.jsonl"
    context = TraceContext(run_id="run-append")

    JsonlTraceSink(path).write(_event(context, "historical"))
    JsonlTraceSink(path).write(_event(context, "new"))

    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [payload["event_type"] for payload in payloads] == ["historical", "new"]
