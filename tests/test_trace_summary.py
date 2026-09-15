"""Phase 4.4 deterministic run-summary coverage."""

from __future__ import annotations

import json

import pytest

from orchestration import JsonlTraceSink, TraceRecorder
from orchestration.trace import TraceContext, TraceEvent
from orchestration.trace_summary import (
    SUMMARY_SCHEMA_VERSION,
    TraceSummaryBuilder,
)


def _event(
    event_type,
    *,
    run_id="run-1",
    status=None,
    task_id=None,
    source=None,
    attempt=None,
    duration_ms=None,
    attributes=None,
    session_id=None,
):
    return {
        "schema_version": "1.0",
        "timestamp": "2026-09-10T00:00:00.000Z",
        "run_id": run_id,
        "session_id": session_id,
        "event_id": f"{run_id}-{event_type}-{task_id}-{source}",
        "event_type": event_type,
        "stage": event_type.split(".")[0],
        "status": status or ("started" if event_type.endswith("started") else "completed"),
        "task_id": task_id,
        "source": source,
        "attempt": attempt,
        "duration_ms": duration_ms,
        "error_type": None,
        "attributes": attributes,
    }


def _plan(*, run_id="run-1", sources=("sql",), task_ids=("T1",), requirements=1):
    return [
        _event("run.started", run_id=run_id),
        _event(
            "planning.task_decomposition.completed",
            run_id=run_id,
            attributes={
                "requirement_count": requirements,
                "required_requirement_count": requirements,
            },
        ),
        _event(
            "planning.source_plan.completed",
            run_id=run_id,
            attributes={
                "primary_source": sources[0] if sources else None,
                "required_sources": list(sources),
                "routing_method": "test",
                "dispatch": "requirement_plan",
                "required_task_count": len(task_ids),
                "required_task_ids": list(task_ids),
            },
        ),
    ]


def _task_success(task_id="T1", source="sql", *, run_id="run-1", duration=5):
    return [
        _event("task.execution.started", run_id=run_id, task_id=task_id, source=source),
        _event(
            "source.execution.started",
            run_id=run_id,
            task_id=task_id,
            source=source,
            attempt=1,
        ),
        _event(
            "source.execution.completed",
            run_id=run_id,
            task_id=task_id,
            source=source,
            attempt=1,
            status="success",
            duration_ms=duration,
        ),
        _event(
            "task.execution.completed",
            run_id=run_id,
            task_id=task_id,
            source=source,
            status="success",
            duration_ms=duration,
        ),
    ]


def _complete(*, run_id="run-1", outcome="success", duration=20):
    return _event(
        "run.completed",
        run_id=run_id,
        status="completed",
        duration_ms=duration,
        attributes={"outcome": outcome},
    )


def test_single_no_tool_direct_run_summary():
    events = _plan(sources=("direct",))
    events += [
        _event("task.execution.started", task_id="T1", source="direct"),
        _event(
            "task.execution.completed", task_id="T1", source="direct", status="success"
        ),
        _complete(),
    ]

    summary = TraceSummaryBuilder().build(events)

    assert summary.summary_schema_version == SUMMARY_SCHEMA_VERSION == "1.1"
    assert summary.required_task_count == summary.completed_task_count == 1
    assert summary.requirement_coverage == 1.0
    assert summary.actual_sources == ()
    assert summary.trace_complete is True


def test_single_sql_success_summary():
    summary = TraceSummaryBuilder().build(_plan() + _task_success() + [_complete()])

    assert summary.planned_primary_source == "sql"
    assert summary.actual_sources == ("sql",)
    assert summary.source_attempt_count_by_source == {"sql": 1}
    assert summary.source_success_count_by_source == {"sql": 1}
    assert summary.source_statuses == {"sql": ("success",)}


def test_compound_sql_two_requirements_are_both_covered():
    events = _plan(task_ids=("T1", "T2"), requirements=2)
    events += _task_success("T1") + _task_success("T2")
    events += [
        _event("synthesis.started", task_id="SYNTHESIS"),
        _event(
            "synthesis.completed",
            task_id="SYNTHESIS",
            status="success",
            duration_ms=4,
        ),
        _complete(),
    ]

    summary = TraceSummaryBuilder().build(events)

    assert summary.required_task_count == 2
    assert summary.completed_task_count == 2
    assert summary.requirement_coverage == 1.0
    assert summary.synthesis_status == "success"


def test_sql_to_web_fallback_keeps_planned_and_actual_sources_distinct():
    events = _plan(sources=("sql",))
    events += [
        _event("task.execution.started", task_id="T1", source="sql"),
        _event("source.execution.started", task_id="T1", source="sql", attempt=1),
        _event(
            "source.execution.failed",
            task_id="T1",
            source="sql",
            attempt=1,
            status="insufficient",
        ),
        _event(
            "fallback.dispatched",
            task_id="T1",
            source="sql",
            status="dispatched",
            attributes={"from_source": "sql", "to_source": "web"},
        ),
        _event("source.execution.started", task_id="T1", source="web", attempt=1),
        _event(
            "source.execution.completed",
            task_id="T1",
            source="web",
            attempt=1,
            status="success",
        ),
        _event(
            "task.execution.completed", task_id="T1", source="sql", status="success"
        ),
        _complete(),
    ]

    summary = TraceSummaryBuilder().build(events)

    assert summary.planned_required_sources == ("sql",)
    assert summary.actual_sources == ("sql", "web")
    assert summary.source_success_count_by_source == {"sql": 0, "web": 1}
    assert summary.source_failure_count_by_source == {"sql": 1, "web": 0}
    assert summary.fallback_count == 1
    assert summary.fallback_transitions == {"sql->web": 1}


def test_partial_success_uses_atomic_required_task_coverage():
    events = _plan(task_ids=("T1", "T2"), requirements=2)
    events += _task_success("T1")
    events += [
        _event("task.execution.started", task_id="T2", source="sql"),
        _event(
            "task.execution.failed",
            task_id="T2",
            source="sql",
            status="insufficient",
        ),
        _event(
            "synthesis.failed",
            task_id="SYNTHESIS",
            status="dependency_failure",
            attributes={"failure_status": "dependency_failure"},
        ),
        _complete(outcome="partial_success"),
    ]

    summary = TraceSummaryBuilder().build(events)

    assert summary.completed_task_count == 1
    assert summary.incomplete_task_count == 1
    assert summary.requirement_coverage == 0.5
    assert summary.dependency_failure is True


def test_planning_time_hybrid_is_not_fallback():
    events = _plan(sources=("sql", "web"), task_ids=("T1", "T2"), requirements=2)
    events += _task_success("T1", "sql") + _task_success("T2", "web") + [_complete()]

    summary = TraceSummaryBuilder().build(events)

    assert summary.actual_sources == ("sql", "web")
    assert summary.fallback_count == 0
    assert summary.fallback_transitions == {}


def test_llm_retry_is_one_logical_call_and_two_attempts():
    events = _plan() + [
        _event(
            "llm.call.started",
            attributes={"operation": "route", "model": "test"},
        ),
        _event(
            "llm.call.completed",
            duration_ms=9,
            attributes={
                "operation": "route",
                "model": "test",
                "attempt_count": 2,
                "retry_count": 1,
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
        ),
        _complete(),
    ]

    summary = TraceSummaryBuilder().build(events)

    assert summary.logical_llm_call_count == 1
    assert summary.llm_attempt_count == 2
    assert summary.llm_retry_count == 1
    assert summary.retried_logical_llm_call_count == 1
    assert summary.llm_calls_by_operation == {"route": 1}


def test_partial_provider_usage_never_converts_null_to_zero():
    events = _plan() + [
        _event(
            "llm.call.completed",
            attributes={
                "operation": "known",
                "attempt_count": 1,
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            },
        ),
        _event(
            "llm.call.completed",
            attributes={
                "operation": "unknown",
                "attempt_count": 1,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        ),
        _complete(),
    ]

    summary = TraceSummaryBuilder().build(events)

    assert summary.provider_reported_total_tokens == 8
    assert summary.token_usage_known_call_count == 1
    assert summary.token_usage_unknown_call_count == 1

    only_unknown = TraceSummaryBuilder().build(
        _plan() + [events[-2], _complete()]
    )
    assert only_unknown.provider_reported_input_tokens is None
    assert only_unknown.provider_reported_total_tokens is None


def test_sql_compilation_is_not_an_llm_or_tool_call():
    events = _plan() + [
        _event("sql.compilation.started", task_id="T1", source="sql"),
        _event(
            "sql.compilation.completed",
            task_id="T1",
            source="sql",
            duration_ms=7,
        ),
        _complete(),
    ]

    summary = TraceSummaryBuilder().build(events)

    assert summary.sql_compilation_count == 1
    assert summary.sql_compilation_latency_ms_sum == 7
    assert summary.logical_llm_call_count == summary.tool_call_count == 0


def test_nested_latencies_are_sums_not_a_run_duration_invariant():
    events = _plan() + [
        _event("llm.call.completed", duration_ms=80, attributes={"attempt_count": 1}),
        _event("tool.call.completed", duration_ms=90, attributes={"tool_name": "query"}),
        _event("sql.compilation.completed", duration_ms=70),
        _complete(duration=100),
    ]
    summary = TraceSummaryBuilder().build(events)

    assert summary.duration_ms == 100
    assert summary.llm_latency_ms_sum + summary.tool_latency_ms_sum + summary.sql_compilation_latency_ms_sum == 240


def test_incomplete_run_and_calls_still_produce_diagnostics():
    events = _plan() + [
        _event("llm.call.started", attributes={"operation": "unfinished"}),
        _event("tool.call.started", attributes={"tool_name": "unfinished"}),
        _event("sql.compilation.started", task_id="T1", source="sql"),
    ]
    summary = TraceSummaryBuilder().build(events)

    assert summary.trace_complete is False
    assert summary.lifecycle_status == "incomplete"
    assert summary.completed_at is None
    assert summary.incomplete_llm_call_count == 1
    assert summary.incomplete_tool_call_count == 1
    assert summary.incomplete_sql_compilation_count == 1


def test_unknown_future_event_and_null_attributes_do_not_crash():
    events = _plan() + [_event("future.phase.event", attributes=None), _complete()]

    summary = TraceSummaryBuilder().build(events)

    assert summary.event_count == len(events)
    assert summary.logical_llm_call_count == 0


def test_multiple_runs_are_isolated():
    first = _plan(run_id="run-a") + _task_success(run_id="run-a") + [_complete(run_id="run-a")]
    second = _plan(run_id="run-b") + [_complete(run_id="run-b", outcome="source_failure")]

    summaries = TraceSummaryBuilder().build_all(first + second)

    assert [summary.run_id for summary in summaries] == ["run-a", "run-b"]
    assert summaries[0].completed_task_count == 1
    assert summaries[1].completed_task_count == 0
    assert TraceSummaryBuilder().build(first + second, run_id="run-b").event_count == len(second)


def test_historical_task_count_keeps_unstarted_required_task_incomplete():
    events = _plan(task_ids=(), requirements=2)
    events[2]["attributes"] = {
        "primary_source": "sql",
        "required_sources": ["sql"],
        "task_count": 2,
    }
    events += _task_success("T1") + [_complete(outcome="partial_success")]

    summary = TraceSummaryBuilder().build(events)

    assert summary.required_task_count == 2
    assert summary.completed_task_count == 1
    assert summary.incomplete_task_count == 1
    assert summary.requirement_coverage == 0.5


def test_summary_persistence_upserts_duplicate_run(tmp_path):
    builder = TraceSummaryBuilder()
    path = tmp_path / "run_summaries.jsonl"
    first = builder.build(_plan() + [_complete(outcome="first")])
    replacement = builder.build(_plan() + [_complete(outcome="replacement")])
    other = builder.build(_plan(run_id="run-2") + [_complete(run_id="run-2")])

    builder.persist(first, path)
    builder.persist(other, path)
    builder.persist(replacement, path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert sum(row["run_id"] == "run-1" for row in rows) == 1
    assert next(row for row in rows if row["run_id"] == "run-1")["business_outcome"] == "replacement"


def test_jsonl_sink_generates_summary_only_after_run_terminal(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    summary_path = tmp_path / "run_summaries.jsonl"
    sink = JsonlTraceSink(trace_path, summary_path=summary_path)
    context = TraceContext(run_id="automatic")
    sink.write(
        TraceEvent.create(context, event_type="run.started", stage="orchestration", status="started")
    )
    assert not summary_path.exists()
    sink.write(
        TraceEvent.create(
            context,
            event_type="run.completed",
            stage="orchestration",
            status="completed",
            duration_ms=1,
            attributes={"outcome": "success"},
        )
    )
    row = json.loads(summary_path.read_text(encoding="utf-8"))
    assert row["run_id"] == "automatic" and row["trace_complete"] is True


def test_trace_disabled_does_not_affect_business_flow_or_write_summary(tmp_path):
    sink = JsonlTraceSink(
        tmp_path / "trace.jsonl", summary_path=tmp_path / "run_summaries.jsonl"
    )
    value = None
    with TraceRecorder(sink, enabled=False).run():
        value = "business answer"

    assert value == "business answer"
    assert not (tmp_path / "trace.jsonl").exists()
    assert not (tmp_path / "run_summaries.jsonl").exists()


def test_build_requires_explicit_run_id_for_mixed_trace():
    with pytest.raises(ValueError, match="run_id is required"):
        TraceSummaryBuilder().build(
            [_event("run.started", run_id="a"), _event("run.started", run_id="b")]
        )


def test_legacy_summary_retry_distribution_is_not_invented():
    summary = TraceSummaryBuilder().build(
        _plan()
        + [
            _event(
                "llm.call.completed",
                attributes={"attempt_count": 2, "retry_count": 1},
            ),
            _complete(),
        ]
    )
    payload = summary.to_dict()
    payload["summary_schema_version"] = "1.0"
    payload.pop("retried_logical_llm_call_count")

    loaded = type(summary).from_dict(payload)

    assert loaded.llm_retry_count == 1
    assert loaded.retried_logical_llm_call_count is None
