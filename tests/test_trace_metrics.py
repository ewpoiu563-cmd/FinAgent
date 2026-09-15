"""Phase 4.5 cross-run descriptive telemetry tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from orchestration.trace_metrics import (
    AGGREGATE_SCHEMA_VERSION,
    AggregateMetrics,
    RunMetricsAggregator,
)
from orchestration.trace_summary import TraceSummaryBuilder


def _event(event_type, *, run_id="run-1", **overrides):
    value = {
        "schema_version": "1.0",
        "timestamp": "2026-09-10T00:00:00.000Z",
        "run_id": run_id,
        "session_id": None,
        "event_id": f"{run_id}-{event_type}",
        "event_type": event_type,
        "stage": event_type.split(".")[0],
        "status": "started" if event_type.endswith("started") else "completed",
        "task_id": None,
        "source": None,
        "attempt": None,
        "duration_ms": None,
        "error_type": None,
        "attributes": {},
    }
    value.update(overrides)
    return value


def _summary(run_id="run-1", **overrides):
    events = [
        _event("run.started", run_id=run_id),
        _event(
            "planning.task_decomposition.completed",
            run_id=run_id,
            attributes={"requirement_count": 1, "required_requirement_count": 1},
        ),
        _event(
            "planning.source_plan.completed",
            run_id=run_id,
            attributes={
                "primary_source": "sql",
                "required_sources": ["sql"],
                "required_task_count": 1,
                "required_task_ids": ["T1"],
            },
        ),
        _event(
            "task.execution.started",
            run_id=run_id,
            task_id="T1",
            source="sql",
        ),
        _event(
            "source.execution.started",
            run_id=run_id,
            task_id="T1",
            source="sql",
        ),
        _event(
            "source.execution.completed",
            run_id=run_id,
            task_id="T1",
            source="sql",
            status="success",
            duration_ms=4,
        ),
        _event(
            "task.execution.completed",
            run_id=run_id,
            task_id="T1",
            source="sql",
            status="success",
        ),
        _event(
            "run.completed",
            run_id=run_id,
            status="completed",
            duration_ms=10,
            attributes={"outcome": "success"},
        ),
    ]
    summary = TraceSummaryBuilder().build(events)
    return replace(summary, **overrides)


def test_empty_summaries_have_null_rates_and_empty_distributions():
    metrics = RunMetricsAggregator().aggregate([])

    assert metrics.run_count == metrics.sample_size == 0
    assert metrics.success_rate is None
    assert metrics.overall_requirement_coverage is None
    assert metrics.fallback_run_rate is None
    assert metrics.token_usage_coverage is None
    assert metrics.run_duration_ms.sample_size == 0
    assert metrics.run_duration_ms.p95 is None


def test_single_successful_run():
    metrics = RunMetricsAggregator().aggregate([_summary()])

    assert metrics.aggregate_schema_version == AGGREGATE_SCHEMA_VERSION == "1.0"
    assert metrics.run_count == metrics.complete_run_count == 1
    assert metrics.success_run_count == 1
    assert metrics.success_rate == 1.0
    assert metrics.overall_requirement_coverage == 1.0


def test_outcome_distribution_success_partial_and_failure():
    summaries = [
        _summary("success"),
        _summary("partial", business_outcome="partial_success"),
        _summary("failed", business_outcome="source_failure"),
    ]
    metrics = RunMetricsAggregator().aggregate(summaries)

    assert metrics.business_outcome_count == {
        "success": 1,
        "partial_success": 1,
        "source_failure": 1,
    }
    assert metrics.success_run_count == 1
    assert metrics.partial_success_run_count == 1
    assert metrics.failed_run_count == 1
    assert metrics.success_rate == pytest.approx(1 / 3)


def test_overall_and_mean_run_requirement_coverage_are_distinct():
    summaries = [
        _summary("small", required_task_count=1, completed_task_count=1, incomplete_task_count=0, requirement_coverage=1.0),
        _summary("large", required_task_count=9, completed_task_count=0, incomplete_task_count=9, requirement_coverage=0.0),
    ]
    metrics = RunMetricsAggregator().aggregate(summaries)

    assert metrics.overall_requirement_coverage == 0.1
    assert metrics.mean_run_requirement_coverage == 0.5
    assert metrics.requirement_coverage_sample_size == 2


def test_planned_and_actual_source_usage_are_separate():
    summary = _summary(
        planned_primary_source="sql",
        actual_sources=("sql", "web"),
        source_attempt_count_by_source={"sql": 1, "web": 1},
        source_success_count_by_source={"sql": 0, "web": 1},
        source_failure_count_by_source={"sql": 1, "web": 0},
        source_statuses={"sql": ("insufficient",), "web": ("success",)},
    )
    metrics = RunMetricsAggregator().aggregate([summary])

    assert metrics.planned_primary_source_run_count == {"sql": 1}
    assert metrics.runs_using_source == {"sql": 1, "web": 1}


def test_fallback_count_can_exceed_runs_with_fallback():
    metrics = RunMetricsAggregator().aggregate(
        [
            _summary(
                "fallback",
                fallback_count=2,
                fallback_transitions={"rag->web": 1, "sql->web": 1},
            ),
            _summary("plain"),
        ]
    )

    assert metrics.fallback_count == 2
    assert metrics.runs_with_fallback == 1
    assert metrics.fallback_run_rate == 0.5


def test_planning_time_hybrid_does_not_create_fallback():
    summary = _summary(
        planned_required_sources=("sql", "web"),
        actual_sources=("sql", "web"),
        fallback_count=0,
        fallback_transitions={},
    )
    metrics = RunMetricsAggregator().aggregate([summary])

    assert metrics.runs_using_source == {"sql": 1, "web": 1}
    assert metrics.fallback_count == metrics.runs_with_fallback == 0


def test_llm_calls_attempts_and_retries_aggregate():
    summaries = [
        _summary(
            "a",
            logical_llm_call_count=2,
            successful_llm_call_count=2,
            llm_attempt_count=3,
            llm_retry_count=1,
            retried_logical_llm_call_count=1,
            llm_calls_by_operation={"route": 2},
        ),
        _summary(
            "b",
            logical_llm_call_count=1,
            successful_llm_call_count=0,
            failed_llm_call_count=1,
            llm_attempt_count=3,
            llm_retry_count=2,
            retried_logical_llm_call_count=1,
            llm_calls_by_operation={"synthesis": 1},
        ),
    ]
    metrics = RunMetricsAggregator().aggregate(summaries)

    assert metrics.logical_llm_call_count == 3
    assert metrics.avg_llm_calls_per_run == 1.5
    assert metrics.llm_attempt_count == 6
    assert metrics.llm_retry_count == metrics.total_retry_count == 3
    assert metrics.retried_logical_call_count == 2
    assert metrics.llm_retry_rate == pytest.approx(2 / 3)
    assert metrics.llm_calls_by_operation == {"route": 2, "synthesis": 1}


def test_token_known_unknown_mix_has_explicit_coverage():
    summaries = [
        _summary(
            "known",
            provider_reported_input_tokens=10,
            provider_reported_output_tokens=4,
            provider_reported_total_tokens=14,
            token_usage_known_call_count=1,
        ),
        _summary(
            "unknown",
            provider_reported_input_tokens=None,
            provider_reported_output_tokens=None,
            provider_reported_total_tokens=None,
            token_usage_unknown_call_count=1,
        ),
    ]
    metrics = RunMetricsAggregator().aggregate(summaries)

    assert metrics.provider_reported_total_tokens == 14
    assert metrics.token_usage_known_call_count == 1
    assert metrics.token_usage_unknown_call_count == 1
    assert metrics.token_usage_coverage == 0.5


def test_null_duration_is_excluded_instead_of_becoming_zero():
    metrics = RunMetricsAggregator().aggregate(
        [_summary("known", duration_ms=20), _summary("unknown", duration_ms=None)]
    )

    assert metrics.run_duration_ms.sample_size == 1
    assert metrics.run_duration_ms.mean == 20


def test_p50_p95_use_deterministic_linear_interpolation():
    metrics = RunMetricsAggregator().aggregate(
        [
            _summary("a", duration_ms=0),
            _summary("b", duration_ms=10),
            _summary("c", duration_ms=20),
            _summary("d", duration_ms=30),
        ]
    )

    assert metrics.run_duration_ms.p50 == 15.0
    assert metrics.run_duration_ms.p95 == pytest.approx(28.5)
    assert metrics.run_duration_ms.max == 30


def test_tool_calls_by_name_and_average():
    metrics = RunMetricsAggregator().aggregate(
        [
            _summary("a", tool_call_count=2, successful_tool_call_count=2, tool_calls_by_name={"search": 1, "fetch": 1}),
            _summary("b", tool_call_count=1, failed_tool_call_count=1, tool_calls_by_name={"search": 1}),
        ]
    )

    assert metrics.tool_call_count == 3
    assert metrics.avg_tool_calls_per_run == 1.5
    assert metrics.tool_calls_by_name == {"search": 2, "fetch": 1}


def test_source_terminal_status_distribution():
    summary = _summary(
        source_statuses={"sql": ("success", "insufficient"), "web": ("success",)},
    )
    metrics = RunMetricsAggregator().aggregate([summary])

    assert metrics.source_terminal_status_distribution == {
        "sql": {"success": 1, "insufficient": 1},
        "web": {"success": 1},
    }


def test_sql_compilation_does_not_pollute_llm_or_tool_counts():
    summary = _summary(
        sql_compilation_count=2,
        sql_compilation_failure_count=1,
        logical_llm_call_count=0,
        successful_llm_call_count=0,
        failed_llm_call_count=0,
        tool_call_count=0,
        successful_tool_call_count=0,
        failed_tool_call_count=0,
    )
    metrics = RunMetricsAggregator().aggregate([summary])

    assert metrics.sql_compilation_count == 2
    assert metrics.sql_compilation_failure_count == 1
    assert metrics.logical_llm_call_count == metrics.tool_call_count == 0


def test_incomplete_run_counts_but_null_outcome_and_duration_do_not_enter_denominators():
    summary = _summary(
        trace_complete=False,
        lifecycle_status="incomplete",
        business_outcome=None,
        duration_ms=None,
    )
    metrics = RunMetricsAggregator().aggregate([summary])

    assert metrics.run_count == metrics.incomplete_run_count == 1
    assert metrics.complete_run_count == metrics.outcome_sample_size == 0
    assert metrics.success_rate is None
    assert metrics.run_duration_ms.sample_size == 0


def test_run_ids_filter():
    metrics = RunMetricsAggregator().aggregate(
        [_summary("a"), _summary("b")], run_ids=["b"]
    )

    assert metrics.run_count == 1
    assert metrics.filters["run_ids"] == ["b"]


def test_session_id_filter_including_explicit_null():
    summaries = [
        _summary("a", session_id="session-a"),
        _summary("b", session_id="session-b"),
        _summary("none", session_id=None),
    ]

    selected = RunMetricsAggregator().aggregate(summaries, session_id="session-b")
    null_selected = RunMetricsAggregator().aggregate(summaries, session_id=None)

    assert selected.run_count == 1
    assert null_selected.run_count == 1
    assert null_selected.filters["session_id_filter_applied"] is True


def test_multiple_summary_jsonl_input(tmp_path):
    path = tmp_path / "run_summaries.jsonl"
    builder = TraceSummaryBuilder()
    builder.persist(_summary("a"), path)
    builder.persist(_summary("b"), path)

    summaries = RunMetricsAggregator().read_run_summaries_jsonl(path)
    metrics = RunMetricsAggregator().aggregate(summaries)

    assert len(summaries) == metrics.run_count == 2


def test_unknown_future_summary_field_is_forward_compatible():
    payload = _summary().to_dict()
    payload["future_metric"] = {"new": True}

    loaded = type(_summary()).from_dict(payload)
    metrics = RunMetricsAggregator().aggregate([loaded])

    assert loaded.run_id == "run-1"
    assert metrics.run_count == 1


def test_aggregate_serialization_round_trip_and_json_persistence(tmp_path):
    metrics = RunMetricsAggregator().aggregate(
        [_summary()], generated_at="2026-09-10T01:00:00.000Z"
    )
    restored = AggregateMetrics.from_dict(metrics.to_dict())
    path = tmp_path / "aggregate_metrics.json"
    RunMetricsAggregator.persist(restored, path)

    assert restored == metrics
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == metrics.to_dict()
