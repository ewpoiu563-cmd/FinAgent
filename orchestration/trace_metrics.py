"""Descriptive cross-run telemetry aggregated from :class:`RunSummary`."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .trace_summary import RunSummary


AGGREGATE_SCHEMA_VERSION = "1.0"
_SESSION_FILTER_UNSET = object()
_FAILED_BUSINESS_OUTCOMES = {
    "clarification_needed",
    "source_failure",
    "generation_failure",
    "provider_content_block",
    "insufficient",
    "source_scope_violation",
    "unhandled_error",
}


@dataclass(frozen=True, slots=True)
class DistributionStats:
    """A distribution with an explicit non-null sample size."""

    sample_size: int
    mean: float | None
    p50: float | None
    p95: float | None
    max: int | float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "max": self.max,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistributionStats":
        return cls(
            sample_size=value.get("sample_size", 0),
            mean=value.get("mean"),
            p50=value.get("p50"),
            p95=value.get("p95"),
            max=value.get("max"),
        )


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Versioned descriptive metrics for a filtered set of run summaries."""

    aggregate_schema_version: str
    generated_at: str | None
    filters: Mapping[str, Any]
    sample_size: int
    run_count: int
    complete_run_count: int
    incomplete_run_count: int
    business_outcome_count: Mapping[str, int]
    outcome_sample_size: int
    success_run_count: int
    partial_success_run_count: int
    failed_run_count: int
    success_rate: float | None
    partial_success_rate: float | None
    total_required_tasks: int
    total_completed_tasks: int
    total_incomplete_tasks: int
    overall_requirement_coverage: float | None
    mean_run_requirement_coverage: float | None
    requirement_coverage_sample_size: int
    run_duration_ms: DistributionStats
    llm_latency_ms_sum: DistributionStats
    tool_latency_ms_sum: DistributionStats
    logical_llm_call_count: int
    avg_llm_calls_per_run: float | None
    successful_llm_call_count: int
    failed_llm_call_count: int
    llm_attempt_count: int
    llm_retry_count: int
    total_retry_count: int
    retried_logical_call_count: int | None
    llm_retry_rate: float | None
    llm_calls_by_operation: Mapping[str, int]
    provider_reported_input_tokens: int | None
    provider_reported_output_tokens: int | None
    provider_reported_total_tokens: int | None
    token_usage_known_call_count: int
    token_usage_unknown_call_count: int
    token_usage_coverage: float | None
    tool_call_count: int
    avg_tool_calls_per_run: float | None
    successful_tool_call_count: int
    failed_tool_call_count: int
    tool_calls_by_name: Mapping[str, int]
    runs_using_source: Mapping[str, int]
    source_attempt_count_by_source: Mapping[str, int]
    source_success_count_by_source: Mapping[str, int]
    source_failure_count_by_source: Mapping[str, int]
    source_terminal_status_distribution: Mapping[str, Mapping[str, int]]
    planned_primary_source_run_count: Mapping[str, int]
    fallback_count: int
    runs_with_fallback: int
    fallback_run_rate: float | None
    fallback_transitions: Mapping[str, int]
    sql_compilation_count: int
    sql_compilation_failure_count: int
    avg_sql_compilations_per_run: float | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            result[item.name] = value.to_dict() if isinstance(value, DistributionStats) else value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AggregateMetrics":
        """Round-trip known fields while ignoring future schema additions."""

        if not isinstance(value, Mapping):
            raise TypeError("AggregateMetrics payload must be a mapping")
        known_fields = {item.name for item in fields(cls)}
        payload = {key: item for key, item in value.items() if key in known_fields}
        for name in ("run_duration_ms", "llm_latency_ms_sum", "tool_latency_ms_sum"):
            item = payload.get(name)
            if isinstance(item, Mapping):
                payload[name] = DistributionStats.from_dict(item)
        return cls(**payload)


class RunMetricsAggregator:
    """Side-effect-free aggregation over an explicitly supplied run set."""

    def aggregate(
        self,
        summaries: Iterable[RunSummary],
        *,
        run_ids: Iterable[str] | None = None,
        session_id: str | None | object = _SESSION_FILTER_UNSET,
        generated_at: str | None = None,
    ) -> AggregateMetrics:
        requested_run_ids = tuple(dict.fromkeys(run_ids)) if run_ids is not None else None
        run_id_filter = set(requested_run_ids) if requested_run_ids is not None else None
        session_filter_applied = session_id is not _SESSION_FILTER_UNSET
        selected = [
            summary
            for summary in summaries
            if (run_id_filter is None or summary.run_id in run_id_filter)
            and (not session_filter_applied or summary.session_id == session_id)
        ]

        filters = {
            "run_ids": list(requested_run_ids) if requested_run_ids is not None else None,
            "session_id_filter_applied": session_filter_applied,
            "session_id": session_id if session_filter_applied else None,
        }
        run_count = len(selected)
        outcome_counts = Counter(
            summary.business_outcome
            for summary in selected
            if isinstance(summary.business_outcome, str) and summary.business_outcome
        )
        outcome_sample_size = sum(outcome_counts.values())
        success_count = outcome_counts.get("success", 0)
        partial_count = outcome_counts.get("partial_success", 0)
        failed_count = sum(
            outcome_counts.get(outcome, 0) for outcome in _FAILED_BUSINESS_OUTCOMES
        )

        total_required = sum(summary.required_task_count for summary in selected)
        total_completed = sum(summary.completed_task_count for summary in selected)
        total_incomplete = sum(summary.incomplete_task_count for summary in selected)
        run_coverages = [
            summary.requirement_coverage
            for summary in selected
            if _number(summary.requirement_coverage) is not None
        ]

        logical_llm_calls = sum(summary.logical_llm_call_count for summary in selected)
        retried_values = [summary.retried_logical_llm_call_count for summary in selected]
        retry_distribution_known = all(
            value is not None or summary.logical_llm_call_count == 0
            for value, summary in zip(retried_values, selected)
        )
        retried_logical_calls = (
            sum(value or 0 for value in retried_values) if retry_distribution_known else None
        )
        llm_retry_count = sum(summary.llm_retry_count for summary in selected)

        token_known = sum(summary.token_usage_known_call_count for summary in selected)
        token_unknown = sum(summary.token_usage_unknown_call_count for summary in selected)
        token_denominator = token_known + token_unknown

        tool_calls = sum(summary.tool_call_count for summary in selected)
        fallback_count = sum(summary.fallback_count for summary in selected)
        runs_with_fallback = sum(summary.fallback_count > 0 for summary in selected)
        compilation_count = sum(summary.sql_compilation_count for summary in selected)

        runs_using_source: Counter[str] = Counter()
        source_attempts: Counter[str] = Counter()
        source_successes: Counter[str] = Counter()
        source_failures: Counter[str] = Counter()
        source_statuses: dict[str, Counter[str]] = {}
        planned_sources: Counter[str] = Counter()
        llm_operations: Counter[str] = Counter()
        tool_names: Counter[str] = Counter()
        fallback_transitions: Counter[str] = Counter()
        for summary in selected:
            runs_using_source.update(set(summary.actual_sources))
            source_attempts.update(summary.source_attempt_count_by_source)
            source_successes.update(summary.source_success_count_by_source)
            source_failures.update(summary.source_failure_count_by_source)
            for source, statuses in summary.source_statuses.items():
                source_statuses.setdefault(source, Counter()).update(statuses)
            if summary.planned_primary_source:
                planned_sources[summary.planned_primary_source] += 1
            llm_operations.update(summary.llm_calls_by_operation)
            tool_names.update(summary.tool_calls_by_name)
            fallback_transitions.update(summary.fallback_transitions)

        return AggregateMetrics(
            aggregate_schema_version=AGGREGATE_SCHEMA_VERSION,
            generated_at=generated_at,
            filters=filters,
            sample_size=run_count,
            run_count=run_count,
            complete_run_count=sum(summary.trace_complete for summary in selected),
            incomplete_run_count=sum(not summary.trace_complete for summary in selected),
            business_outcome_count=dict(outcome_counts),
            outcome_sample_size=outcome_sample_size,
            success_run_count=success_count,
            partial_success_run_count=partial_count,
            failed_run_count=failed_count,
            success_rate=_rate(success_count, outcome_sample_size),
            partial_success_rate=_rate(partial_count, outcome_sample_size),
            total_required_tasks=total_required,
            total_completed_tasks=total_completed,
            total_incomplete_tasks=total_incomplete,
            overall_requirement_coverage=_rate(total_completed, total_required),
            mean_run_requirement_coverage=_mean(run_coverages),
            requirement_coverage_sample_size=len(run_coverages),
            run_duration_ms=_distribution(summary.duration_ms for summary in selected),
            llm_latency_ms_sum=_distribution(
                summary.llm_latency_ms_sum for summary in selected
            ),
            tool_latency_ms_sum=_distribution(
                summary.tool_latency_ms_sum for summary in selected
            ),
            logical_llm_call_count=logical_llm_calls,
            avg_llm_calls_per_run=_rate(logical_llm_calls, run_count),
            successful_llm_call_count=sum(
                summary.successful_llm_call_count for summary in selected
            ),
            failed_llm_call_count=sum(summary.failed_llm_call_count for summary in selected),
            llm_attempt_count=sum(summary.llm_attempt_count for summary in selected),
            llm_retry_count=llm_retry_count,
            total_retry_count=llm_retry_count,
            retried_logical_call_count=retried_logical_calls,
            llm_retry_rate=(
                _rate(retried_logical_calls, logical_llm_calls)
                if retried_logical_calls is not None
                else None
            ),
            llm_calls_by_operation=dict(llm_operations),
            provider_reported_input_tokens=_optional_sum(
                summary.provider_reported_input_tokens for summary in selected
            ),
            provider_reported_output_tokens=_optional_sum(
                summary.provider_reported_output_tokens for summary in selected
            ),
            provider_reported_total_tokens=_optional_sum(
                summary.provider_reported_total_tokens for summary in selected
            ),
            token_usage_known_call_count=token_known,
            token_usage_unknown_call_count=token_unknown,
            token_usage_coverage=_rate(token_known, token_denominator),
            tool_call_count=tool_calls,
            avg_tool_calls_per_run=_rate(tool_calls, run_count),
            successful_tool_call_count=sum(
                summary.successful_tool_call_count for summary in selected
            ),
            failed_tool_call_count=sum(summary.failed_tool_call_count for summary in selected),
            tool_calls_by_name=dict(tool_names),
            runs_using_source=dict(runs_using_source),
            source_attempt_count_by_source=dict(source_attempts),
            source_success_count_by_source=dict(source_successes),
            source_failure_count_by_source=dict(source_failures),
            source_terminal_status_distribution={
                source: dict(counts) for source, counts in source_statuses.items()
            },
            planned_primary_source_run_count=dict(planned_sources),
            fallback_count=fallback_count,
            runs_with_fallback=runs_with_fallback,
            fallback_run_rate=_rate(runs_with_fallback, run_count),
            fallback_transitions=dict(fallback_transitions),
            sql_compilation_count=compilation_count,
            sql_compilation_failure_count=sum(
                summary.sql_compilation_failure_count for summary in selected
            ),
            avg_sql_compilations_per_run=_rate(compilation_count, run_count),
        )

    @staticmethod
    def read_run_summaries_jsonl(path: str | Path) -> tuple[RunSummary, ...]:
        """Read multiple summary rows, ignoring malformed/non-object lines."""

        summaries: list[RunSummary] = []
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        summaries.append(RunSummary.from_dict(value))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return tuple(summaries)

    @staticmethod
    def persist(metrics: AggregateMetrics, path: str | Path) -> None:
        """Atomically write one JSON aggregate report without mutating metrics."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = metrics.to_dict()
        if payload["generated_at"] is None:
            payload["generated_at"] = utc_generated_at()
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
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def utc_generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _rate(numerator: int | float | None, denominator: int) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return numerator / denominator


def _mean(values: Iterable[int | float]) -> float | None:
    samples = list(values)
    return sum(samples) / len(samples) if samples else None


def _percentile(sorted_values: list[int | float], quantile: float) -> float | None:
    """Linear interpolation at index ``(n - 1) * quantile`` (Hyndman-Fan 7)."""

    if not sorted_values:
        return None
    index = (len(sorted_values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = index - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _distribution(values: Iterable[int | float | None]) -> DistributionStats:
    samples = sorted(
        value for raw in values if (value := _number(raw)) is not None
    )
    return DistributionStats(
        sample_size=len(samples),
        mean=_mean(samples),
        p50=_percentile(samples, 0.50),
        p95=_percentile(samples, 0.95),
        max=samples[-1] if samples else None,
    )


def _optional_sum(values: Iterable[int | None]) -> int | None:
    samples = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
    return sum(samples) if samples else None


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "AggregateMetrics",
    "DistributionStats",
    "RunMetricsAggregator",
    "utc_generated_at",
]
