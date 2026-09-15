"""Thin adapter to Phase 4 RunSummary telemetry; no business recomputation."""

from __future__ import annotations

from typing import Iterable

from orchestration.trace_metrics import AggregateMetrics, RunMetricsAggregator
from orchestration.trace_summary import RunSummary


class RunSummaryTelemetryAdapter:
    def aggregate(self, summaries: Iterable[RunSummary]) -> AggregateMetrics:
        return RunMetricsAggregator().aggregate(summaries)

