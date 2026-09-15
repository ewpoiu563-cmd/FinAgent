"""Bounded SQL-to-Web recovery for one atomic requirement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .evidence import EvidenceBundle
from .sql_executor import DirectSQLExecutionResult
from .trace import record_trace_event, trace_execution_scope
from .web_fallback import ControlledWebFallbackExecutor, WebFallbackExecutionResult

if TYPE_CHECKING:
    from .task_planning import TaskRequirement


@dataclass(frozen=True)
class SQLWebFallbackExecutionResult:
    """Terminal result after a permitted SQL-to-Web recovery attempt.

    ``sql_result`` remains available even when the Web recovery succeeds or
    fails, so callers retain the primary-source insufficiency provenance.
    """

    status: str
    answer: str | None
    evidence: EvidenceBundle | None
    web_evidence: EvidenceBundle | None
    missing_information: tuple[str, ...]
    error_type: str | None
    latency_ms: int
    sql_result: DirectSQLExecutionResult
    web_result: WebFallbackExecutionResult | None
    fallback_used: bool = True

    def public_answer(self) -> str:
        if self.status == "success" and self.answer:
            return self.answer
        if self.web_result is not None:
            return self.web_result.public_answer()
        return "insufficient: SQL 查询不足，Web 补充未返回可用结果。"


class SQLWebFallbackPolicy:
    """Recover only an allowed, insufficient SQL requirement through Web.

    The first version intentionally treats only ``insufficient`` as
    recoverable.  SQL source failures do not yet have a sufficiently precise
    taxonomy to distinguish transient coverage failures from validator or
    safety failures, so they must not be promoted to Web automatically.
    """

    def __init__(self, web_executor: ControlledWebFallbackExecutor) -> None:
        self.web_executor = web_executor

    @staticmethod
    def should_recover(task: "TaskRequirement", result: DirectSQLExecutionResult) -> bool:
        # Source planning owns the recovery boundary.  In particular, an
        # explicit local-database / SQL-only constraint sets this flag to false;
        # an otherwise recoverable zero-row result must not override a true flag.
        return task.web_fallback_allowed and result.status == "insufficient"

    async def recover(
        self,
        question: str,
        task: "TaskRequirement",
        sql_result: DirectSQLExecutionResult,
    ) -> DirectSQLExecutionResult | SQLWebFallbackExecutionResult:
        if not self.should_recover(task, sql_result):
            return sql_result

        record_trace_event(
            event_type="fallback.dispatched",
            stage="fallback",
            status="dispatched",
            task_id=task.id,
            source="sql",
            attributes={
                "from_stage": "source_execution",
                "from_dispatch": "sql_primary",
                "from_source": "sql",
                "to_source": "web",
                "target": "web",
                "reason": "sql_insufficient",
                "original_status": sql_result.status,
            },
        )
        started = time.perf_counter()
        record_trace_event(
            event_type="source.execution.started",
            stage="source_execution",
            status="started",
            task_id=task.id,
            source="web",
            attempt=1,
            attributes={"evidence_count": 0},
        )
        try:
            with trace_execution_scope(task_id=task.id, source="web"):
                web_result = await self.web_executor.execute(
                    question,
                    prior_evidence=sql_result.evidence,
                    missing_information=sql_result.missing_information,
                    execution_path="sql_web_fallback",
                    trace_final=False,
                    web_source_policy=task.web_source_policy,
                )
        except Exception as error:
            duration_ms = round((time.perf_counter() - started) * 1000)
            record_trace_event(
                event_type="source.execution.failed",
                stage="source_execution",
                status="source_failure",
                task_id=task.id,
                source="web",
                attempt=1,
                duration_ms=duration_ms,
                error_type=type(error).__name__,
                attributes={
                    "evidence_count": 0,
                    "source_result_status": "source_failure",
                },
            )
            return SQLWebFallbackExecutionResult(
                status="source_failure",
                answer=None,
                evidence=sql_result.evidence,
                web_evidence=None,
                missing_information=_merge_missing(
                    sql_result.missing_information,
                    ("Web fallback execution failed",),
                ),
                error_type=type(error).__name__,
                latency_ms=duration_ms,
                sql_result=sql_result,
                web_result=None,
            )

        duration_ms = round((time.perf_counter() - started) * 1000)
        successful = web_result.status == "success"
        web_evidence = web_result.web_evidence or web_result.evidence
        record_trace_event(
            event_type=(
                "source.execution.completed" if successful else "source.execution.failed"
            ),
            stage="source_execution",
            status=web_result.status,
            task_id=task.id,
            source="web",
            attempt=1,
            duration_ms=duration_ms,
            error_type=None if successful else web_result.error_type,
            attributes={
                "evidence_count": len(web_evidence.items) if web_evidence else 0,
                "source_result_status": web_result.status,
                "original_sql_status": sql_result.status,
            },
        )
        return SQLWebFallbackExecutionResult(
            status=web_result.status,
            answer=web_result.answer,
            evidence=web_result.evidence,
            web_evidence=web_result.web_evidence,
            missing_information=(
                ()
                if successful
                else _merge_missing(sql_result.missing_information, web_result.missing_information)
            ),
            error_type=web_result.error_type,
            latency_ms=duration_ms,
            sql_result=sql_result,
            web_result=web_result,
        )


def _merge_missing(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group if item))
