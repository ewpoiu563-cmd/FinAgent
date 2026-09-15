"""Serial orchestration for explicit two-source FinAgent plans."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from .evidence import EvidenceBundle
from .executors import DirectRAGExecutor
from .models import PlanMode, SourcePlan, SourceType
from .multi_source_synthesis import MultiSourceSynthesisResult, MultiSourceSynthesizer
from .sql_executor import DirectSQLExecutor
from .state import AgentState, SourceStatus
from .task_planning import TaskPlan, TaskStatus
from .trace import record_trace_event, trace_execution_scope
from .web_fallback import DirectWebExecutor
from .web_planner import WebTaskComplexity, classify_web_task
from .web_research import StructuredWebResearchExecutor


logger = logging.getLogger(__name__)
_TRACE_PREVIEW_CHARS = 300
_SUPPORTED_PAIRS = frozenset(
    {
        frozenset({"rag", "web"}),
        frozenset({"sql", "rag"}),
        frozenset({"sql", "web"}),
    }
)


@dataclass(frozen=True)
class MultiSourceExecutionResult:
    status: str
    answer: str | None
    state: AgentState
    synthesis: MultiSourceSynthesisResult | None
    latency_ms: int
    error_type: str | None = None
    task_plan: TaskPlan | None = None

    def public_answer(self) -> str:
        if self.status == "success" and self.answer:
            return self.answer
        if self.status in {"partial_success", "source_failure", "insufficient"}:
            return self._user_facing_partial_answer()
        if self.status == "provider_content_block":
            return "已取得相关资料，但暂时无法生成最终综合答案。"
        if self.status == "generation_failure":
            return "多来源结果暂时无法整理为可靠答案，未补写未经证据支持的内容。"
        return self._user_facing_partial_answer()

    def _user_facing_partial_answer(self) -> str:
        labels = {"sql": "数据库", "rag": "本地文档", "web": "Web"}
        parts = [
            f"{labels.get(source, source)}结果：{answer}"
            for source, answer in sorted(self.state.answers_by_source.items())
            if answer
        ]
        failures = [
            f"{labels.get(source, source)}未能提供足够信息"
            + (f"（{error}）" if error else "")
            for source, error in sorted(self.state.source_errors.items())
        ]
        if failures:
            parts.append("；".join(failures) + "。")
        return "\n\n".join(parts) or "当前可用数据源未能提供足够信息来回答该问题。"

    def partial_result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "required_sources": sorted(self.state.required_sources),
            "completed_sources": sorted(self.state.completed_sources),
            "failed_sources": sorted(self.state.failed_sources),
            "source_statuses": {
                source: status.value
                for source, status in sorted(self.state.source_statuses.items())
            },
            "source_answers": dict(sorted(self.state.answers_by_source.items())),
            "source_errors": dict(sorted(self.state.source_errors.items())),
            "evidence_count_by_source": self.state.evidence_count_by_source(),
            "evidence_by_source": {
                source: self.state.evidence_by_source[source].to_dict()
                for source in sorted(self.state.completed_sources)
                if source in self.state.evidence_by_source
            },
            "message": "仅返回已完成来源的结果；未把失败来源视为已完成，也未执行完整多源综合。",
        }
        if self.task_plan is not None:
            result["tasks"] = [task.to_dict() for task in self.task_plan.tasks]
        return result


class MultiSourceOrchestrator:
    """Execute an immutable, explicit source set without legacy ReAct routing."""

    def __init__(
        self,
        rag_executor: DirectRAGExecutor,
        sql_executor: DirectSQLExecutor,
        web_executor: DirectWebExecutor,
        synthesizer: MultiSourceSynthesizer,
        *,
        structured_web_executor: StructuredWebResearchExecutor | None = None,
    ) -> None:
        self.rag_executor = rag_executor
        self.sql_executor = sql_executor
        self.web_executor = web_executor
        self.structured_web_executor = structured_web_executor
        self.synthesizer = synthesizer

    async def execute(
        self,
        question: str,
        plan: SourcePlan,
        *,
        task_plan: TaskPlan | None = None,
    ) -> MultiSourceExecutionResult:
        if not is_supported_multi_source_plan(plan):
            raise ValueError("Phase D requires one supported explicit two-source plan")
        if task_plan is not None:
            _validate_hybrid_task_plan(task_plan, plan)
        started = time.perf_counter()
        state = AgentState({source.value for source in plan.required_sources})
        logger.debug(
            "REQUIRED_SOURCES: %s",
            json.dumps(
                {
                    "mode": plan.mode.value,
                    "required_sources": sorted(state.required_sources),
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                },
                ensure_ascii=False,
            ),
        )
        for source in plan.required_sources:
            logger.debug(
                "HYBRID_SOURCE_REQUIRED: %s",
                json.dumps(
                    {
                        "source": source.value,
                        "required_sources": sorted(state.required_sources),
                    },
                    ensure_ascii=False,
                ),
            )

        for source in plan.required_sources:
            source_name = source.value
            task = task_plan.task_for_source(source_name) if task_plan else None
            if task is not None and task.successful:
                continue
            if not state.start_source(source_name):
                continue
            task_started = time.perf_counter()
            if task is not None:
                task.start()
                _record_task_started(task)
            source_started = time.perf_counter()
            _record_source_started(
                source_name,
                task.id if task is not None else None,
                state.source_attempts[source_name],
            )
            logger.debug(
                "SOURCE_STARTED: %s",
                json.dumps(
                    {
                        "source": source_name,
                        "attempt": state.source_attempts[source_name],
                    },
                    ensure_ascii=False,
                ),
            )
            try:
                source_question = task.question if task is not None else _source_question(question, plan, source)
                with trace_execution_scope(
                    task_id=task.id if task is not None else None,
                    source=source_name,
                ):
                    result = await self._execute_source(source_question, plan, source)
                evidence = _result_evidence(source, result)
                result_status = str(getattr(result, "status", "source_failure"))
                answer = getattr(result, "answer", None)
                if (
                    result_status == "success"
                    and isinstance(answer, str)
                    and bool(answer.strip())
                    and evidence is not None
                    and evidence.execution_success
                    and evidence.source_match
                    and evidence.items
                ):
                    state.complete_source(source_name, evidence, answer=answer)
                    if task is not None:
                        task.complete(
                            answer=answer,
                            evidence_ids=[
                                f"{source_name}:{item.retrieval_unit_id}"
                                for item in evidence.items
                                if item.retrieval_unit_id
                            ],
                        )
                    logger.debug(
                        "SOURCE_COMPLETED: %s",
                        json.dumps(
                            {
                                "source": source_name,
                                "status": "success",
                                "attempt": state.source_attempts[source_name],
                                "evidence_count": len(evidence.items),
                            },
                            ensure_ascii=False,
                        ),
                    )
                    _record_source_result(
                        source_name,
                        task.id if task is not None else None,
                        state.source_attempts[source_name],
                        source_started,
                        result_status,
                        len(evidence.items),
                        None,
                        source_result_status=result_status,
                    )
                    if task is not None:
                        _record_task_result(
                            task,
                            task_started,
                            evidence_count=len(evidence.items),
                        )
                else:
                    terminal = _source_failure_status(result_status)
                    error = str(getattr(result, "error_type", None) or result_status)
                    state.fail_source(
                        source_name,
                        terminal,
                        evidence=evidence,
                        error=error,
                    )
                    if task is not None:
                        task.fail(terminal.value, missing_information=_result_missing(result))
                    logger.debug(
                        "SOURCE_FAILED: %s",
                        json.dumps(
                            {
                                "source": source_name,
                                "status": terminal.value,
                                "source_result_status": result_status,
                                "attempt": state.source_attempts[source_name],
                                "evidence_count": len(evidence.items) if evidence else 0,
                                "error_type": error,
                            },
                            ensure_ascii=False,
                        ),
                    )
                    _record_source_result(
                        source_name,
                        task.id if task is not None else None,
                        state.source_attempts[source_name],
                        source_started,
                        terminal.value,
                        len(evidence.items) if evidence else 0,
                        error,
                        source_result_status=result_status,
                    )
                    if task is not None:
                        _record_task_result(
                            task,
                            task_started,
                            evidence_count=len(evidence.items) if evidence else 0,
                            error_type=error,
                        )
            except Exception as error:
                state.fail_source(
                    source_name,
                    SourceStatus.SOURCE_FAILURE,
                    error=type(error).__name__,
                )
                if task is not None:
                    task.fail(TaskStatus.SOURCE_FAILURE.value, missing_information=[type(error).__name__])
                logger.debug(
                    "SOURCE_FAILED: %s",
                    json.dumps(
                        {
                            "source": source_name,
                            "status": SourceStatus.SOURCE_FAILURE.value,
                            "attempt": state.source_attempts[source_name],
                            "error_type": type(error).__name__,
                        },
                        ensure_ascii=False,
                    ),
                )
                _record_source_result(
                    source_name,
                    task.id if task is not None else None,
                    state.source_attempts[source_name],
                    source_started,
                    SourceStatus.SOURCE_FAILURE.value,
                    0,
                    type(error).__name__,
                    source_result_status=SourceStatus.SOURCE_FAILURE.value,
                )
                if task is not None:
                    _record_task_result(
                        task,
                        task_started,
                        evidence_count=0,
                        error_type=type(error).__name__,
                    )

        source_tasks_complete = not task_plan or all(
            task.successful for task in task_plan.execution_tasks
        )
        if not state.ready_for_synthesis or not source_tasks_complete:
            if task_plan is not None and task_plan.synthesis_tasks:
                _record_synthesis_failure(
                    task_plan.synthesis_tasks[0],
                    task_plan,
                    state,
                    time.perf_counter(),
                    "dependency_failure",
                    None,
                )
            status = "partial_success" if state.completed_sources else _aggregate_failure_status(state)
            return self._finish(started, state, status=status, task_plan=task_plan)

        logger.debug(
            "EVIDENCE_MERGE: %s",
            json.dumps(
                {
                    "required_sources": sorted(state.required_sources),
                    "evidence_count_by_source": state.evidence_count_by_source(),
                    "source_boundaries_preserved": True,
                },
                ensure_ascii=False,
            ),
        )
        synthesis_task = task_plan.synthesis_tasks[0] if task_plan else None
        if synthesis_task is not None:
            if not task_plan.dependencies_satisfied(synthesis_task):
                return self._finish(
                    started,
                    state,
                    status="partial_success",
                    task_plan=task_plan,
                )
            synthesis_task.start()
        synthesis_started = time.perf_counter()
        _record_synthesis_started(synthesis_task, task_plan, state)
        try:
            synthesis = self.synthesizer.synthesize(
                question,
                state.evidence_by_source,
                state.required_sources,
            )
        except Exception as error:
            if synthesis_task is not None:
                synthesis_task.fail(
                    TaskStatus.GENERATION_FAILURE.value,
                    missing_information=[type(error).__name__],
                )
            _record_synthesis_failure(
                synthesis_task,
                task_plan,
                state,
                synthesis_started,
                "generation_failure",
                type(error).__name__,
            )
            return self._finish(
                started,
                state,
                status="generation_failure",
                error_type=type(error).__name__,
                task_plan=task_plan,
            )
        if not synthesis.successful:
            if synthesis_task is not None:
                synthesis_task.fail(
                    synthesis.status,
                    missing_information=[synthesis.error_type or synthesis.status],
                )
            _record_synthesis_failure(
                synthesis_task,
                task_plan,
                state,
                synthesis_started,
                synthesis.status,
                synthesis.error_type,
            )
            return self._finish(
                started,
                state,
                status=synthesis.status,
                synthesis=synthesis,
                error_type=synthesis.error_type,
                task_plan=task_plan,
            )
        if not synthesis.answer:
            if synthesis_task is not None:
                synthesis_task.fail(
                    TaskStatus.GENERATION_FAILURE.value,
                    missing_information=["synthesis returned no answer"],
                )
            _record_synthesis_failure(
                synthesis_task,
                task_plan,
                state,
                synthesis_started,
                "generation_failure",
                "MissingSynthesisAnswer",
            )
            return self._finish(
                started,
                state,
                status="generation_failure",
                synthesis=synthesis,
                error_type="MissingSynthesisAnswer",
                task_plan=task_plan,
            )
        if synthesis_task is not None:
            synthesis_task.complete(
                answer=synthesis.answer,
                evidence_ids=[
                    identifier
                    for claim in synthesis.claims
                    for identifier in claim.supported_evidence_ids
                ],
            )
        _record_synthesis_completed(
            synthesis_task,
            task_plan,
            state,
            synthesis_started,
            synthesis,
        )
        return self._finish(
            started,
            state,
            status="success",
            answer=synthesis.answer,
            synthesis=synthesis,
            task_plan=task_plan,
        )

    async def _execute_source(
        self,
        question: str,
        plan: SourcePlan,
        source: SourceType,
    ) -> Any:
        source_plan = _single_source_projection(plan, source)
        if source is SourceType.RAG:
            return self.rag_executor.execute(
                question,
                source_plan,
                trace_final=False,
            )
        if source is SourceType.SQL:
            return self.sql_executor.execute(
                question,
                source_plan,
                trace_final=False,
            )
        if source is SourceType.WEB:
            if (
                self.structured_web_executor is not None
                and classify_web_task(question) is WebTaskComplexity.STRUCTURED_RESEARCH
            ):
                return await self.structured_web_executor.execute(
                    question,
                    source_plan,
                    trace_final=False,
                )
            return await self.web_executor.execute_for_source(
                question,
                source_plan,
                trace_final=False,
            )
        raise ValueError(f"unsupported hybrid source: {source.value}")

    @staticmethod
    def _finish(
        started: float,
        state: AgentState,
        *,
        status: str,
        answer: str | None = None,
        synthesis: MultiSourceSynthesisResult | None = None,
        error_type: str | None = None,
        task_plan: TaskPlan | None = None,
    ) -> MultiSourceExecutionResult:
        if status == "success" and (
            not state.ready_for_synthesis
            or (task_plan is not None and not task_plan.all_required_successful)
        ):
            status = "partial_success" if state.completed_sources else _aggregate_failure_status(state)
            answer = None
        if status == "success" and not answer:
            status = "generation_failure"
        result = MultiSourceExecutionResult(
            status=status,
            answer=answer,
            state=state,
            synthesis=synthesis,
            latency_ms=round((time.perf_counter() - started) * 1000),
            error_type=error_type,
            task_plan=task_plan,
        )
        logger.debug(
            "FINAL: %s",
            json.dumps(
                {
                    "status": status,
                    "required_sources": sorted(state.required_sources),
                    "completed_sources": sorted(state.completed_sources),
                    "failed_sources": sorted(state.failed_sources),
                    "evidence_count_by_source": state.evidence_count_by_source(),
                    "claim_count": len(synthesis.claims) if synthesis else 0,
                    "answer_preview": (answer or "")[:_TRACE_PREVIEW_CHARS],
                    "error_type": error_type,
                    "latency_ms": result.latency_ms,
                },
                ensure_ascii=False,
            ),
        )
        return result


def is_supported_multi_source_plan(plan: SourcePlan) -> bool:
    required = frozenset(source.value for source in plan.required_sources)
    return (
        plan.mode is PlanMode.MULTI_SOURCE
        and len(plan.required_sources) == 2
        and required in _SUPPORTED_PAIRS
    )


def _single_source_projection(plan: SourcePlan, source: SourceType) -> SourcePlan:
    return SourcePlan(
        mode=PlanMode.SINGLE_SOURCE,
        primary_source=source,
        required_sources=(source,),
        company_name=plan.company_name,
        catalog_doc_ids=plan.catalog_doc_ids,
        document_type=plan.document_type,
        web_fallback_allowed=False,
        routing_method=f"{plan.routing_method}:hybrid_projection",
        reason=f"execute the {source.value} portion of an explicit multi-source plan",
    )


def _source_question(question: str, plan: SourcePlan, source: SourceType) -> str:
    company = plan.company_name or "目标实体"
    if source is SourceType.RAG:
        document = plan.document_type or "本地金融文档"
        instruction = f"仅提取{company}{document}中与原问题有关的文档事实，不判断新闻或数据库部分。"
    elif source is SourceType.SQL:
        instruction = "仅回答原问题中可由本地结构化基金数据库支持的历史指标、行情或持仓部分，不解释文档或新闻。"
    else:
        instruction = "仅检索原问题中需要由当前或近期公开 Web 信息支持的事实，不把本地文档或数据库结果当作 Web 事实。"
    return f"{question.strip()}\n\n多源子任务边界：{instruction}"


def _result_evidence(source: SourceType, result: Any) -> EvidenceBundle | None:
    if source is SourceType.WEB:
        return getattr(result, "web_evidence", None) or getattr(result, "evidence", None)
    return getattr(result, "evidence", None)


def _source_failure_status(status: str) -> SourceStatus:
    if status == "insufficient" or status == "clarification_needed":
        return SourceStatus.INSUFFICIENT
    if status == "provider_content_block":
        return SourceStatus.PROVIDER_CONTENT_BLOCK
    if status == "generation_failure":
        return SourceStatus.GENERATION_FAILURE
    return SourceStatus.SOURCE_FAILURE


def _aggregate_failure_status(state: AgentState) -> str:
    statuses = set(state.source_statuses.values())
    if SourceStatus.PROVIDER_CONTENT_BLOCK in statuses:
        return "provider_content_block"
    if SourceStatus.GENERATION_FAILURE in statuses:
        return "generation_failure"
    return "source_failure"


def _validate_hybrid_task_plan(task_plan: TaskPlan, plan: SourcePlan) -> None:
    task_sources = [task.source for task in task_plan.execution_tasks]
    required = {source.value for source in plan.required_sources}
    if len(task_sources) != len(required) or set(task_sources) != required:
        raise ValueError("hybrid TaskPlan must contain exactly one task per required source")
    if len(task_plan.synthesis_tasks) != 1:
        raise ValueError("hybrid TaskPlan must contain exactly one synthesis task")
    synthesis = task_plan.synthesis_tasks[0]
    if set(synthesis.depends_on) != {task.id for task in task_plan.execution_tasks}:
        raise ValueError("hybrid synthesis task must depend on every source task")


def _result_missing(result: Any) -> list[str]:
    missing = getattr(result, "missing_information", ())
    return [str(item) for item in missing if str(item)]


def _record_task_started(task) -> None:
    record_trace_event(
        event_type="task.execution.started",
        stage="task_execution",
        status="started",
        task_id=task.id,
        source=task.source,
        attributes={
            "evidence_count": 0,
            "missing_information_count": 0,
        },
    )


def _record_task_result(
    task,
    started: float,
    *,
    evidence_count: int,
    error_type: str | None = None,
) -> None:
    successful = task.status == TaskStatus.SUCCESS.value
    record_trace_event(
        event_type=(
            "task.execution.completed" if successful else "task.execution.failed"
        ),
        stage="task_execution",
        status=task.status,
        task_id=task.id,
        source=task.source,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=None if successful else error_type,
        attributes={
            "evidence_count": evidence_count,
            "missing_information_count": len(task.missing_information),
            "failure_status": None if successful else task.status,
        },
    )


def _record_source_started(source: str, task_id: str | None, attempt: int) -> None:
    record_trace_event(
        event_type="source.execution.started",
        stage="source_execution",
        status="started",
        task_id=task_id,
        source=source,
        attempt=attempt,
        attributes={"evidence_count": 0},
    )


def _record_source_result(
    source: str,
    task_id: str | None,
    attempt: int,
    started: float,
    result_status: str,
    evidence_count: int,
    error_type: str | None,
    *,
    source_result_status: str,
) -> None:
    successful = result_status == "success"
    record_trace_event(
        event_type=(
            "source.execution.completed" if successful else "source.execution.failed"
        ),
        stage="source_execution",
        status=result_status,
        task_id=task_id,
        source=source,
        attempt=attempt,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=None if successful else error_type,
        attributes={
            "evidence_count": evidence_count,
            "source_result_status": source_result_status,
        },
    )


def _synthesis_counts(task, task_plan: TaskPlan | None, state: AgentState) -> dict[str, int]:
    dependency_count = len(task.depends_on) if task is not None else len(state.required_sources)
    completed_dependency_count = (
        sum(
            1
            for dependency in task.depends_on
            if task_plan is not None
            and next(
                (candidate.successful for candidate in task_plan.tasks if candidate.id == dependency),
                False,
            )
        )
        if task is not None
        else len(state.completed_sources)
    )
    return {
        "dependency_count": dependency_count,
        "completed_dependency_count": completed_dependency_count,
        "evidence_count": sum(len(bundle.items) for bundle in state.evidence_by_source.values()),
    }


def _record_synthesis_started(task, task_plan: TaskPlan | None, state: AgentState) -> None:
    record_trace_event(
        event_type="synthesis.started",
        stage="synthesis",
        status="started",
        task_id=task.id if task is not None else None,
        attributes=_synthesis_counts(task, task_plan, state),
    )


def _record_synthesis_completed(
    task,
    task_plan: TaskPlan | None,
    state: AgentState,
    started: float,
    synthesis: MultiSourceSynthesisResult,
) -> None:
    attributes = _synthesis_counts(task, task_plan, state)
    attributes["evidence_id_count"] = sum(
        len(claim.supported_evidence_ids) for claim in synthesis.claims
    )
    record_trace_event(
        event_type="synthesis.completed",
        stage="synthesis",
        status="success",
        task_id=task.id if task is not None else None,
        duration_ms=round((time.perf_counter() - started) * 1000),
        attributes=attributes,
    )


def _record_synthesis_failure(
    task,
    task_plan: TaskPlan | None,
    state: AgentState,
    started: float,
    failure_status: str,
    error_type: str | None,
) -> None:
    attributes = _synthesis_counts(task, task_plan, state)
    attributes["failure_status"] = failure_status
    record_trace_event(
        event_type="synthesis.failed",
        stage="synthesis",
        status=failure_status,
        task_id=task.id if task is not None else None,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=error_type,
        attributes=attributes,
    )
