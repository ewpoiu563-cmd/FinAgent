"""High-level dispatch between direct source executors and the legacy agent."""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path

from .context_builder import ContextBuildResult, ContextBuilder
from .context_compressor import (
    COMPRESSION_MAX_CHARS,
    SUMMARY_KEEP_RECENT_MESSAGES,
    SUMMARY_TRIGGER_MESSAGES,
    ContextCompressor,
)
from .executors import DirectRAGExecutor, is_direct_rag_plan
from .direct_execution import DirectNoToolExecutor
from .multi_source import MultiSourceOrchestrator, is_supported_multi_source_plan
from .models import SourcePlan, SourceType
from .rag_policy import RAGRetryFallbackPolicy
from .sql_fallback import SQLWebFallbackPolicy
from .sql_executor import DirectSQLExecutor, is_direct_sql_plan
from .source_planner import SourcePlanner
from .session_state import RedisSessionStore, SessionState, normalize_chat_history
from .task_execution import TaskPlanExecutor
from .task_decomposition import TaskDecompositionError
from .task_planning import TaskPlanner
from .tool_necessity import ToolNecessityError
from .trace import (
    JsonlTraceSink,
    TraceRecorder,
    notify_runtime_artifact,
    record_trace_event,
    trace_execution_scope,
)
from .web_fallback import DirectWebExecutor, is_direct_web_plan
from .web_research import (
    StructuredWebResearchExecutor,
    WebResearchPlanningError,
    is_structured_web_plan,
)


logger = logging.getLogger(__name__)


class OrchestrationEntrypoint:
    """Dispatch requirement plans, keeping model-only tasks outside legacy ReAct."""

    def __init__(
        self,
        planner: SourcePlanner,
        direct_rag_executor: DirectRAGExecutor,
        legacy_runner: Callable[[str], Awaitable[str]],
        *,
        rag_policy: RAGRetryFallbackPolicy | None = None,
        direct_sql_executor: DirectSQLExecutor | None = None,
        direct_web_executor: DirectWebExecutor | None = None,
        structured_web_executor: StructuredWebResearchExecutor | None = None,
        multi_source_orchestrator: MultiSourceOrchestrator | None = None,
        task_planner: TaskPlanner | None = None,
        task_executor: TaskPlanExecutor | None = None,
        direct_executor: DirectNoToolExecutor | None = None,
        sql_fallback_policy: SQLWebFallbackPolicy | None = None,
        trace_recorder: TraceRecorder | None = None,
        session_store: RedisSessionStore | None = None,
        context_builder: ContextBuilder | None = None,
        context_compressor: ContextCompressor | None = None,
    ) -> None:
        self.planner = planner
        self.direct_rag_executor = direct_rag_executor
        self.legacy_runner = legacy_runner
        self.rag_policy = rag_policy
        self.direct_sql_executor = direct_sql_executor
        self.direct_web_executor = direct_web_executor
        self.structured_web_executor = structured_web_executor
        self.multi_source_orchestrator = multi_source_orchestrator
        self.task_planner = task_planner or TaskPlanner(planner)
        self.direct_executor = direct_executor or DirectNoToolExecutor()
        self.sql_fallback_policy = sql_fallback_policy
        self.trace_recorder = trace_recorder or TraceRecorder(enabled=False)
        self.session_store = session_store
        self.context_builder = context_builder or ContextBuilder()
        self.context_compressor = context_compressor or ContextCompressor()
        self.task_executor = task_executor or TaskPlanExecutor(
            sql_executor=direct_sql_executor,
            direct_executor=self.direct_executor,
            rag_executor=direct_rag_executor,
            web_executor=direct_web_executor,
            structured_web_executor=structured_web_executor,
            sql_fallback_policy=sql_fallback_policy,
            rag_policy=rag_policy,
        )

    async def run(
        self,
        question: str,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        chat_history: list | None = None,
    ) -> str:
        history = normalize_chat_history(chat_history)
        resolved_plan: list[SourcePlan] = []
        with self.trace_recorder.run(
            session_id=session_id,
            run_id=run_id,
            attributes={
                "question_preview": question[:200],
                "question_length": len(question),
            },
        ) as trace_context:
            context_result = self._build_context(question, session_id, history)
            answer = await self._run(
                context_result.standalone_question,
                resolved_plan.append,
            )
            self.trace_recorder.set_run_outcome(_infer_business_outcome(answer), only_if_unset=True)
            if session_id is not None and self.session_store is not None:
                self._persist_session(
                    session_id=session_id,
                    run_id=trace_context.run_id,
                    question=question,
                    answer=answer,
                    plan=resolved_plan[-1] if resolved_plan else None,
                    initial_history=history,
                )
            return answer

    def _build_context(
        self,
        question: str,
        session_id: str | None,
        chat_history: list | None = None,
    ) -> ContextBuildResult:
        """Load and resolve session context without making a run depend on Redis/LLM."""
        result = ContextBuildResult.unchanged(question)
        if session_id is None and not chat_history:
            return result
        state = None
        if self.session_store is not None and session_id is not None:
            try:
                state = self.session_store.get(session_id)
            except Exception as error:
                logger.warning(
                    "Session context unavailable for session %s: %s",
                    session_id,
                    error,
                )
        # Existing server history is authoritative; avoid appending the full
        # client history again on every request.
        if chat_history and (state is None or not state.recent_messages and not state.conversation_summary):
            state = SessionState(session_id or "request-history", recent_messages=chat_history)
        if state is not None:
            result = self.context_builder.build(question, state)
        record_trace_event(
            event_type="context.build.completed",
            stage="context_build",
            status="completed",
            attributes={
                "depends_on_context": result.depends_on_context,
                "original_question_length": len(question),
                "standalone_question_length": len(result.standalone_question),
                "used_message_count": len(result.used_message_indices),
                "resolution_method": result.resolution_method,
            },
        )
        return result

    async def _run(
        self,
        question: str,
        plan_observer: Callable[[SourcePlan], None] | None = None,
    ) -> str:
        plan_started = time.perf_counter()
        try:
            if self.task_planner.uses_tool_necessity:
                task_plan = self.task_planner.plan(question)
                plan = task_plan.source_plan
            else:
                plan = self.planner.plan(question)
                task_plan = self.task_planner.plan(question, plan)
        except Exception as error:
            logger.debug(
                "TOOL_DECISION: %s",
                json.dumps(
                    {
                        "status": "planning_failure_legacy_dispatch",
                        "error_type": type(error).__name__,
                        "reason": str(error)[:300],
                        "latency_ms": round((time.perf_counter() - plan_started) * 1000),
                    },
                    ensure_ascii=False,
                ),
            )
            if isinstance(error, (TaskDecompositionError, ToolNecessityError)):
                _trace_fallback(
                    from_stage="requirement_planning",
                    from_dispatch="task_decomposition_or_tool_necessity",
                    target="bounded_source_planning",
                    reason="structured_planning_contract_failure",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                try:
                    # Recover through the rule-first SourcePlanner. Its
                    # ambiguous branch uses the bounded LLMSourceRouter, which
                    # may select only SQL/RAG/Web and cannot execute tools.
                    plan = self.planner.plan(question)
                    task_plan = self.task_planner.plan(question, plan)
                except Exception as recovery_error:
                    _trace_fallback(
                        from_stage="source_planning_recovery",
                        from_dispatch="bounded_source_planning",
                        target="legacy_react",
                        reason="source_planning_recovery_failure",
                        error_type=type(recovery_error).__name__,
                        error_message=str(recovery_error),
                    )
                    return await self._run_legacy(question)
            else:
                _trace_fallback(
                    from_stage="planning",
                    from_dispatch="source_planning",
                    target="legacy_react",
                    reason="planning_failure",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                return await self._run_legacy(question)
        plan_latency_ms = round((time.perf_counter() - plan_started) * 1000)
        if plan_observer is not None:
            plan_observer(plan)
        execution_question = (
            task_plan.execution_tasks[0].question
            if len(task_plan.execution_tasks) == 1
            else task_plan.resolved_question
        )
        task_owned = self.task_executor.can_execute(task_plan)
        dispatch = (
            "requirement_plan"
            if task_owned
            else "multi_source"
            if is_supported_multi_source_plan(plan)
            and self.multi_source_orchestrator is not None
            else "rag_first"
            if is_direct_rag_plan(plan)
            else "sql_first"
            if is_direct_sql_plan(plan) and self.direct_sql_executor is not None
            else "web_factual"
            if is_direct_web_plan(question, plan) and self.direct_web_executor is not None
            else "structured_web_multi_hop"
            if is_structured_web_plan(question, plan)
            and self.structured_web_executor is not None
            else "legacy"
        )
        record_trace_event(
            event_type="planning.source_plan.completed",
            stage="planning",
            status="completed",
            duration_ms=plan_latency_ms,
            attributes={
                "mode": plan.mode.value,
                "primary_source": (
                    plan.primary_source.value if plan.primary_source else None
                ),
                "required_sources": [
                    source.value for source in plan.required_sources
                ],
                "routing_method": plan.routing_method,
                "dispatch": dispatch,
                "task_count": len(task_plan.execution_tasks),
                "required_task_count": sum(
                    1 for task in task_plan.execution_tasks if task.required
                ),
                "required_task_ids": [
                    task.id for task in task_plan.execution_tasks if task.required
                ],
            },
        )
        logger.debug(
            "QUERY_PLAN: %s",
            json.dumps(
                {
                    "mode": plan.mode.value,
                    "primary_source": plan.primary_source.value if plan.primary_source else None,
                    "required_sources": [source.value for source in plan.required_sources],
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                    "web_fallback_allowed": plan.web_fallback_allowed,
                    "routing_method": plan.routing_method,
                    "dispatch": dispatch,
                    "latency_ms": plan_latency_ms,
                },
                ensure_ascii=False,
            ),
        )
        logger.debug(
            "REQUIRED_TASKS: %s",
            json.dumps(
                {
                    "required_task_ids": [task.id for task in task_plan.required_tasks],
                    "task_sources": [task.source for task in task_plan.required_tasks],
                    "reference_date": task_plan.reference_date.isoformat(),
                },
                ensure_ascii=False,
            ),
        )
        for task in task_plan.execution_tasks:
            if task.tool_decision is not None:
                logger.debug(
                    "TOOL_NECESSITY: %s",
                    json.dumps(task.tool_decision.to_dict(), ensure_ascii=False),
                )
        logger.debug(
            "ENTITY_RESOLUTION: %s",
            json.dumps(
                {
                    "company_name": plan.company_name,
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                    "document_type": plan.document_type,
                    "latency_ms": plan_latency_ms,
                },
                ensure_ascii=False,
            ),
        )

        if task_owned:
            result = await self.task_executor.execute(task_plan)
            return self._public_result(result)

        if is_supported_multi_source_plan(plan) and self.multi_source_orchestrator is not None:
            result = await self.multi_source_orchestrator.execute(
                task_plan.resolved_question,
                plan,
                task_plan=task_plan,
            )
            return self._public_result(result)

        if is_direct_rag_plan(plan):
            task = _single_execution_task(task_plan)
            task_started = _trace_task_started(task)
            if self.rag_policy is not None:
                try:
                    with trace_execution_scope(
                        task_id=task.id if task else None,
                        source=SourceType.RAG.value,
                    ):
                        result = await self.rag_policy.execute(execution_question, plan)
                except Exception as error:
                    _trace_task_exception(task, task_started, error)
                    raise
                _trace_task_result(task, task_started, result)
                return self._public_result(result)
            source_started = _trace_source_started(SourceType.RAG.value, task)
            try:
                with trace_execution_scope(
                    task_id=task.id if task else None,
                    source=SourceType.RAG.value,
                ):
                    result = self.direct_rag_executor.execute(execution_question, plan)
            except Exception as error:
                _trace_source_exception(SourceType.RAG.value, task, source_started, error)
                _trace_task_exception(task, task_started, error)
                raise
            _trace_source_result(SourceType.RAG.value, task, source_started, result)
            _trace_task_result(task, task_started, result)
            return self._public_result(result)
        if is_direct_sql_plan(plan) and self.direct_sql_executor is not None:
            task = _single_execution_task(task_plan)
            task_started = _trace_task_started(task)
            source_started = _trace_source_started(SourceType.SQL.value, task)
            try:
                with trace_execution_scope(
                    task_id=task.id if task else None,
                    source=SourceType.SQL.value,
                ):
                    result = self.direct_sql_executor.execute(execution_question, plan)
            except Exception as error:
                _trace_source_exception(SourceType.SQL.value, task, source_started, error)
                _trace_task_exception(task, task_started, error)
                raise
            _trace_source_result(SourceType.SQL.value, task, source_started, result)
            _trace_task_result(task, task_started, result)
            return self._public_result(result)
        if is_direct_web_plan(question, plan) and self.direct_web_executor is not None:
            task = _single_execution_task(task_plan)
            task_started = _trace_task_started(task)
            source_started = _trace_source_started(SourceType.WEB.value, task)
            try:
                with trace_execution_scope(
                    task_id=task.id if task else None,
                    source=SourceType.WEB.value,
                ):
                    result = await self.direct_web_executor.execute(execution_question, plan)
            except Exception as error:
                _trace_source_exception(SourceType.WEB.value, task, source_started, error)
                _trace_task_exception(task, task_started, error)
                raise
            _trace_source_result(SourceType.WEB.value, task, source_started, result)
            _trace_task_result(task, task_started, result)
            return self._public_result(result)
        if is_structured_web_plan(question, plan) and self.structured_web_executor is not None:
            task = _single_execution_task(task_plan)
            task_started = _trace_task_started(task)
            source_started = _trace_source_started(SourceType.WEB.value, task)
            try:
                with trace_execution_scope(
                    task_id=task.id if task else None,
                    source=SourceType.WEB.value,
                ):
                    result = await self.structured_web_executor.execute(
                        execution_question,
                        plan,
                    )
                answer = self._public_result(result)
            except WebResearchPlanningError as error:
                _trace_source_exception(SourceType.WEB.value, task, source_started, error)
                _trace_task_exception(task, task_started, error)
                logger.debug(
                    "WEB_RESEARCH_PLAN: %s",
                    json.dumps(
                        {
                            "status": "planning_failure_legacy_dispatch",
                            "error_type": type(error).__name__,
                        },
                        ensure_ascii=False,
                    ),
                )
                _trace_fallback(
                    from_stage="source_execution",
                    from_dispatch="structured_web_multi_hop",
                    target="legacy_react",
                    reason="web_research_planning_failure",
                    error_type=type(error).__name__,
                )
                return await self._run_legacy(task_plan.resolved_question, plan=plan)
            except Exception as error:
                _trace_source_exception(SourceType.WEB.value, task, source_started, error)
                _trace_task_exception(task, task_started, error)
                raise
            _trace_source_result(SourceType.WEB.value, task, source_started, result)
            _trace_task_result(task, task_started, result)
            return answer
        _trace_fallback(
            from_stage="dispatch",
            from_dispatch=dispatch,
            target="legacy_react",
            reason="no_supported_direct_dispatch",
        )
        return await self._run_legacy(task_plan.resolved_question, plan=plan)

    def _public_result(self, result) -> str:
        self.trace_recorder.set_run_outcome(result.status)
        return result.public_answer()

    def _persist_session(
        self,
        *,
        session_id: str,
        run_id: str,
        question: str,
        answer: str,
        plan: SourcePlan | None,
        initial_history: list | None = None,
    ) -> None:
        """Persist completed-run state without feeding it back into planning."""
        assert self.session_store is not None
        try:
            state = self.session_store.get(session_id) or SessionState(session_id)
            if initial_history and not state.recent_messages and not state.conversation_summary:
                state.recent_messages = [dict(item) for item in initial_history]
            if plan is not None and plan.company_name is not None:
                state.current_entity = plan.company_name
            if plan is not None and plan.catalog_doc_ids:
                state.current_doc_scope = list(plan.catalog_doc_ids)
            # An entity-free or failed plan must not erase a previously resolved
            # conversation subject. A later explicit entity match replaces it.
            state.append_message("user", question)
            state.append_message("assistant", answer)
            state.last_run_id = run_id
            self._maybe_compress_session(state)
            self.session_store.set(state)
        except Exception as error:
            logger.warning(
                "Redis session persistence unavailable for session %s: %s",
                session_id,
                error,
            )

    def _maybe_compress_session(self, state: SessionState) -> None:
        """Compress a complete older prefix; mutate state only after validation."""
        messages_before = len(state.recent_messages)
        if messages_before <= SUMMARY_TRIGGER_MESSAGES:
            record_trace_event(
                event_type="context.compression.completed",
                stage="context_compression",
                status="completed",
                attributes={
                    "compression_triggered": False,
                    "messages_before": messages_before,
                    "messages_after": messages_before,
                    "compressed_message_count": 0,
                    "remaining_message_count": messages_before,
                    "compression_input_chars": 0,
                    "summary_length": len(state.conversation_summary),
                },
            )
            return

        split_index = messages_before - SUMMARY_KEEP_RECENT_MESSAGES
        if (
            split_index > 0
            and split_index < messages_before
            and state.recent_messages[split_index - 1].get("role") == "user"
            and state.recent_messages[split_index].get("role") == "assistant"
        ):
            split_index -= 1
        pending_prefix = [
            dict(message) for message in state.recent_messages[:split_index]
        ]
        if not pending_prefix:
            return
        old_messages, compression_input_chars = self._select_compression_batch(
            state.conversation_summary,
            pending_prefix,
        )
        if not old_messages:
            logger.warning(
                "Session context compression skipped for session %s: oldest complete "
                "turn exceeds the %s character budget",
                state.session_id,
                COMPRESSION_MAX_CHARS,
            )
            record_trace_event(
                event_type="context.compression.failed",
                stage="context_compression",
                status="failed",
                error_type="CompressionBudgetExceeded",
                attributes={
                    "compression_triggered": True,
                    "messages_before": messages_before,
                    "messages_after": messages_before,
                    "compressed_message_count": 0,
                    "remaining_message_count": messages_before,
                    "compression_input_chars": compression_input_chars,
                    "summary_length": len(state.conversation_summary),
                },
            )
            return
        try:
            new_summary = self.context_compressor.compress(
                state.conversation_summary,
                old_messages,
            )
        except Exception as error:
            logger.warning(
                "Session context compression failed open for session %s: %s",
                state.session_id,
                error,
            )
            record_trace_event(
                event_type="context.compression.failed",
                stage="context_compression",
                status="failed",
                error_type=type(error).__name__,
                attributes={
                    "compression_triggered": True,
                    "messages_before": messages_before,
                    "messages_after": messages_before,
                    "compressed_message_count": 0,
                    "remaining_message_count": messages_before,
                    "compression_input_chars": compression_input_chars,
                    "summary_length": len(state.conversation_summary),
                },
            )
            return

        state.conversation_summary = new_summary
        state.recent_messages = [
            dict(message) for message in state.recent_messages[len(old_messages) :]
        ]
        record_trace_event(
            event_type="context.compression.completed",
            stage="context_compression",
            status="completed",
            attributes={
                "compression_triggered": True,
                "messages_before": messages_before,
                "messages_after": len(state.recent_messages),
                "compressed_message_count": len(old_messages),
                "remaining_message_count": len(state.recent_messages),
                "compression_input_chars": compression_input_chars,
                "summary_length": len(new_summary),
            },
        )

    @staticmethod
    def _select_compression_batch(
        existing_summary: str,
        pending_prefix: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], int]:
        """Return the largest budgeted prefix made only of complete turns."""
        selected: list[dict[str, str]] = []
        first_turn_chars = 0
        index = 0
        while index < len(pending_prefix):
            message = pending_prefix[index]
            role = message.get("role")
            if role == "user":
                if (
                    index + 1 >= len(pending_prefix)
                    or pending_prefix[index + 1].get("role") != "assistant"
                ):
                    break
                turn = [message, pending_prefix[index + 1]]
            elif role == "assistant":
                # An assistant message without its user message is not a safe
                # compression unit; leave it untouched for fail-open recovery.
                break
            else:
                turn = [message]

            candidate = selected + turn
            candidate_chars = ContextCompressor.prompt_length(
                existing_summary,
                candidate,
            )
            if not selected:
                first_turn_chars = candidate_chars
            if candidate_chars > COMPRESSION_MAX_CHARS:
                break
            selected = candidate
            index += len(turn)
        input_chars = (
            ContextCompressor.prompt_length(existing_summary, selected)
            if selected
            else first_turn_chars
        )
        return selected, input_chars

    async def _run_legacy(self, question: str, *, plan: SourcePlan | None = None) -> str:
        """Expose the otherwise opaque legacy boundary to eval observers."""
        allowed_doc_ids = self._legacy_document_scope(question, plan)
        notify_runtime_artifact(
            "legacy.run.started",
            {"question": question, "allowed_doc_ids": list(allowed_doc_ids)},
        )
        if allowed_doc_ids and _accepts_allowed_doc_ids(self.legacy_runner):
            answer = await self.legacy_runner(question, allowed_doc_ids=allowed_doc_ids)
        else:
            answer = await self.legacy_runner(question)
        notify_runtime_artifact("legacy.run.completed", {"answer": answer})
        return answer

    def _legacy_document_scope(
        self,
        question: str,
        plan: SourcePlan | None,
    ) -> tuple[str, ...]:
        catalog_doc_ids = plan.catalog_doc_ids if plan is not None else ()
        if not catalog_doc_ids:
            try:
                entity = self.planner.resolver.resolve(question)
                catalog_doc_ids = (
                    (entity.catalog_doc_id,) if entity.catalog_doc_id is not None else ()
                )
            except Exception:
                return ()
        if not catalog_doc_ids:
            return ()
        try:
            return tuple(
                self.direct_rag_executor.scope_resolver.resolve(catalog_doc_ids)
            )
        except Exception as error:
            logger.warning("Legacy document scope unavailable: %s", error)
            return ()


def _single_execution_task(task_plan):
    return task_plan.execution_tasks[0] if len(task_plan.execution_tasks) == 1 else None


def _trace_task_started(task) -> float:
    started = time.perf_counter()
    if task is not None:
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
    return started


def _trace_task_result(task, started: float, result) -> None:
    if task is None:
        return
    result_status = _result_status(result)
    successful = result_status == "success"
    record_trace_event(
        event_type=(
            "task.execution.completed" if successful else "task.execution.failed"
        ),
        stage="task_execution",
        status=result_status,
        task_id=task.id,
        source=task.source,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=None if successful else _result_error_type(result),
        attributes={
            "evidence_count": _result_evidence_count(result),
            "missing_information_count": _missing_information_count(result),
            "failure_status": None if successful else result_status,
        },
    )


def _trace_task_exception(task, started: float, error: Exception) -> None:
    if task is None:
        return
    record_trace_event(
        event_type="task.execution.failed",
        stage="task_execution",
        status="source_failure",
        task_id=task.id,
        source=task.source,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=type(error).__name__,
        attributes={
            "evidence_count": 0,
            "missing_information_count": 0,
            "failure_status": "source_failure",
        },
    )


def _trace_source_started(source: str, task) -> float:
    started = time.perf_counter()
    record_trace_event(
        event_type="source.execution.started",
        stage="source_execution",
        status="started",
        task_id=task.id if task else None,
        source=source,
        attempt=1,
        attributes={"evidence_count": 0},
    )
    return started


def _trace_source_result(source: str, task, started: float, result) -> None:
    result_status = _result_status(result)
    successful = result_status == "success"
    record_trace_event(
        event_type=(
            "source.execution.completed" if successful else "source.execution.failed"
        ),
        stage="source_execution",
        status=result_status,
        task_id=task.id if task else None,
        source=source,
        attempt=1,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=None if successful else _result_error_type(result),
        attributes={
            "evidence_count": _result_evidence_count(result),
            "source_result_status": result_status,
        },
    )


def _trace_source_exception(source: str, task, started: float, error: Exception) -> None:
    record_trace_event(
        event_type="source.execution.failed",
        stage="source_execution",
        status="source_failure",
        task_id=task.id if task else None,
        source=source,
        attempt=1,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=type(error).__name__,
        attributes={
            "evidence_count": 0,
            "source_result_status": "source_failure",
        },
    )


def _trace_fallback(
    *,
    from_stage: str,
    from_dispatch: str,
    target: str,
    reason: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    record_trace_event(
        event_type="fallback.dispatched",
        stage="fallback",
        status="dispatched",
        error_type=error_type,
        attributes={
            "from_stage": from_stage,
            "from_dispatch": from_dispatch,
            "target": target,
            "reason": reason,
            "error_message": error_message[:2000] if error_message else None,
        },
    )


def _result_status(result) -> str:
    status = _result_field(result, "status")
    return status if isinstance(status, str) and status else "success"


def _result_error_type(result) -> str | None:
    error_type = _result_field(result, "error_type")
    return error_type if isinstance(error_type, str) and error_type else None


def _result_evidence_count(result) -> int:
    fallback = _result_field(result, "fallback_result")
    if fallback is not None:
        return _result_evidence_count(fallback)
    evidence = _result_field(result, "web_evidence") or _result_field(
        result,
        "evidence",
    )
    if evidence is None:
        attempts = _result_field(result, "attempts", ())
        if attempts:
            return _result_evidence_count(attempts[-1])
        return 0
    try:
        return len(evidence.items)
    except (AttributeError, TypeError):
        return 0


def _missing_information_count(result) -> int:
    missing = _result_field(result, "missing_information", ())
    try:
        return len(missing)
    except TypeError:
        return 0


def _result_field(result, name: str, default=None):
    """Read only concrete result fields; Mock must not synthesize trace data."""

    namespace = getattr(result, "__dict__", None)
    if isinstance(namespace, dict) and name in namespace:
        return namespace[name]
    return default


def _accepts_allowed_doc_ids(call) -> bool:
    try:
        parameters = inspect.signature(call).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "allowed_doc_ids"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _infer_business_outcome(answer: str) -> str:
    for status in (
        "partial_success",
        "clarification_needed",
        "source_failure",
        "generation_failure",
        "provider_content_block",
        "insufficient",
        "source_scope_violation",
    ):
        if answer.startswith(f"{status}:"):
            return status
    try:
        payload = json.loads(answer)
    except (TypeError, json.JSONDecodeError):
        return "success"
    status = payload.get("status") if isinstance(payload, dict) else None
    return status if isinstance(status, str) and status else "success"


def build_default_entrypoint(
    legacy_runner: Callable[[str], Awaitable[str]],
    *,
    reference_date: date | None = None,
) -> OrchestrationEntrypoint:
    from config import REDIS_URL, SESSION_TTL_SECONDS, TRACE_DIR, TRACE_ENABLED

    from .document_catalog import load_default_catalog
    from .context_builder import ContextBuilder
    from .context_compressor import ContextCompressor
    from .entity_resolver import EntityResolver
    from .llm_router import LLMSourceRouter
    from .multi_source import MultiSourceOrchestrator
    from .multi_source_synthesis import MultiSourceSynthesizer
    from .task_decomposition import LightweightTaskDecomposer
    from .tool_necessity import ToolNecessityPolicy
    from .tool_registry import build_default_tool_registry
    from .web_fallback import ControlledWebFallbackExecutor
    from .web_research import StructuredWebResearchExecutor

    catalog = load_default_catalog()
    registry = build_default_tool_registry()
    planner = SourcePlanner(
        EntityResolver(catalog),
        registry,
        ambiguous_router=LLMSourceRouter(),
    )
    direct_rag_executor = DirectRAGExecutor(catalog)
    controlled_web_executor = ControlledWebFallbackExecutor()
    direct_sql_executor = DirectSQLExecutor()
    direct_web_executor = DirectWebExecutor(controlled_web_executor)
    structured_web_executor = StructuredWebResearchExecutor(
        step_executor=controlled_web_executor,
    )
    trace_recorder = (
        TraceRecorder(
            JsonlTraceSink(
                Path(TRACE_DIR) / "trace.jsonl",
                summary_path=Path(TRACE_DIR) / "run_summaries.jsonl",
            )
        )
        if TRACE_ENABLED
        else TraceRecorder(enabled=False)
    )
    try:
        session_store = RedisSessionStore(REDIS_URL, SESSION_TTL_SECONDS)
    except Exception as error:
        logger.warning("Redis session store initialization failed: %s", error)
        session_store = None
    return OrchestrationEntrypoint(
        planner=planner,
        direct_rag_executor=direct_rag_executor,
        legacy_runner=legacy_runner,
        rag_policy=RAGRetryFallbackPolicy(
            direct_rag_executor,
            controlled_web_executor,
        ),
        direct_sql_executor=direct_sql_executor,
        sql_fallback_policy=SQLWebFallbackPolicy(controlled_web_executor),
        direct_web_executor=direct_web_executor,
        structured_web_executor=structured_web_executor,
        multi_source_orchestrator=MultiSourceOrchestrator(
            direct_rag_executor,
            direct_sql_executor,
            direct_web_executor,
            MultiSourceSynthesizer(),
            structured_web_executor=structured_web_executor,
        ),
        task_planner=TaskPlanner(
            planner,
            reference_date=reference_date,
            task_decomposer=LightweightTaskDecomposer(),
            necessity_policy=ToolNecessityPolicy(registry),
        ),
        trace_recorder=trace_recorder,
        session_store=session_store,
        context_builder=ContextBuilder(),
        context_compressor=ContextCompressor(),
    )
