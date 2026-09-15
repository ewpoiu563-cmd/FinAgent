"""Phase C3 direct SQL execution and evidence-only result synthesis."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from config import TRACE_INCLUDE_CONTENT

from .evidence import EvidenceBundle, normalize_sql_tool_result
from .models import PlanMode, SourcePlan, SourceType
from .sufficiency import SQLResultSufficiencyEvaluator, SufficiencyResult
from .trace import record_trace_event
from text2sql.semantic_planner import SQLSemanticPlanner


logger = logging.getLogger(__name__)
_TRACE_PREVIEW_CHARS = 300


def _trace_io(*, input_value=None, output_value=None) -> dict:
    if not TRACE_INCLUDE_CONTENT:
        return {}
    payload = {}
    if input_value is not None:
        payload["input"] = input_value
    if output_value is not None:
        payload["output"] = output_value
    return payload


def _sql_trace_output(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = result.get("rows")
    row_items = rows if isinstance(rows, (list, tuple)) else ()
    return {
        "success": bool(result.get("success")),
        "generated_sql": result.get("generated_sql"),
        "columns": list(result.get("columns") or ()),
        "rows": list(row_items[:20]),
        "row_count": result.get("row_count", len(row_items)),
        "truncated": bool(result.get("truncated", False)) or len(row_items) > 20,
        "error_type": result.get("error_type"),
        "error": str(result.get("error") or "")[:2000] or None,
    }


@dataclass(frozen=True)
class DirectSQLExecutionResult:
    status: str
    answer: str | None
    evidence: EvidenceBundle | None
    sufficiency: SufficiencyResult | None
    missing_information: tuple[str, ...] = ()
    error_type: str | None = None
    latency_ms: int = 0
    attempts: tuple[Mapping[str, Any], ...] = ()

    @property
    def execution_success(self) -> bool:
        return bool(self.evidence and self.evidence.execution_success)

    @property
    def result_sufficient(self) -> bool:
        return bool(self.sufficiency and self.sufficiency.sufficient)

    def public_answer(self) -> str:
        if self.status == "success" and self.answer:
            return self.answer
        if self.status == "source_failure":
            return "数据库查询未能完成。请稍后重试，或缩小查询范围后再试。"
        if self.status == "generation_failure":
            return "查询已完成，但结果暂时无法整理为可靠答案。请稍后重试。"
        if self.status == "clarification_needed":
            detail = "；".join(self.missing_information) or "SQL 查询时间或指标口径需要澄清"
            return detail
        detail = "；".join(self.missing_information) or "SQL 查询结果不足以完整回答问题"
        return f"当前数据库结果不足以完整回答该问题：{detail}"


@dataclass(frozen=True)
class SQLRepairDecision:
    retry: bool
    reason: str
    repair_instruction: str | None = None


class ControlledSQLRepairDecider:
    """Deterministically decide whether an SQL observation has a bounded repair path."""

    _REPAIRABLE_SUFFICIENCY_SIGNALS = (
        "limit", "排序", "时间条件", "日期格式", "字段", "语义", "截断", "范围",
    )

    def decide(
        self,
        observation: Mapping[str, Any],
        *,
        attempt: int,
        max_attempts: int,
    ) -> SQLRepairDecision:
        if attempt >= max_attempts:
            return SQLRepairDecision(False, "maximum_attempts_reached")
        validation_error = str(observation.get("validation_error") or "")
        execution_error = str(observation.get("execution_error") or "")
        error_type = str(observation.get("error_type") or "")
        if validation_error or error_type == "validation_error":
            return SQLRepairDecision(
                True,
                "repairable_validation_error",
                "修正语法、字段、语义契约、LIMIT、排序或时间条件；保持已确认 intent 不变。",
            )
        if execution_error or error_type == "sqlite_error":
            return SQLRepairDecision(
                True,
                "repairable_execution_error",
                "根据 SQLite 错误修正表名、字段名、表达式或语法；保持只读 SELECT。",
            )
        if error_type in {"generation_error", "compilation_error"} and observation.get("generated_sql"):
            return SQLRepairDecision(
                True,
                "repairable_generation_error",
                "重新生成一条符合 confirmed_sql_intent 与只读安全限制的 SQL。",
            )
        if observation.get("row_count") == 0:
            direction = _empty_result_repair_direction(observation)
            return SQLRepairDecision(bool(direction), "empty_result_with_direction" if direction else "empty_result_without_direction", direction)
        sufficiency = observation.get("sufficiency")
        if isinstance(sufficiency, Mapping) and sufficiency.get("sufficient") is False:
            missing = "；".join(str(item) for item in sufficiency.get("missing_information", ()) if item)
            if any(signal.casefold() in missing.casefold() for signal in self._REPAIRABLE_SUFFICIENCY_SIGNALS):
                return SQLRepairDecision(True, "repairable_sufficiency_gap", missing)
        return SQLRepairDecision(False, "no_explicit_repair_direction")


class DirectSQLExecutor:
    """Run a bounded SQL retrieve-evaluate-repair loop."""

    def __init__(
        self,
        *,
        query_call: Callable[[str], Mapping[str, Any]] | None = None,
        sufficiency_evaluator: SQLResultSufficiencyEvaluator | None = None,
        semantic_planner: SQLSemanticPlanner | None = None,
        repair_decider: ControlledSQLRepairDecider | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._query_call = query_call
        self.sufficiency_evaluator = sufficiency_evaluator or SQLResultSufficiencyEvaluator()
        self.semantic_planner = semantic_planner or SQLSemanticPlanner()
        self.repair_decider = repair_decider or ControlledSQLRepairDecider()
        self.max_attempts = max_attempts

    def execute(
        self,
        question: str,
        plan: SourcePlan,
        *,
        trace_final: bool = True,
    ) -> DirectSQLExecutionResult:
        if not is_direct_sql_plan(plan):
            raise ValueError("DirectSQLExecutor requires a clear single-source SQL plan")
        started = time.perf_counter()
        intent = self.semantic_planner.plan(question)
        logger.debug("SQL_SEMANTIC_INTENT: %s", json.dumps(intent.to_dict(), ensure_ascii=False))
        if intent.clarification_needed:
            return self._finish(
                started,
                status="clarification_needed",
                missing_information=(
                    intent.clarification_question or "SQL 查询语义需要澄清",
                ),
                trace_final=trace_final,
            )
        query = self._query_call or _default_query_financial_db
        observations: list[Mapping[str, Any]] = []
        repair_observation: Mapping[str, Any] | None = None
        terminal: DirectSQLExecutionResult | None = None
        for attempt in range(1, self.max_attempts + 1):
            terminal = self._execute_attempt(
                question,
                plan,
                intent=intent,
                query=query,
                attempt=attempt,
                repair_observation=repair_observation,
            )
            observation = _build_observation(terminal, attempt)
            observations.append(observation)
            decision = self.repair_decider.decide(
                observation,
                attempt=attempt,
                max_attempts=self.max_attempts,
            )
            record_trace_event(
                event_type="sql.repair.decision",
                stage="sql_repair",
                status="retry" if decision.retry else "stop",
                source="sql",
                attempt=attempt,
                attributes={
                    "retry": decision.retry,
                    "reason": decision.reason,
                    "next_attempt": attempt + 1 if decision.retry else None,
                    "row_count": observation.get("row_count"),
                    "validation_error": observation.get("validation_error"),
                    "execution_error": observation.get("execution_error"),
                    "result_sufficient": observation.get("sufficiency", {}).get("sufficient") if isinstance(observation.get("sufficiency"), Mapping) else None,
                },
            )
            if not decision.retry:
                break
            repair_observation = {
                **observation,
                "repair_instruction": decision.repair_instruction,
            }
        assert terminal is not None
        return self._finish(
            started,
            status=terminal.status,
            answer=terminal.answer,
            evidence=terminal.evidence,
            sufficiency=terminal.sufficiency,
            missing_information=terminal.missing_information,
            error_type=terminal.error_type,
            attempts=tuple(observations),
            trace_final=trace_final,
        )

    def _execute_attempt(
        self,
        question: str,
        plan: SourcePlan,
        *,
        intent,
        query: Callable[..., Mapping[str, Any]],
        attempt: int,
        repair_observation: Mapping[str, Any] | None,
    ) -> DirectSQLExecutionResult:
        started = time.perf_counter()
        logger.debug(
            "TOOL_CALL: %s",
            json.dumps(
                {
                    "tool_name": "query_financial_db",
                    "question_preview": question[:_TRACE_PREVIEW_CHARS],
                },
                ensure_ascii=False,
            ),
        )
        tool_started = time.perf_counter()
        record_trace_event(
            event_type="tool.call.started",
            stage="tool_call",
            status="started",
            source="sql",
            attempt=attempt,
            attributes={
                "tool_name": "query_financial_db",
                "timeout": None,
                "retry_count": attempt - 1,
                **_trace_io(input_value={"question": question}),
            },
        )
        try:
            raw_result = (
                query(question)
                if self._query_call is not None
                else _default_query_financial_db(
                    question,
                    sql_intent=intent,
                    repair_observation=repair_observation,
                )
            )
            if not isinstance(raw_result, Mapping):
                raise ValueError("query_financial_db must return an object")
            evidence = normalize_sql_tool_result(raw_result)
        except Exception as error:
            record_trace_event(
                event_type="tool.call.failed",
                stage="tool_call",
                status="failed",
                source="sql",
                attempt=attempt,
                duration_ms=round((time.perf_counter() - tool_started) * 1000),
                error_type=type(error).__name__,
                attributes={
                    "tool_name": "query_financial_db",
                    "timeout": None,
                    "retry_count": attempt - 1,
                    "row_count": None,
                    **_trace_io(
                        input_value={"question": question},
                        output_value={
                            "error_type": type(error).__name__,
                            "error": str(error)[:2000],
                        },
                    ),
                },
            )
            logger.debug(
                "SQL_EXECUTION: %s",
                json.dumps(
                    {
                        "execution_success": False,
                        "error_type": type(error).__name__,
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                    },
                    ensure_ascii=False,
                ),
            )
            return self._finish(
                started,
                status="source_failure",
                error_type=type(error).__name__,
                trace_final=False,
            )

        row_count = raw_result.get("row_count")
        if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
            row_count = len(raw_result.get("rows", ())) if isinstance(raw_result.get("rows"), (list, tuple)) else 0
        record_trace_event(
            event_type=(
                "tool.call.completed" if evidence.execution_success else "tool.call.failed"
            ),
            stage="tool_call",
            status="completed" if evidence.execution_success else "failed",
            source="sql",
            attempt=attempt,
            duration_ms=round((time.perf_counter() - tool_started) * 1000),
            error_type=(
                str(raw_result.get("error_type")) if raw_result.get("error_type") else None
            ),
            attributes={
                "tool_name": "query_financial_db",
                "timeout": None,
                "retry_count": attempt - 1,
                "row_count": row_count,
                "execution_success": evidence.execution_success,
                "truncated": bool(raw_result.get("truncated", False)),
                **_trace_io(
                    input_value={"question": question},
                    output_value=_sql_trace_output(raw_result),
                ),
            },
        )

        logger.debug(
            "TOOL_RESULT: %s",
            json.dumps(
                {
                    "tool_name": "query_financial_db",
                    "success": evidence.execution_success,
                    "row_count": raw_result.get("row_count", 0),
                    "truncated": raw_result.get("truncated", False),
                    "error_type": raw_result.get("error_type"),
                    "latency_ms": raw_result.get("latency_ms"),
                },
                ensure_ascii=False,
            ),
        )
        logger.debug(
            "SQL_EXECUTION: %s",
            json.dumps(
                {
                    "execution_success": evidence.execution_success,
                    "generated_sql_preview": str(raw_result.get("generated_sql") or "")[:_TRACE_PREVIEW_CHARS],
                    "row_count": raw_result.get("row_count", 0),
                    "truncated": raw_result.get("truncated", False),
                    "error_type": raw_result.get("error_type"),
                    "latency_ms": raw_result.get("latency_ms"),
                },
                ensure_ascii=False,
            ),
        )
        if not evidence.execution_success:
            if raw_result.get("error_type") == "clarification_needed":
                return self._finish(
                    started,
                    status="clarification_needed",
                    evidence=evidence,
                    missing_information=(str(raw_result.get("error") or "SQL 查询语义需要澄清"),),
                    error_type="clarification_needed",
                    trace_final=False,
                )
            return self._finish(
                started,
                status="source_failure",
                evidence=evidence,
                error_type=str(raw_result.get("error_type") or "sql_source_error"),
                trace_final=False,
            )

        empty_result = _is_empty_record_result(evidence, intent)
        if empty_result:
            if any(marker in question for marker in ("不存在", "没有数据", "没有记录", "未查到")):
                return self._finish(
                    started,
                    status="success",
                    answer="本地数据库未查到符合条件的记录。",
                    evidence=evidence,
                    sufficiency=SufficiencyResult(
                        sufficient=True,
                        answer="本地数据库未查到符合条件的记录。",
                        missing_information=(),
                        supported_evidence_ids=(),
                    ),
                    trace_final=False,
                )
            # A successful SQL statement can still leave the atomic requirement
            # unanswered.  Do this deterministically rather than relying on an
            # answer model to infer that ``rows=[]`` cannot support a requested
            # ranking/entity/metric record.  The evidence remains attached so
            # technical execution provenance is not conflated with insufficiency.
            sufficiency = SufficiencyResult(
                sufficient=False,
                answer=None,
                missing_information=(
                    "数据库未返回满足当前 requirement 的记录（rows=0）",
                ),
                supported_evidence_ids=(),
            )
        else:
            sufficiency = self.sufficiency_evaluator.evaluate(
                question,
                evidence,
                sql_intent=intent,
            )
        logger.debug(
            "SQL_RESULT_EVALUATION: %s",
            json.dumps(
                {
                    "execution_success": True,
                    "result_sufficient": sufficiency.sufficient,
                    "generation_failure": sufficiency.generation_failure,
                    "empty_record_result": empty_result,
                    "row_count": raw_result.get("row_count", 0),
                    "supported_evidence_ids": list(sufficiency.supported_evidence_ids),
                    "latency_ms": sufficiency.latency_ms,
                },
                ensure_ascii=False,
            ),
        )
        if sufficiency.generation_failure:
            return self._finish(
                started,
                status="generation_failure",
                evidence=evidence,
                sufficiency=sufficiency,
                error_type=sufficiency.error_type,
                trace_final=False,
            )
        if not sufficiency.sufficient:
            return self._finish(
                started,
                status="insufficient",
                evidence=evidence,
                sufficiency=sufficiency,
                missing_information=sufficiency.missing_information,
                trace_final=False,
            )

        assert sufficiency.answer is not None
        item = evidence.items[0]
        answer = (
            sufficiency.answer.strip()
            + f"\n\n数据来源：本地金融数据库（共返回 {item.row_count} 条记录）。"
        )
        logger.debug(
            "SYNTHESIS: %s",
            json.dumps(
                {
                    "source_type": "sql",
                    "status": "success",
                    "answer_preview": answer[:_TRACE_PREVIEW_CHARS],
                    "latency_ms": sufficiency.latency_ms,
                },
                ensure_ascii=False,
            ),
        )
        return self._finish(
            started,
            status="success",
            answer=answer,
            evidence=evidence,
            sufficiency=sufficiency,
            trace_final=False,
        )

    @staticmethod
    def _finish(
        started: float,
        *,
        status: str,
        answer: str | None = None,
        evidence: EvidenceBundle | None = None,
        sufficiency: SufficiencyResult | None = None,
        missing_information: tuple[str, ...] = (),
        error_type: str | None = None,
        attempts: tuple[Mapping[str, Any], ...] = (),
        trace_final: bool = True,
    ) -> DirectSQLExecutionResult:
        result = DirectSQLExecutionResult(
            status=status,
            answer=answer,
            evidence=evidence,
            sufficiency=sufficiency,
            missing_information=tuple(missing_information),
            error_type=error_type,
            latency_ms=round((time.perf_counter() - started) * 1000),
            attempts=attempts,
        )
        if trace_final:
            logger.debug(
                "SOURCE_COMPLETED: %s",
                json.dumps(
                    {
                        "source": "sql",
                        "status": result.status,
                        "execution_success": result.execution_success,
                        "result_sufficient": result.result_sufficient,
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
            logger.debug(
                "FINAL: %s",
                json.dumps(
                    {
                        "status": result.status,
                        "answer_preview": (result.answer or "")[:_TRACE_PREVIEW_CHARS],
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
        return result


def is_direct_sql_plan(plan: SourcePlan) -> bool:
    return (
        plan.mode is PlanMode.SINGLE_SOURCE
        and plan.primary_source is SourceType.SQL
        and plan.required_sources == (SourceType.SQL,)
    )


def _is_empty_record_result(evidence: EvidenceBundle, intent) -> bool:
    """Identify a successful zero-row result that cannot answer this intent.

    This intentionally examines result cardinality, never numeric cell values:
    a one-row ``COUNT/SUM/... = 0`` aggregate remains a valid observation.  The
    a zero-row result contains no record evidence. Aggregate queries such as
    COUNT/SUM still return one row (possibly with a numeric zero), so they remain
    eligible for sufficiency evaluation.
    """
    if len(evidence.items) != 1:
        return False
    item = evidence.items[0]
    if item.row_count != 0:
        return False
    return True


def _build_observation(result: DirectSQLExecutionResult, attempt: int) -> dict[str, Any]:
    raw = dict(result.evidence.raw_tool_result) if result.evidence is not None else {}
    sufficiency = result.sufficiency
    return {
        "attempt": attempt,
        "generated_sql": raw.get("generated_sql"),
        "rows": raw.get("rows", []),
        "row_count": raw.get("row_count", 0),
        "columns": raw.get("columns", []),
        "validation_error": raw.get("validation_error") or (
            raw.get("error") if raw.get("error_type") == "validation_error" else None
        ),
        "execution_error": raw.get("execution_error") or (
            raw.get("error") if raw.get("error_type") in {"sqlite_error", "timeout"} else None
        ),
        "error_type": raw.get("error_type") or result.error_type,
        "sufficiency": {
            "sufficient": sufficiency.sufficient,
            "generation_failure": sufficiency.generation_failure,
            "missing_information": list(sufficiency.missing_information),
        } if sufficiency is not None else None,
    }


def _empty_result_repair_direction(observation: Mapping[str, Any]) -> str | None:
    """Return a repair instruction only when the SQL itself exposes a concrete mismatch."""
    sql = str(observation.get("generated_sql") or "")
    if re.search(r"\[(?:交易日期|持仓日期)\].*['\"]\d{4}-\d{2}", sql, re.IGNORECASE):
        return "日期字段使用 YYYYMMDD；移除日期中的连字符后重试，不得扩大时间范围。"
    if re.search(r"\[截止日期\].*['\"]\d{8}", sql, re.IGNORECASE):
        return "截止日期使用 YYYY-MM-DD HH:MM:SS；按该格式修正后重试，不得扩大时间范围。"
    return None


def _default_query_financial_db(
    question: str,
    *,
    sql_intent=None,
    repair_observation: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    try:
        from tools.financial_sql_tool import query_financial_db
    except ImportError:
        from ..tools.financial_sql_tool import query_financial_db
    return query_financial_db(
        question,
        sql_intent=sql_intent,
        repair_observation=repair_observation,
    )
