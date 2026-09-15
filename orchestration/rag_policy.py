"""Phase C2 bounded retry and controlled Web fallback policy for direct RAG."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

from .evidence import EvidenceBundle, merge_evidence_bundles
from .executors import DirectRAGExecutionResult, DirectRAGExecutor
from .models import SourcePlan
from .state import AgentState, SourceStatus
from .trace import get_current_trace_task_id, record_trace_event, trace_execution_scope
from .web_fallback import ControlledWebFallbackExecutor, WebFallbackExecutionResult


logger = logging.getLogger(__name__)
_TRACE_PREVIEW_CHARS = 300
_RISK_QUESTION_RE = re.compile(
    r"核心(?:经营)?风险|主要(?:经营)?风险|经营风险(?:是什么|有哪些|包括)|风险因素"
)
_RISK_RETRIEVAL_QUERY = (
    "风险因素 经营风险 产品质量控制 产品结构单一 市场竞争加剧 "
    "毛利率下降 原材料采购集中 应收账款 产品注册 经营业绩 管理 技术泄密"
)
_RISK_RETRY_QUERY = "第四节 风险因素 主要风险 持续盈利能力 不利因素 经营业绩不能持续增长"
_CONTAMINATED_FOCUS_RE = re.compile(
    r"证据[一二三四五六七八九十\d]|提供的证据|召回|检索结果|未明确提及|"
    r"没有任何证据|仅为|涉及.*(?:战略|术语|名称|格式)"
)
_MAX_RAG_QUERY_CHARS = 500
_RAG_QUERY_REWRITE_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class RAGPolicyExecutionResult:
    status: str
    answer: str | None
    attempts: tuple[DirectRAGExecutionResult, ...]
    fallback_result: WebFallbackExecutionResult | None = None
    missing_information: tuple[str, ...] = ()
    error_type: str | None = None
    latency_ms: int = 0
    state: AgentState | None = None

    def public_answer(self) -> str:
        if self.fallback_result is not None:
            return self.fallback_result.public_answer()
        if self.attempts:
            return self.attempts[-1].public_answer()
        return "insufficient: 本地文档检索未执行。"


class LLMRAGQueryRewriter:
    """Generate retrieval parameters semantically, then constrain them locally."""

    def __init__(
        self,
        llm_call: Callable[..., str] | None = None,
        *,
        timeout: int = _RAG_QUERY_REWRITE_TIMEOUT_SECONDS,
    ) -> None:
        self._llm_call = llm_call
        self.timeout = timeout

    def rewrite(
        self,
        question: str,
        plan: SourcePlan,
        *,
        attempt: int,
        previous_query: str | None = None,
        missing_information: tuple[str, ...] = (),
    ) -> str:
        call = self._llm_call
        if call is None:
            from config import call_llm

            call = call_llm
        prompt = self._prompt(
            question,
            plan,
            attempt=attempt,
            previous_query=previous_query,
            missing_information=missing_information,
        )
        kwargs = {"temperature": 0.0, "timeout": self.timeout}
        if self._llm_call is None:
            kwargs["trace_operation"] = "rag_query_rewrite"
        raw = call(prompt, **kwargs)
        candidate = self._parse(raw)
        return constrain_rag_query(
            candidate,
            question,
            plan,
            missing_information=missing_information,
        )

    @staticmethod
    def _prompt(
        question: str,
        plan: SourcePlan,
        *,
        attempt: int,
        previous_query: str | None,
        missing_information: tuple[str, ...],
    ) -> str:
        payload = {
            "question": question,
            "company_name": plan.company_name,
            "document_type": plan.document_type,
            "document_scope_locked": bool(plan.catalog_doc_ids),
            "attempt": attempt,
            "previous_query": previous_query,
            "missing_information": list(missing_information),
        }
        return (
            "你是金融文档 RAG 的 Query Planner，只生成检索参数，不回答问题、不选择数据源。"
            "文档范围已由系统锁定，不得改变公司、文档或年份。把自然语言问题改写成文档中可能真实出现的"
            "章节标题、指标名称、同义词和必要限定词；删除无检索价值的口语词。若 attempt=2，必须换一个"
            "检索角度，不要复述错误召回内容。只返回严格 JSON："
            '{"query":"核心检索短语","keywords":["关键词"],"section_hints":["章节标题"]}。'
            "每个数组最多 12 项，禁止输出解释。输入："
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    @staticmethod
    def _parse(raw: str) -> str:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("RAG query rewrite output is empty")
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
        if fenced:
            text = fenced.group(1)
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("RAG query rewrite output must be an object")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("RAG query rewrite query must be non-empty")
        parts = [query.strip()]
        for field in ("section_hints", "keywords"):
            values = payload.get(field, [])
            if not isinstance(values, list) or len(values) > 12 or any(
                not isinstance(value, str) for value in values
            ):
                raise ValueError(f"RAG query rewrite {field} is invalid")
            parts.extend(value.strip() for value in values if value.strip())
        result = _dedupe_query_terms(" ".join(parts), limit=24)
        if len(result) > _MAX_RAG_QUERY_CHARS:
            raise ValueError("RAG query rewrite is too long")
        return result


class RAGRetryFallbackPolicy:
    """Run at most two retrievals under one immutable physical scope."""

    def __init__(
        self,
        rag_executor: DirectRAGExecutor,
        web_executor: ControlledWebFallbackExecutor,
        query_rewriter: LLMRAGQueryRewriter | None = None,
    ) -> None:
        self.rag_executor = rag_executor
        self.web_executor = web_executor
        self.query_rewriter = query_rewriter

    async def execute(
        self,
        question: str,
        plan: SourcePlan,
        *,
        retrieval_query: str | None = None,
        retrieval_query_prepared: bool = False,
    ) -> RAGPolicyExecutionResult:
        started = time.perf_counter()
        state = AgentState({"rag"})
        allowed_index_doc_ids = self.rag_executor.scope_resolver.resolve(plan.catalog_doc_ids)
        attempts: list[DirectRAGExecutionResult] = []

        source_question = retrieval_query or question
        if retrieval_query_prepared and retrieval_query:
            first_query = constrain_rag_query(retrieval_query, question, plan)
            self._record_prepared_query(question, retrieval_query, first_query, plan)
        else:
            first_query = await asyncio.to_thread(
                self._rewrite_query,
                source_question,
                plan,
                attempt=1,
            )
        logger.debug(
            "RAG_ATTEMPT: %s",
            json.dumps(
                {
                    "attempt": 1,
                    "max_attempts": 2,
                    "query_preview": first_query[:_TRACE_PREVIEW_CHARS],
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                    "allowed_index_doc_ids": list(allowed_index_doc_ids),
                },
                ensure_ascii=False,
            ),
        )
        state.start_source("rag")
        first_started = _record_source_started("rag", 1)
        try:
            with trace_execution_scope(
                task_id=get_current_trace_task_id(), source="rag"
            ):
                first = await asyncio.to_thread(
                    self.rag_executor.execute,
                    question,
                    plan,
                    retrieval_query=first_query,
                    allowed_index_doc_ids=allowed_index_doc_ids,
                    trace_final=False,
                )
        except Exception as error:
            _record_source_exception("rag", 1, first_started, error)
            raise
        attempts.append(first)
        _record_source_trace_result("rag", 1, first_started, first)
        _record_source_result(state, "rag", first)
        if first.status != "insufficient":
            return self._finish(started, attempts, state=state)

        rewritten_query = await asyncio.to_thread(
            self._rewrite_query,
            source_question,
            plan,
            attempt=2,
            previous_query=first_query,
            missing_information=first.missing_information,
        )
        logger.debug(
            "RAG_QUERY_REWRITE: %s",
            json.dumps(
                {
                    "original_query": question[:_TRACE_PREVIEW_CHARS],
                    "missing_information": [
                        item[:_TRACE_PREVIEW_CHARS] for item in first.missing_information[:10]
                    ],
                    "rewritten_query": rewritten_query[:_TRACE_PREVIEW_CHARS],
                },
                ensure_ascii=False,
            ),
        )
        logger.debug(
            "RAG_ATTEMPT: %s",
            json.dumps(
                {
                    "attempt": 2,
                    "max_attempts": 2,
                    "query_preview": rewritten_query[:_TRACE_PREVIEW_CHARS],
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                    "allowed_index_doc_ids": list(allowed_index_doc_ids),
                },
                ensure_ascii=False,
            ),
        )
        state.start_source("rag")
        second_started = _record_source_started("rag", 2)
        try:
            with trace_execution_scope(
                task_id=get_current_trace_task_id(), source="rag"
            ):
                second = await asyncio.to_thread(
                    self.rag_executor.execute,
                    question,
                    plan,
                    retrieval_query=rewritten_query,
                    allowed_index_doc_ids=allowed_index_doc_ids,
                    trace_final=False,
                )
        except Exception as error:
            _record_source_exception("rag", 2, second_started, error)
            raise
        attempts.append(second)
        _record_source_trace_result("rag", 2, second_started, second)
        _record_source_result(state, "rag", second)
        if second.status != "insufficient":
            return self._finish(started, attempts, state=state)

        if not plan.web_fallback_allowed:
            return self._finish(started, attempts, state=state)

        logger.debug(
            "FALLBACK: %s",
            json.dumps(
                {
                    "from": "rag",
                    "to": "web",
                    "reason": "attempt1_and_attempt2_insufficient",
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                    "allowed_index_doc_ids": list(allowed_index_doc_ids),
                },
                ensure_ascii=False,
            ),
        )
        record_trace_event(
            event_type="fallback.dispatched",
            stage="fallback",
            status="dispatched",
            task_id=get_current_trace_task_id(),
            source="rag",
            attributes={
                "from_stage": "source_execution",
                "from_dispatch": "rag_first",
                "target": "web",
                "reason": "attempt1_and_attempt2_insufficient",
            },
        )
        state.record_fallback("rag", "web", "attempt1_and_attempt2_insufficient")
        state.start_source("web")
        rag_evidence = _merge_attempt_evidence(attempts)
        fallback_started = _record_source_started("web", 1)
        try:
            with trace_execution_scope(
                task_id=get_current_trace_task_id(), source="web"
            ):
                fallback = await self.web_executor.execute(
                    question,
                    rag_evidence=rag_evidence,
                    missing_information=second.missing_information or first.missing_information,
                    web_source_policy=plan.web_source_policy,
                )
        except Exception as error:
            _record_source_exception("web", 1, fallback_started, error)
            raise
        _record_source_trace_result("web", 1, fallback_started, fallback)
        _record_source_result(state, "web", fallback)
        return self._finish(started, attempts, fallback, state=state)

    @staticmethod
    def _record_prepared_query(
        question: str,
        prepared_query: str,
        final_query: str,
        plan: SourcePlan,
    ) -> None:
        record_trace_event(
            event_type="rag.query.rewrite.completed",
            stage="query_rewrite",
            status="completed",
            task_id=get_current_trace_task_id(),
            source="rag",
            attempt=1,
            attributes={
                "method": "upstream_llm+deterministic_guard",
                "error_type": None,
                "input": {
                    "question": question,
                    "prepared_query": prepared_query,
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                },
                "output": {"query": final_query},
            },
        )

    def _rewrite_query(
        self,
        question: str,
        plan: SourcePlan,
        *,
        attempt: int,
        previous_query: str | None = None,
        missing_information: tuple[str, ...] = (),
    ) -> str:
        method = "deterministic_fallback"
        error_type = None
        try:
            if self.query_rewriter is None:
                raise RuntimeError("LLM query rewriter is not configured")
            rewritten = self.query_rewriter.rewrite(
                question,
                plan,
                attempt=attempt,
                previous_query=previous_query,
                missing_information=missing_information,
            )
            method = "llm+deterministic_guard"
        except Exception as error:
            error_type = type(error).__name__
            logger.warning("RAG query rewrite fell back deterministically: %s", error)
            rewritten = (
                build_rag_query(question, plan)
                if attempt == 1
                else rewrite_rag_query(previous_query or question, missing_information)
            )
        record_trace_event(
            event_type="rag.query.rewrite.completed",
            stage="query_rewrite",
            status="completed",
            task_id=get_current_trace_task_id(),
            source="rag",
            attempt=attempt,
            attributes={
                "method": method,
                "error_type": error_type,
                "input": {
                    "question": question,
                    "previous_query": previous_query,
                    "missing_information": list(missing_information),
                    "catalog_doc_ids": list(plan.catalog_doc_ids),
                },
                "output": {"query": rewritten},
            },
        )
        return rewritten

    @staticmethod
    def _finish(
        started: float,
        attempts: list[DirectRAGExecutionResult],
        fallback: WebFallbackExecutionResult | None = None,
        *,
        state: AgentState | None = None,
    ) -> RAGPolicyExecutionResult:
        terminal = fallback or attempts[-1]
        result = RAGPolicyExecutionResult(
            status=terminal.status,
            answer=terminal.answer,
            attempts=tuple(attempts),
            fallback_result=fallback,
            missing_information=terminal.missing_information,
            error_type=terminal.error_type,
            latency_ms=round((time.perf_counter() - started) * 1000),
            state=state,
        )
        logger.debug(
            "SOURCE_COMPLETED: %s",
            json.dumps(
                {
                    "source": "rag" if fallback is None else "rag+web",
                    "status": result.status,
                    "attempt_count": len(result.attempts),
                    "fallback_used": fallback is not None,
                    "required_sources": sorted(state.required_sources) if state else ["rag"],
                    "completed_sources": sorted(state.completed_sources) if state else [],
                    "failed_sources": sorted(state.failed_sources) if state else [],
                    "latency_ms": result.latency_ms,
                },
                ensure_ascii=False,
            ),
        )
        if fallback is None:
            logger.debug(
                "FINAL: %s",
                json.dumps(
                    {
                        "status": result.status,
                        "attempt_count": len(result.attempts),
                        "fallback_used": False,
                        "answer_preview": (result.answer or "")[:_TRACE_PREVIEW_CHARS],
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
        return result


def rewrite_rag_query(question: str, missing_information: tuple[str, ...]) -> str:
    """Build a clean second query without feeding retrieval failures back into search."""

    clean_focus = _clean_missing_focus(missing_information)
    if _is_risk_question(question):
        suffix = f" {clean_focus}" if clean_focus else ""
        return f"{_RISK_RETRY_QUERY}{suffix}"

    focus = clean_focus or "相关章节 定义 数据 具体事实"
    return f"{question.strip()}\n补充检索关键词：{focus}"


def build_rag_query(question: str, plan: SourcePlan) -> str:
    """Turn a scoped business question into terms likely to occur in the document.

    Once the physical document scope is fixed, repeating the company name has little
    retrieval value.  For risk questions, prospectuses usually express the answer as
    a ``风险因素`` section and concrete ``...风险`` headings, so use those lexical
    anchors for dense retrieval, BM25, and reranking instead of the raw wording.
    """

    normalized = question.strip()
    if plan.catalog_doc_ids and _is_risk_question(normalized):
        return _RISK_RETRIEVAL_QUERY
    return normalized


def constrain_rag_query(
    candidate: str,
    question: str,
    plan: SourcePlan,
    *,
    missing_information: tuple[str, ...] = (),
) -> str:
    """Apply invariant-preserving terms after semantic LLM generation."""

    value = " ".join(candidate.strip().split())
    if not value or len(value) > _MAX_RAG_QUERY_CHARS:
        raise ValueError("generated RAG query is unusable")
    if plan.catalog_doc_ids and _is_risk_question(question):
        value = _dedupe_query_terms(f"{value} {_RISK_RETRIEVAL_QUERY}", limit=36)
    focus = _clean_missing_focus(missing_information)
    if focus and focus not in value:
        value += f" {focus}"
    return value[:_MAX_RAG_QUERY_CHARS].strip()


def _dedupe_query_terms(value: str, *, limit: int) -> str:
    terms: list[str] = []
    for term in value.split():
        # Section numbers are frequently document-specific and must not be
        # hallucinated by the semantic rewriter. Semantic heading text remains.
        if re.fullmatch(r"第[一二三四五六七八九十百\d]+[章节篇]", term):
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) == limit:
            break
    return " ".join(terms)


def _is_risk_question(text: str) -> bool:
    return bool(_RISK_QUESTION_RE.search(text))


def _clean_missing_focus(missing_information: tuple[str, ...]) -> str:
    clean: list[str] = []
    for item in missing_information:
        value = " ".join(item.strip().split())
        if not value or len(value) > 60 or _CONTAMINATED_FOCUS_RE.search(value):
            continue
        clean.append(value.rstrip("。；;，,"))
        if len(clean) == 3:
            break
    return " ".join(clean)


def _record_source_result(
    state: AgentState,
    source: str,
    result: DirectRAGExecutionResult | WebFallbackExecutionResult,
) -> None:
    evidence = result.web_evidence if source == "web" and isinstance(
        result, WebFallbackExecutionResult
    ) else result.evidence
    if (
        result.status == "success"
        and evidence is not None
        and evidence.execution_success
        and evidence.source_match
        and evidence.items
    ):
        state.complete_source(source, evidence, answer=result.answer)
        return
    status = {
        "insufficient": SourceStatus.INSUFFICIENT,
        "provider_content_block": SourceStatus.PROVIDER_CONTENT_BLOCK,
        "generation_failure": SourceStatus.GENERATION_FAILURE,
    }.get(result.status, SourceStatus.SOURCE_FAILURE)
    state.fail_source(
        source,
        status,
        evidence=evidence,
        error=result.error_type or result.status,
    )


def _merge_attempt_evidence(attempts: list[DirectRAGExecutionResult]) -> EvidenceBundle:
    bundles = [attempt.evidence for attempt in attempts if attempt.evidence is not None]
    if not bundles:
        raise ValueError("RAG fallback requires validated RAG evidence bundles")
    return merge_evidence_bundles(*bundles)


def _record_source_started(source: str, attempt: int) -> float:
    started = time.perf_counter()
    record_trace_event(
        event_type="source.execution.started",
        stage="source_execution",
        status="started",
        task_id=get_current_trace_task_id(),
        source=source,
        attempt=attempt,
        attributes={"evidence_count": 0},
    )
    return started


def _record_source_trace_result(source: str, attempt: int, started: float, result) -> None:
    result_status = str(getattr(result, "status", "source_failure"))
    successful = result_status == "success"
    evidence = (
        getattr(result, "web_evidence", None)
        if source == "web"
        else getattr(result, "evidence", None)
    )
    record_trace_event(
        event_type=(
            "source.execution.completed" if successful else "source.execution.failed"
        ),
        stage="source_execution",
        status=result_status,
        task_id=get_current_trace_task_id(),
        source=source,
        attempt=attempt,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=None if successful else getattr(result, "error_type", None),
        attributes={
            "evidence_count": len(evidence.items) if evidence is not None else 0,
            "source_result_status": result_status,
        },
    )


def _record_source_exception(
    source: str,
    attempt: int,
    started: float,
    error: Exception,
) -> None:
    record_trace_event(
        event_type="source.execution.failed",
        stage="source_execution",
        status="source_failure",
        task_id=get_current_trace_task_id(),
        source=source,
        attempt=attempt,
        duration_ms=round((time.perf_counter() - started) * 1000),
        error_type=type(error).__name__,
        attributes={
            "evidence_count": 0,
            "source_result_status": "source_failure",
        },
    )
