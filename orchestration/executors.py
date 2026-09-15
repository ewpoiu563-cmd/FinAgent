"""Direct source executors for the source-aware orchestration layer."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from config import TRACE_INCLUDE_CONTENT

from .document_catalog import DocumentCatalog
from .evidence import EvidenceBundle, normalize_rag_tool_result, validate_rag_evidence_scope
from .models import PlanMode, SourcePlan, SourceType
from .source_scope import SourceScopeResolver
from .sufficiency import EvidenceSufficiencyEvaluator, SufficiencyResult
from .trace import record_trace_event


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


def _rag_trace_output(result: Mapping[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence")
    items = evidence if isinstance(evidence, (list, tuple)) else ()
    return {
        "success": bool(result.get("success")),
        "error_type": result.get("error_type"),
        "error": str(result.get("error") or "")[:2000] or None,
        "candidate_count": result.get("candidate_count"),
        "retrieval_metadata": dict(result.get("retrieval_metadata") or {}),
        "evidence": [
            {
                "rank": item.get("rank"),
                "doc_id": item.get("doc_id"),
                "source_file": item.get("source_file"),
                "page": item.get("page"),
                "headings": item.get("headings"),
                "rerank_score": item.get("rerank_score"),
                "text": str(item.get("text") or "")[:1200],
                "text_truncated": len(str(item.get("text") or "")) > 1200,
            }
            for item in items[:8]
            if isinstance(item, Mapping)
        ],
        "truncated": len(items) > 8,
    }


@dataclass(frozen=True)
class DirectRAGExecutionResult:
    status: str
    answer: str | None
    allowed_index_doc_ids: tuple[str, ...]
    evidence: EvidenceBundle | None
    sufficiency: SufficiencyResult | None
    missing_information: tuple[str, ...] = ()
    error_type: str | None = None
    latency_ms: int = 0

    @property
    def successful(self) -> bool:
        return self.status == "success"

    def public_answer(self) -> str:
        if self.status == "success" and self.answer:
            return self.answer
        if self.status == "insufficient":
            detail = "；".join(self.missing_information) or "当前证据无法完整回答问题"
            return f"insufficient: 本地文档证据不足。缺失信息：{detail}"
        if self.status == "source_scope_violation":
            return "source_scope_violation: 检测到越界文档证据，已拦截且未进入答案合成。"
        if self.status == "generation_failure":
            return "generation_failure: 文档答案生成失败或超时，未返回猜测答案。"
        return "insufficient: 本地文档检索未获得可用于回答的有效证据。"


class DirectRAGExecutor:
    """Execute one scoped RAG call and one evidence-only synthesis call."""

    def __init__(
        self,
        catalog: DocumentCatalog,
        *,
        scope_resolver: SourceScopeResolver | None = None,
        retrieve_call: Callable[..., Mapping[str, Any]] | None = None,
        sufficiency_evaluator: EvidenceSufficiencyEvaluator | None = None,
    ) -> None:
        self.catalog = catalog
        self.scope_resolver = scope_resolver or SourceScopeResolver(catalog)
        self._retrieve_call = retrieve_call
        self.sufficiency_evaluator = sufficiency_evaluator or EvidenceSufficiencyEvaluator()

    def execute(
        self,
        question: str,
        plan: SourcePlan,
        *,
        retrieval_query: str | None = None,
        allowed_index_doc_ids: tuple[str, ...] | None = None,
        trace_final: bool = True,
    ) -> DirectRAGExecutionResult:
        if not is_direct_rag_plan(plan):
            raise ValueError("DirectRAGExecutor requires a clear single-source RAG plan")
        started = time.perf_counter()
        scoped_query = retrieval_query or question

        try:
            if allowed_index_doc_ids is None:
                allowed_index_doc_ids = self.scope_resolver.resolve(plan.catalog_doc_ids)
            if not allowed_index_doc_ids:
                raise ValueError("allowed_index_doc_ids must not be empty")
            retrieve = self._retrieve_call or _default_retrieve_document
            logger.debug(
                "TOOL_CALL: %s",
                json.dumps(
                    {
                        "tool_name": "retrieve_document",
                        "question_preview": scoped_query[:_TRACE_PREVIEW_CHARS],
                        "catalog_doc_ids": list(plan.catalog_doc_ids),
                        "allowed_index_doc_ids": list(allowed_index_doc_ids),
                    },
                    ensure_ascii=False,
                ),
            )
            tool_started = time.perf_counter()
            record_trace_event(
                event_type="tool.call.started",
                stage="tool_call",
                status="started",
                source="rag",
                attempt=1,
                attributes={
                    "tool_name": "retrieve_document",
                    "timeout": None,
                    "retry_count": 0,
                    "candidate_count": None,
                    **_trace_io(
                        input_value={
                            "query": scoped_query,
                            "catalog_doc_ids": list(plan.catalog_doc_ids),
                            "allowed_index_doc_ids": list(allowed_index_doc_ids),
                        }
                    ),
                },
            )
            try:
                raw_result = retrieve(scoped_query, allowed_doc_ids=allowed_index_doc_ids)
            except Exception as error:
                record_trace_event(
                    event_type="tool.call.failed",
                    stage="tool_call",
                    status="failed",
                    source="rag",
                    attempt=1,
                    duration_ms=round((time.perf_counter() - tool_started) * 1000),
                    error_type=type(error).__name__,
                    attributes={
                        "tool_name": "retrieve_document",
                        "timeout": None,
                        "retry_count": 0,
                        "candidate_count": None,
                        "evidence_count": None,
                        **_trace_io(
                            input_value={
                                "query": scoped_query,
                                "allowed_index_doc_ids": list(allowed_index_doc_ids),
                            },
                            output_value={
                                "error_type": type(error).__name__,
                                "error": str(error)[:2000],
                            },
                        ),
                    },
                )
                raise
            tool_latency_ms = round((time.perf_counter() - tool_started) * 1000)
            if not isinstance(raw_result, Mapping):
                record_trace_event(
                    event_type="tool.call.failed",
                    stage="tool_call",
                    status="failed",
                    source="rag",
                    attempt=1,
                    duration_ms=tool_latency_ms,
                    error_type="ValueError",
                    attributes={
                        "tool_name": "retrieve_document",
                        "timeout": None,
                        "retry_count": 0,
                        "candidate_count": None,
                        "evidence_count": None,
                    },
                )
                raise ValueError("retrieve_document must return an object")
            raw_evidence = raw_result.get("evidence", [])
            raw_evidence_count = len(raw_evidence) if isinstance(raw_evidence, (list, tuple)) else 0
            candidate_count = raw_result.get("candidate_count")
            retrieval_metadata = raw_result.get("retrieval_metadata")
            if (
                (not isinstance(candidate_count, int) or isinstance(candidate_count, bool))
                and isinstance(retrieval_metadata, Mapping)
            ):
                candidate_count = retrieval_metadata.get("candidate_count")
            if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
                candidate_count = raw_result.get("retrieved_count")
            if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
                candidate_count = None
            tool_success = bool(raw_result.get("success"))
            record_trace_event(
                event_type="tool.call.completed" if tool_success else "tool.call.failed",
                stage="tool_call",
                status="completed" if tool_success else "failed",
                source="rag",
                attempt=1,
                duration_ms=tool_latency_ms,
                error_type=(
                    str(raw_result.get("error_type")) if raw_result.get("error_type") else None
                ),
                attributes={
                    "tool_name": "retrieve_document",
                    "timeout": None,
                    "retry_count": 0,
                    "candidate_count": candidate_count,
                    "evidence_count": raw_evidence_count,
                    "result_count": raw_evidence_count,
                    "execution_success": tool_success,
                    **_trace_io(
                        input_value={
                            "query": scoped_query,
                            "catalog_doc_ids": list(plan.catalog_doc_ids),
                            "allowed_index_doc_ids": list(allowed_index_doc_ids),
                        },
                        output_value=_rag_trace_output(raw_result),
                    ),
                },
            )
            logger.debug(
                "TOOL_RESULT: %s",
                json.dumps(
                    {
                        "tool_name": "retrieve_document",
                        "success": bool(raw_result.get("success")),
                        "evidence_count": raw_evidence_count,
                        "allowed_index_doc_ids": list(allowed_index_doc_ids),
                        "latency_ms": tool_latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
            bundle = normalize_rag_tool_result(raw_result, self.catalog)
            validated = validate_rag_evidence_scope(bundle, allowed_index_doc_ids)
        except Exception as error:
            return self._finish(
                started,
                trace_final=trace_final,
                status="retrieval_failure",
                allowed_index_doc_ids=locals().get("allowed_index_doc_ids", ()),
                error_type=type(error).__name__,
            )

        if validated.scope_violation_count:
            return self._finish(
                started,
                trace_final=trace_final,
                status="source_scope_violation",
                allowed_index_doc_ids=allowed_index_doc_ids,
                evidence=validated,
                error_type="SOURCE_SCOPE_VIOLATION",
            )
        if not validated.execution_success or not validated.items:
            return self._finish(
                started,
                trace_final=trace_final,
                status="insufficient",
                allowed_index_doc_ids=allowed_index_doc_ids,
                evidence=validated,
                missing_information=("未检索到有效的限定范围文档证据",),
            )

        sufficiency = self.sufficiency_evaluator.evaluate(
            question,
            validated,
            evidence_selection_context=(scoped_query if scoped_query != question else None),
        )
        if sufficiency.generation_failure:
            logger.debug(
                "SYNTHESIS: %s",
                json.dumps(
                    {
                        "status": "generation_failure",
                        "evidence_count": len(validated.items),
                        "supported_evidence_ids": [],
                        "error_type": sufficiency.error_type,
                        "latency_ms": sufficiency.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
            return self._finish(
                started,
                trace_final=trace_final,
                status="generation_failure",
                allowed_index_doc_ids=allowed_index_doc_ids,
                evidence=validated,
                sufficiency=sufficiency,
                error_type=sufficiency.error_type,
            )
        if not sufficiency.sufficient:
            logger.debug(
                "SYNTHESIS: %s",
                json.dumps(
                    {
                        "status": "skipped_insufficient",
                        "evidence_count": len(validated.items),
                        "supported_evidence_ids": list(sufficiency.supported_evidence_ids),
                        "latency_ms": sufficiency.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
            return self._finish(
                started,
                trace_final=trace_final,
                status="insufficient",
                allowed_index_doc_ids=allowed_index_doc_ids,
                evidence=validated,
                sufficiency=sufficiency,
                missing_information=sufficiency.missing_information,
            )

        assert sufficiency.answer is not None
        if _contains_out_of_scope_reference(
            sufficiency.answer,
            validated,
            self.catalog,
        ):
            logger.debug(
                "SYNTHESIS: %s",
                json.dumps(
                    {
                        "status": "generation_failure",
                        "evidence_count": len(validated.items),
                        "supported_evidence_ids": list(sufficiency.supported_evidence_ids),
                        "error_type": "OutOfScopeAnswerReference",
                        "latency_ms": sufficiency.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
            return self._finish(
                started,
                trace_final=trace_final,
                status="generation_failure",
                allowed_index_doc_ids=allowed_index_doc_ids,
                evidence=validated,
                sufficiency=sufficiency,
                error_type="OutOfScopeAnswerReference",
            )
        answer = _answer_with_provenance(
            sufficiency.answer,
            sufficiency.supported_evidence_ids,
            validated,
        )
        logger.debug(
            "SYNTHESIS: %s",
            json.dumps(
                {
                    "status": "success",
                    "evidence_count": len(validated.items),
                    "supported_evidence_ids": list(sufficiency.supported_evidence_ids),
                    "answer_preview": answer[:_TRACE_PREVIEW_CHARS],
                    "latency_ms": sufficiency.latency_ms,
                },
                ensure_ascii=False,
            ),
        )
        return self._finish(
            started,
            trace_final=trace_final,
            status="success",
            answer=answer,
            allowed_index_doc_ids=allowed_index_doc_ids,
            evidence=validated,
            sufficiency=sufficiency,
        )

    @staticmethod
    def _finish(
        started: float,
        *,
        status: str,
        answer: str | None = None,
        allowed_index_doc_ids: tuple[str, ...] = (),
        evidence: EvidenceBundle | None = None,
        sufficiency: SufficiencyResult | None = None,
        missing_information: tuple[str, ...] = (),
        error_type: str | None = None,
        trace_final: bool = True,
    ) -> DirectRAGExecutionResult:
        result = DirectRAGExecutionResult(
            status=status,
            answer=answer,
            allowed_index_doc_ids=tuple(allowed_index_doc_ids),
            evidence=evidence,
            sufficiency=sufficiency,
            missing_information=tuple(missing_information),
            error_type=error_type,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        if trace_final:
            logger.debug(
                "FINAL: %s",
                json.dumps(
                    {
                        "status": result.status,
                        "allowed_index_doc_ids": list(result.allowed_index_doc_ids),
                        "evidence_count": len(result.evidence.items) if result.evidence else 0,
                        "violation_count": (
                            result.evidence.scope_violation_count if result.evidence else 0
                        ),
                        "sufficient": bool(result.sufficiency and result.sufficiency.sufficient),
                        "supported_evidence_ids": (
                            list(result.sufficiency.supported_evidence_ids)
                            if result.sufficiency
                            else []
                        ),
                        "answer_preview": (result.answer or "")[:_TRACE_PREVIEW_CHARS],
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
        return result


def is_direct_rag_plan(plan: SourcePlan) -> bool:
    return (
        plan.mode is PlanMode.SINGLE_SOURCE
        and plan.primary_source is SourceType.RAG
        and plan.required_sources == (SourceType.RAG,)
        and bool(plan.catalog_doc_ids)
    )


def _default_retrieve_document(question: str, *, allowed_doc_ids: tuple[str, ...]) -> Mapping[str, Any]:
    try:
        from tools.document_retrieval_tool import retrieve_document
    except ImportError:
        from ..tools.document_retrieval_tool import retrieve_document

    return retrieve_document(question, allowed_doc_ids=allowed_doc_ids)


def _answer_with_provenance(
    answer: str,
    supported_evidence_ids: tuple[str, ...],
    evidence: EvidenceBundle,
) -> str:
    by_id = {item.retrieval_unit_id: item for item in evidence.items}
    citations = []
    for identifier in supported_evidence_ids:
        item = by_id[identifier]
        page = "、".join(str(value) for value in item.page) if item.page else "未知"
        citations.append(
            f"- [{identifier}] {item.company_name or '未知公司'}；"
            f"{item.source_file or '未知文件'}；第{page}页"
        )
    return answer.strip() + "\n\n证据来源：\n" + "\n".join(citations)


def _contains_out_of_scope_reference(
    answer: str,
    evidence: EvidenceBundle,
    catalog: DocumentCatalog,
) -> bool:
    allowed_catalog_ids = {item.catalog_doc_id for item in evidence.items}
    answer_folded = answer.casefold()
    for document in catalog.records:
        if document.catalog_doc_id in allowed_catalog_ids:
            continue
        forbidden = (
            document.catalog_doc_id,
            document.source_file,
            document.company_name,
            *document.aliases,
            *document.index_doc_ids,
        )
        if any(value.casefold() in answer_folded for value in forbidden):
            logger.debug(
                "SYNTHESIS_SCOPE_VIOLATION: %s",
                json.dumps(
                    {
                        "catalog_doc_id": document.catalog_doc_id,
                        "answer_preview": answer[:_TRACE_PREVIEW_CHARS],
                    },
                    ensure_ascii=False,
                ),
            )
            return True
    return False
