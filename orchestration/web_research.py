"""Dependency-aware execution of structured Web research plans."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, replace

from .evidence import EvidenceBundle, merge_evidence_bundles
from .models import PlanMode, SourcePlan, SourceType
from .sufficiency import EvidenceSufficiencyEvaluator, SufficiencyResult
from .web_fallback import ControlledWebFallbackExecutor
from .finding_extractor import is_atomic_expected_output
from .web_planner import (
    WebResearchPlan,
    WebResearchPlanner,
    WebResearchPlanningError,
    WebTaskComplexity,
    classify_web_task,
    materialize_subquestion,
)


logger = logging.getLogger(__name__)
_TRACE_PREVIEW_CHARS = 300


@dataclass(frozen=True)
class StructuredWebResearchResult:
    status: str
    answer: str | None
    plan: WebResearchPlan | None
    findings: tuple[tuple[str, str], ...]
    evidence: EvidenceBundle | None
    sufficiency: SufficiencyResult | None
    missing_information: tuple[str, ...] = ()
    error_type: str | None = None
    latency_ms: int = 0
    finding_records: tuple["WebFinding", ...] = ()
    final_hop_search_completed: bool = False
    final_hop_evidence: EvidenceBundle | None = None

    def public_answer(self) -> str:
        if self.status == "success" and self.answer:
            return self.answer
        if self.status == "planning_failure":
            raise WebResearchPlanningError(self.error_type or "Web planning failed")
        detail = "；".join(self.missing_information) or "结构化 Web 研究证据不足"
        verified = [record for record in self.finding_records if record.verified]
        if verified and self.status != "provider_content_block":
            by_id = {
                item.retrieval_unit_id: item
                for item in (self.evidence.items if self.evidence else ())
                if item.verified
            }
            confirmed = [
                f"- {record.value} "
                + " ".join(f"[web:{identifier}]" for identifier in record.supported_evidence_ids)
                for record in verified
            ]
            sources = []
            seen = set()
            for record in verified:
                for identifier in record.supported_evidence_ids:
                    if identifier in seen or identifier not in by_id:
                        continue
                    seen.add(identifier)
                    item = by_id[identifier]
                    excerpt = re.sub(r"\s+", " ", item.text).strip()[:220]
                    sources.append(
                        f"- [web:{identifier}] 原文：“{excerpt}”；来源："
                        f"{item.title or '网页来源'}；{item.url or '未知 URL'}"
                    )
            source_text = "\n\n证据来源：\n" + "\n".join(sources) if sources else ""
            return (
                "已确认部分：\n" + "\n".join(confirmed)
                + f"\n\n尚缺信息：\n- {detail}" + source_text
            )
        if self.status == "generation_failure":
            return "generation_failure: Web 多跳答案生成失败或超时，未返回猜测答案。"
        if self.status == "provider_content_block":
            return json.dumps(self.partial_result(), ensure_ascii=False, separators=(",", ":"))
        return f"insufficient: {detail}"

    def partial_result(self) -> dict[str, object]:
        """Expose verified progress without synthesizing an ungrounded answer."""

        verified = [
            record
            for record in self.finding_records
            if record.verified and record.source_type == "web"
        ]
        final_items = self.final_hop_evidence.items if self.final_hop_evidence else ()
        return {
            "status": self.status,
            "completed_subquestions": [record.id for record in verified],
            "verified_findings": [record.to_dict() for record in verified],
            "final_hop_search_completed": self.final_hop_search_completed,
            "evidence": [
                {
                    "evidence_id": item.retrieval_unit_id,
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "source_tier": item.source_tier,
                    "published_at": item.published_at,
                    "retrieved_at": item.retrieved_at,
                }
                for item in final_items
                if item.verified and item.source_type is SourceType.WEB
            ],
            "message": "最终综合因 provider content policy 未完成；未使用模型内部知识猜测答案。",
        }


@dataclass(frozen=True)
class WebFinding:
    id: str
    value: str
    supported_evidence_ids: tuple[str, ...]
    source_type: str = "web"
    verified: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "value": self.value,
            "supported_evidence_ids": list(self.supported_evidence_ids),
            "source_type": self.source_type,
            "verified": self.verified,
        }


class StructuredWebResearchExecutor:
    def __init__(
        self,
        planner: WebResearchPlanner | None = None,
        step_executor: ControlledWebFallbackExecutor | None = None,
        final_evaluator: EvidenceSufficiencyEvaluator | None = None,
        hard_timeout_seconds: float = 180.0,
    ) -> None:
        self.planner = planner or WebResearchPlanner()
        self.step_executor = step_executor or ControlledWebFallbackExecutor()
        self.final_evaluator = final_evaluator or EvidenceSufficiencyEvaluator()
        if hard_timeout_seconds <= 0:
            raise ValueError("hard_timeout_seconds must be positive")
        self.hard_timeout_seconds = hard_timeout_seconds

    async def execute(
        self,
        question: str,
        source_plan: SourcePlan,
        *,
        trace_final: bool = True,
    ) -> StructuredWebResearchResult:
        if not is_structured_web_plan(question, source_plan):
            raise ValueError("StructuredWebResearchExecutor requires a structured single-source Web plan")
        started = time.perf_counter()
        hard_deadline = time.monotonic() + self.hard_timeout_seconds
        try:
            plan = self.planner.plan(question)
        except Exception as error:
            logger.debug(
                "WEB_RESEARCH_PLAN: %s",
                json.dumps({"status": "failure", "error_type": type(error).__name__}, ensure_ascii=False),
            )
            raise WebResearchPlanningError(str(error)) from error

        logger.debug("WEB_RESEARCH_PLAN: %s", json.dumps(plan.to_dict(), ensure_ascii=False))
        findings: dict[str, str] = {}
        finding_records: list[WebFinding] = []
        bundles: list[EvidenceBundle] = []
        original_normalized = _normalize(question)

        for step in plan.subquestions:
            if time.monotonic() >= hard_deadline:
                return self._finish(
                    started,
                    status="generation_failure",
                    plan=plan,
                    findings=findings,
                    finding_records=finding_records,
                    evidence=merge_evidence_bundles(*bundles) if bundles else None,
                    error_type="TimeoutError",
                    trace_final=trace_final,
                )
            query = materialize_subquestion(step, findings)
            if _normalize(query) == original_normalized:
                raise WebResearchPlanningError("structured executor refused to search the full original question")
            logger.debug(
                "WEB_SUBQUESTION: %s",
                json.dumps(
                    {
                        "id": step.id,
                        "depends_on": list(step.depends_on),
                        "query_preview": query[:_TRACE_PREVIEW_CHARS],
                        "expected_output": step.expected_output,
                    },
                    ensure_ascii=False,
                ),
            )
            is_final_step = step is plan.subquestions[-1]
            use_extractor = not is_final_step and is_atomic_expected_output(step.expected_output)
            prior_evidence = merge_evidence_bundles(*bundles) if bundles else None
            if is_final_step:
                verified_findings = tuple(record.to_dict() for record in finding_records if record.verified)
                final_constraints = (
                    "用中文一次性回答 original_question；按依赖链综合已验证的中间 Finding 与最后一步 Web evidence。"
                    "不得使用模型常识或未验证信息。supported_evidence_ids 只引用最后一步 evidence；"
                    "中间 Finding 的原始检索证据仅保留在 Trace，不会重复注入当前 prompt。"
                )
                result = await self.step_executor.execute(
                    query,
                    execution_path=f"structured_web_multi_hop:{step.id}",
                    answer_constraints=final_constraints,
                    transient_retry_limit=1,
                    hard_deadline=hard_deadline,
                    evaluation_question=question,
                    sufficiency_evaluator_override=self.final_evaluator,
                    web_evidence_prefix=step.id,
                    llm_stage=f"structured_final_hop:{step.id}",
                    synthesis_evidence_limit=6,
                    verified_findings=verified_findings,
                    web_source_policy=source_plan.web_source_policy,
                    trace_final=trace_final,
                )
                full_evidence = _merge_optional_evidence(prior_evidence, result.web_evidence)
                if result.status != "success" or result.sufficiency is None or result.sufficiency.answer is None:
                    return self._finish(
                        started,
                        status=result.status,
                        plan=plan,
                        findings=findings,
                        finding_records=finding_records,
                        evidence=full_evidence,
                        sufficiency=result.sufficiency,
                        missing_information=result.missing_information or (f"{step.id} 未解决",),
                        error_type=result.error_type,
                        final_hop_search_completed=result.web_evidence is not None,
                        final_hop_evidence=result.web_evidence,
                        trace_final=trace_final,
                    )
                final_value = result.sufficiency.answer.strip()
                findings[step.id] = final_value
                finding_records.append(
                    WebFinding(
                        id=step.id,
                        value=final_value,
                        supported_evidence_ids=result.sufficiency.supported_evidence_ids,
                    )
                )
                logger.debug(
                    "WEB_FINDING: %s",
                    json.dumps(
                        {
                            "id": step.id,
                            "finding_preview": final_value[:_TRACE_PREVIEW_CHARS],
                            "extractor": "structured_final_synthesis",
                            "verified": True,
                        },
                        ensure_ascii=False,
                    ),
                )
                answer_with_chain = _append_intermediate_citations(
                    result.answer,
                    finding_records[:-1],
                    prior_evidence,
                )
                return self._finish(
                    started,
                    status="success",
                    answer=answer_with_chain,
                    plan=plan,
                    findings=findings,
                    finding_records=finding_records,
                    evidence=full_evidence,
                    sufficiency=result.sufficiency,
                    final_hop_search_completed=True,
                    final_hop_evidence=result.web_evidence,
                    trace_final=trace_final,
                )

            result = await self.step_executor.execute(
                query,
                execution_path=f"structured_web_multi_hop:{step.id}",
                answer_constraints=(
                    "只回答当前子问题，不提前回答原始问题的后续步骤；"
                    f"返回可直接用于依赖替换的 {step.expected_output}，保持简短且不添加证据清单。"
                ),
                finding_expected_output=step.expected_output if use_extractor else None,
                transient_retry_limit=1,
                hard_deadline=hard_deadline,
                llm_stage=f"structured_intermediate_hop:{step.id}",
                web_source_policy=source_plan.web_source_policy,
                trace_final=trace_final,
            )
            if result.status != "success" or result.sufficiency is None or result.sufficiency.answer is None:
                return self._finish(
                    started,
                    status=result.status,
                    plan=plan,
                    findings=findings,
                    finding_records=finding_records,
                    evidence=merge_evidence_bundles(*bundles) if bundles else result.evidence,
                    sufficiency=result.sufficiency,
                    missing_information=result.missing_information or (f"{step.id} 未解决",),
                    error_type=result.error_type,
                    trace_final=trace_final,
                )
            finding = result.sufficiency.answer.strip()
            findings[step.id] = finding
            finding_records.append(
                WebFinding(
                    id=step.id,
                    value=finding,
                    supported_evidence_ids=tuple(
                        f"{step.id}-{identifier}" for identifier in result.sufficiency.supported_evidence_ids
                    ),
                )
            )
            if result.web_evidence is not None:
                bundles.append(_prefix_evidence(result.web_evidence, step.id))
            logger.debug(
                "WEB_FINDING: %s",
                json.dumps(
                    {
                        "id": step.id,
                        "finding_preview": finding[:_TRACE_PREVIEW_CHARS],
                        "extractor": "atomic" if use_extractor else "full_synthesis",
                        "verified": True,
                    },
                    ensure_ascii=False,
                ),
            )

        raise WebResearchPlanningError("structured plan has no final hop")

    @staticmethod
    def _finish(
        started: float,
        *,
        status: str,
        plan: WebResearchPlan | None,
        findings: dict[str, str],
        finding_records: list[WebFinding] | None = None,
        answer: str | None = None,
        evidence: EvidenceBundle | None = None,
        sufficiency: SufficiencyResult | None = None,
        missing_information: tuple[str, ...] = (),
        error_type: str | None = None,
        final_hop_search_completed: bool = False,
        final_hop_evidence: EvidenceBundle | None = None,
        trace_final: bool = True,
    ) -> StructuredWebResearchResult:
        result = StructuredWebResearchResult(
            status=status,
            answer=answer,
            plan=plan,
            findings=tuple(findings.items()),
            evidence=evidence,
            sufficiency=sufficiency,
            missing_information=tuple(missing_information),
            error_type=error_type,
            latency_ms=round((time.perf_counter() - started) * 1000),
            finding_records=tuple(finding_records or ()),
            final_hop_search_completed=final_hop_search_completed,
            final_hop_evidence=final_hop_evidence,
        )
        if trace_final:
            logger.debug(
                "SOURCE_COMPLETED: %s",
                json.dumps(
                    {
                        "source": "web",
                        "execution_path": "structured_web_multi_hop",
                        "status": status,
                        "completed_subquestions": len(findings),
                        "trace_evidence_count": len(evidence.items) if evidence else 0,
                        "final_synthesis_evidence_count": (
                            len(final_hop_evidence.items) if final_hop_evidence else 0
                        ),
                        "latency_ms": result.latency_ms,
                    },
                    ensure_ascii=False,
                ),
            )
            logger.debug(
                "FINAL: %s",
                json.dumps(
                    {"status": status, "answer_preview": (answer or "")[:_TRACE_PREVIEW_CHARS]},
                    ensure_ascii=False,
                ),
            )
        return result


def is_structured_web_plan(question: str, plan: SourcePlan) -> bool:
    return (
        plan.mode is PlanMode.SINGLE_SOURCE
        and plan.primary_source is SourceType.WEB
        and plan.required_sources == (SourceType.WEB,)
        and classify_web_task(question) is WebTaskComplexity.STRUCTURED_RESEARCH
    )


def _prefix_evidence(bundle: EvidenceBundle, prefix: str) -> EvidenceBundle:
    items = tuple(
        replace(item, retrieval_unit_id=f"{prefix}-{item.retrieval_unit_id}")
        for item in bundle.items
    )
    return replace(bundle, items=items)


def _merge_optional_evidence(
    prior_evidence: EvidenceBundle | None,
    final_evidence: EvidenceBundle | None,
) -> EvidenceBundle | None:
    if prior_evidence is not None and final_evidence is not None:
        return merge_evidence_bundles(prior_evidence, final_evidence)
    return prior_evidence or final_evidence


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold().rstrip("?？。！!")


def _append_intermediate_citations(
    answer: str | None,
    findings: list[WebFinding],
    prior_evidence: EvidenceBundle | None,
) -> str | None:
    """Keep every dependency hop auditable without reinjecting bulky old pages."""
    if not answer or prior_evidence is None:
        return answer
    by_id = {item.retrieval_unit_id: item for item in prior_evidence.items}
    lines: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        for identifier in finding.supported_evidence_ids:
            if identifier in seen or identifier not in by_id:
                continue
            seen.add(identifier)
            item = by_id[identifier]
            excerpt = re.sub(r"\s+", " ", item.text).strip()[:220]
            date = f"；发布日期={item.published_at}" if item.published_at else ""
            lines.append(
                f"- {finding.id} 结论“{finding.value}” → [web:{identifier}] "
                f"原文：“{excerpt}”；来源：{item.title or '网页来源'}；{item.url or '未知 URL'}{date}"
            )
    if not lines:
        return answer
    return answer.rstrip() + "\n\n中间链路证据：\n" + "\n".join(lines)
