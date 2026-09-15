"""Bounded Web evidence execution for direct facts and RAG fallback."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from config import TRACE_INCLUDE_CONTENT

from .evidence import EvidenceBundle, merge_evidence_bundles, normalize_web_tool_result
from .models import PlanMode, SourcePlan, SourceType
from .web_source_policy import WebSourcePolicy
from .sufficiency import EvidenceSufficiencyEvaluator, SufficiencyResult
from .web_planner import WebTaskComplexity, classify_web_task
from .finding_extractor import FindingExtractor
from .llm_errors import is_retryable_llm_error
from .trace import record_trace_event
from .web_list_navigation import (
    date_tokens,
    fetch_raw_html,
    is_listing_page,
    pagination_urls,
    target_entries,
)
from page_fetcher import format_relevant_content


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


def _search_trace_output(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    projected = [
        {
            "title": str(item.get("title") or "")[:500],
            "url": item.get("url"),
            "snippet": str(item.get("snippet") or "")[:1200],
            "source_type": item.get("source_type"),
        }
        for item in results[:20]
        if isinstance(item, Mapping)
    ]
    return {"results": projected, "result_count": len(results), "truncated": len(results) > len(projected)}


@dataclass(frozen=True)
class WebFallbackExecutionResult:
    status: str
    answer: str | None
    evidence: EvidenceBundle | None
    web_evidence: EvidenceBundle | None
    sufficiency: SufficiencyResult | None
    missing_information: tuple[str, ...] = ()
    error_type: str | None = None
    latency_ms: int = 0
    execution_path: str = "controlled_web_fallback"

    def public_answer(self) -> str:
        if self.status == "success" and self.answer:
            return self.answer
        if self.status == "generation_failure":
            label = "Web 事实证据" if self.execution_path == "simple_web_factual" else "Web 补充证据"
            return f"generation_failure: {label}答案生成失败或超时，未返回猜测答案。"
        if self.status == "provider_content_block":
            return "provider_content_block: Web 检索已完成，但提供方内容策略阻止了答案综合。"
        detail = "；".join(self.missing_information) or "Web 补充证据仍不足"
        if self.execution_path == "simple_web_factual":
            return f"insufficient: Web 事实证据不足。缺失信息：{detail}"
        if self.execution_path == "sql_web_fallback":
            return f"insufficient: SQL 查询不足后 Web 补充仍证据不足。缺失信息：{detail}"
        return f"insufficient: 两次本地文档检索及受限 Web 补充后仍证据不足。缺失信息：{detail}"


class ControlledWebFallbackExecutor:
    """Execute a finite search/fetch path with no RAG or SQL action surface."""

    def __init__(
        self,
        *,
        search_call: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]] | None = None,
        fetch_call: Callable[[str], Awaitable[str]] | None = None,
        sufficiency_evaluator: EvidenceSufficiencyEvaluator | None = None,
        max_results: int = 8,
        max_fetches: int = 2,
        max_search_rounds: int = 2,
        max_list_pages: int = 4,
        finding_extractor: FindingExtractor | None = None,
    ) -> None:
        if max_results <= 0 or max_fetches < 0 or max_search_rounds <= 0 or max_list_pages < 0:
            raise ValueError("Web fallback budgets must be non-negative and max_results positive")
        self._search_call = search_call
        self._fetch_call = fetch_call
        self.sufficiency_evaluator = sufficiency_evaluator or EvidenceSufficiencyEvaluator()
        self.max_results = max_results
        self.max_fetches = max_fetches
        self.max_search_rounds = max_search_rounds
        self.max_list_pages = max_list_pages
        self.finding_extractor = finding_extractor or FindingExtractor(
            getattr(self.sufficiency_evaluator, "_llm_call", None)
        )

    async def execute(
        self,
        question: str,
        *,
        rag_evidence: EvidenceBundle | None = None,
        prior_evidence: EvidenceBundle | None = None,
        missing_information: tuple[str, ...] = (),
        execution_path: str = "controlled_web_fallback",
        answer_constraints: str | None = None,
        finding_expected_output: str | None = None,
        transient_retry_limit: int = 0,
        hard_deadline: float | None = None,
        evaluation_question: str | None = None,
        sufficiency_evaluator_override: EvidenceSufficiencyEvaluator | None = None,
        web_evidence_prefix: str | None = None,
        llm_stage: str = "web_evidence_sufficiency",
        synthesis_evidence_limit: int | None = None,
        verified_findings: Sequence[Mapping[str, Any]] = (),
        trace_final: bool = True,
        web_source_policy: WebSourcePolicy | None = None,
    ) -> WebFallbackExecutionResult:
        web_source_policy = web_source_policy or WebSourcePolicy()
        if not web_source_policy.web_permitted:
            return self._finish(
                started=time.perf_counter(), status="insufficient", rag_evidence=rag_evidence,
                missing_information=("该任务禁止使用 Web 来源",), execution_path=execution_path,
                trace_final=trace_final,
            )
        if prior_evidence is not None:
            if rag_evidence is not None:
                raise ValueError("provide either rag_evidence or prior_evidence, not both")
            # The retained parameter name is a backwards-compatible detail of
            # the original RAG fallback API.  The bounded executor itself is
            # source-agnostic and can evaluate prior SQL provenance as well.
            rag_evidence = prior_evidence
        started = time.perf_counter()
        search = self._search_call or _default_search
        fetch = self._fetch_call or _default_fetch
        search_query = _apply_source_policy_to_query(
            _build_web_query(question, missing_information), web_source_policy
        )
        try:
            logger.debug(
                "TOOL_CALL: %s",
                json.dumps(
                    {
                        "tool_name": "search",
                        "execution_path": execution_path,
                        "query_preview": search_query[:_TRACE_PREVIEW_CHARS],
                    },
                    ensure_ascii=False,
                ),
            )
            search_started = time.perf_counter()
            record_trace_event(
                event_type="tool.call.started",
                stage="tool_call",
                status="started",
                source="web",
                attempt=1,
                attributes={
                    "tool_name": "search",
                    "timeout": None,
                    "retry_count": 0,
                    "query_count": 1,
                    **_trace_io(
                        input_value={
                            "query": search_query,
                            "max_results": self.max_results,
                            "execution_path": execution_path,
                        }
                    ),
                },
            )
            try:
                search_results = await search(search_query, num=self.max_results)
            except Exception as error:
                record_trace_event(
                    event_type="tool.call.failed",
                    stage="tool_call",
                    status="failed",
                    source="web",
                    attempt=1,
                    duration_ms=round((time.perf_counter() - search_started) * 1000),
                    error_type=type(error).__name__,
                    attributes={
                        "tool_name": "search",
                        "timeout": None,
                        "retry_count": 0,
                        "query_count": 1,
                        "result_count": None,
                        **_trace_io(
                            input_value={"query": search_query, "max_results": self.max_results},
                            output_value={
                                "error_type": type(error).__name__,
                                "error": str(error)[:2000],
                            },
                        ),
                    },
                )
                raise
            if not isinstance(search_results, Sequence) or isinstance(search_results, (str, bytes)):
                record_trace_event(
                    event_type="tool.call.failed",
                    stage="tool_call",
                    status="failed",
                    source="web",
                    attempt=1,
                    duration_ms=round((time.perf_counter() - search_started) * 1000),
                    error_type="ValueError",
                    attributes={
                        "tool_name": "search",
                        "timeout": None,
                        "retry_count": 0,
                        "query_count": 1,
                        "result_count": None,
                    },
                )
                raise ValueError("search must return a sequence")
            unfiltered_result_count = len(search_results)
            search_results = [
                result for result in search_results
                if isinstance(result, Mapping) and web_source_policy.permits_url(str(result.get("url", "")))
            ]
            record_trace_event(
                event_type="tool.call.completed",
                stage="tool_call",
                status="completed",
                source="web",
                attempt=1,
                duration_ms=round((time.perf_counter() - search_started) * 1000),
                attributes={
                    "tool_name": "search",
                    "timeout": None,
                    "retry_count": 0,
                    "query_count": 1,
                    "result_count": len(search_results),
                    **_trace_io(
                        input_value={
                            "query": search_query,
                            "max_results": self.max_results,
                            "execution_path": execution_path,
                        },
                        output_value=_search_trace_output(search_results),
                    ),
                },
            )
            logger.debug(
                "TOOL_RESULT: %s",
                json.dumps(
                    {
                        "tool_name": "search",
                        "result_count": len(search_results),
                        "filtered_result_count": unfiltered_result_count - len(search_results),
                        "latency_ms": round((time.perf_counter() - search_started) * 1000),
                    },
                    ensure_ascii=False,
                ),
            )

            # Search snippets are evidence in their own right.  Judge them before
            # paying the latency/cost of fetching full pages.
            web_evidence = normalize_web_tool_result(search_results, {})
            if web_evidence_prefix:
                web_evidence = _prefix_web_evidence(web_evidence, web_evidence_prefix)
            if synthesis_evidence_limit is not None:
                web_evidence = select_web_evidence(
                    web_evidence,
                    max_items=synthesis_evidence_limit,
                )
        except Exception as error:
            return self._finish(
                started,
                status="insufficient",
                rag_evidence=rag_evidence,
                missing_information=missing_information or ("Web 补充检索失败",),
                error_type=type(error).__name__,
                execution_path=execution_path,
                trace_final=trace_final,
            )

        if not web_evidence.items:
            return self._finish(
                started,
                status="insufficient",
                rag_evidence=rag_evidence,
                web_evidence=web_evidence,
                missing_information=missing_information or (
                    (
                        "没有符合用户来源限制的 Web 证据"
                        if web_source_policy.mode.value == "allowlist" else "Web 未返回可用证据"
                    ),
                ),
                execution_path=execution_path,
                trace_final=trace_final,
            )

        combined, sufficiency = self._evaluate_with_retry(
            evaluation_question or question,
            rag_evidence,
            web_evidence,
            answer_constraints=answer_constraints,
            finding_expected_output=finding_expected_output,
            transient_retry_limit=transient_retry_limit,
            hard_deadline=hard_deadline,
            sufficiency_evaluator_override=sufficiency_evaluator_override,
            llm_stage=llm_stage,
            verified_findings=verified_findings,
        )
        if sufficiency.generation_failure:
            return self._finish(
                started,
                status=_failure_status(sufficiency),
                rag_evidence=rag_evidence,
                web_evidence=web_evidence,
                evidence=combined,
                sufficiency=sufficiency,
                error_type=sufficiency.error_type,
                execution_path=execution_path,
                trace_final=trace_final,
            )
        if sufficiency.sufficient:
            return self._finish_success(
                started,
                rag_evidence=rag_evidence,
                web_evidence=web_evidence,
                combined=combined,
                sufficiency=sufficiency,
                execution_path=execution_path,
                finding_mode=finding_expected_output is not None,
                trace_final=trace_final,
            )

        fetched = {}
        if self.max_fetches > 0:
            fetched = await self._fetch_insufficient_results(
                search_results,
                fetch,
                query=search_query,
                execution_path=execution_path,
            )
        if fetched:
            web_evidence = normalize_web_tool_result(search_results, fetched)
            if web_evidence_prefix:
                web_evidence = _prefix_web_evidence(web_evidence, web_evidence_prefix)
            if synthesis_evidence_limit is not None:
                web_evidence = select_web_evidence(
                    web_evidence,
                    max_items=synthesis_evidence_limit,
                )
            combined, sufficiency = self._evaluate_with_retry(
                evaluation_question or question,
                rag_evidence,
                web_evidence,
                answer_constraints=answer_constraints,
                finding_expected_output=finding_expected_output,
                transient_retry_limit=transient_retry_limit,
                hard_deadline=hard_deadline,
                sufficiency_evaluator_override=sufficiency_evaluator_override,
                llm_stage=llm_stage,
                verified_findings=verified_findings,
            )
        if sufficiency.generation_failure:
            return self._finish(
                started,
                status=_failure_status(sufficiency),
                rag_evidence=rag_evidence,
                web_evidence=web_evidence,
                evidence=combined,
                sufficiency=sufficiency,
                error_type=sufficiency.error_type,
                execution_path=execution_path,
                trace_final=trace_final,
            )
        if not sufficiency.sufficient:
            # A notice on an official listing page may sit on a later numbered
            # page and never surface in a domain-restricted search. Follow the
            # listing's own pagination once, bounded, before rewriting the query.
            discovered = await self._navigate_official_listings(
                evaluation_question or question,
                fetched,
                fetch,
                web_source_policy,
                execution_path=execution_path,
                limit=max(0, self.max_fetches - len(fetched)),
            )
            if discovered["results"]:
                search_results = [*search_results, *discovered["results"]]
                fetched.update(discovered["fetched"])
                web_evidence = normalize_web_tool_result(search_results, fetched)
                if web_evidence_prefix:
                    web_evidence = _prefix_web_evidence(web_evidence, web_evidence_prefix)
                if synthesis_evidence_limit is not None:
                    web_evidence = select_web_evidence(
                        web_evidence, max_items=synthesis_evidence_limit
                    )
                combined, sufficiency = self._evaluate_with_retry(
                    evaluation_question or question,
                    rag_evidence,
                    web_evidence,
                    answer_constraints=answer_constraints,
                    finding_expected_output=finding_expected_output,
                    transient_retry_limit=transient_retry_limit,
                    hard_deadline=hard_deadline,
                    sufficiency_evaluator_override=sufficiency_evaluator_override,
                    llm_stage=llm_stage,
                    verified_findings=verified_findings,
                )
                if sufficiency.generation_failure:
                    return self._finish(
                        started,
                        status=_failure_status(sufficiency),
                        rag_evidence=rag_evidence,
                        web_evidence=web_evidence,
                        evidence=combined,
                        sufficiency=sufficiency,
                        error_type=sufficiency.error_type,
                        execution_path=execution_path,
                        trace_final=trace_final,
                    )
                if sufficiency.sufficient:
                    return self._finish_success(
                        started,
                        rag_evidence=rag_evidence,
                        web_evidence=web_evidence,
                        combined=combined,
                        sufficiency=sufficiency,
                        execution_path=execution_path,
                        finding_mode=finding_expected_output is not None,
                        trace_final=trace_final,
                    )
        if not sufficiency.sufficient:
            # A missing-information verdict is actionable. Run one bounded,
            # targeted query rewrite instead of terminating after the first
            # mechanical search/fetch sequence.
            if self.max_search_rounds > 1 and sufficiency.missing_information:
                retry_query = _apply_source_policy_to_query(
                    _build_web_query(question, sufficiency.missing_information), web_source_policy
                )
                if _normalize_query(retry_query) != _normalize_query(search_query):
                    try:
                        retry_results = await search(retry_query, num=self.max_results)
                    except Exception:
                        retry_results = ()
                    if isinstance(retry_results, Sequence) and not isinstance(retry_results, (str, bytes)):
                        retry_results = [
                            result for result in retry_results
                            if isinstance(result, Mapping)
                            and web_source_policy.permits_url(str(result.get("url", "")))
                        ]
                        combined_results = _merge_search_results(search_results, retry_results)
                        retry_fetched = (
                            await self._fetch_insufficient_results(
                                retry_results,
                                fetch,
                                query=retry_query,
                                execution_path=execution_path,
                                limit=max(0, self.max_fetches - len(fetched)),
                            )
                            if self.max_fetches > len(fetched)
                            else {}
                        )
                        fetched.update(retry_fetched)
                        web_evidence = normalize_web_tool_result(combined_results, fetched)
                        if web_evidence_prefix:
                            web_evidence = _prefix_web_evidence(web_evidence, web_evidence_prefix)
                        if synthesis_evidence_limit is not None:
                            web_evidence = select_web_evidence(
                                web_evidence, max_items=synthesis_evidence_limit
                            )
                        combined, sufficiency = self._evaluate_with_retry(
                            evaluation_question or question,
                            rag_evidence,
                            web_evidence,
                            answer_constraints=answer_constraints,
                            finding_expected_output=finding_expected_output,
                            transient_retry_limit=transient_retry_limit,
                            hard_deadline=hard_deadline,
                            sufficiency_evaluator_override=sufficiency_evaluator_override,
                            llm_stage=llm_stage,
                            verified_findings=verified_findings,
                        )
                        if sufficiency.sufficient:
                            return self._finish_success(
                                started,
                                rag_evidence=rag_evidence,
                                web_evidence=web_evidence,
                                combined=combined,
                                sufficiency=sufficiency,
                                execution_path=execution_path,
                                finding_mode=finding_expected_output is not None,
                                trace_final=trace_final,
                            )
            return self._finish(
                started,
                status="insufficient",
                rag_evidence=rag_evidence,
                web_evidence=web_evidence,
                evidence=combined,
                sufficiency=sufficiency,
                missing_information=sufficiency.missing_information,
                execution_path=execution_path,
                trace_final=trace_final,
            )
        return self._finish_success(
            started,
            rag_evidence=rag_evidence,
            web_evidence=web_evidence,
            combined=combined,
            sufficiency=sufficiency,
            execution_path=execution_path,
            finding_mode=finding_expected_output is not None,
            trace_final=trace_final,
        )

    def _evaluate(
        self,
        question: str,
        rag_evidence: EvidenceBundle | None,
        web_evidence: EvidenceBundle,
        *,
        answer_constraints: str | None,
        finding_expected_output: str | None = None,
        timeout_override: float | None = None,
        sufficiency_evaluator_override: EvidenceSufficiencyEvaluator | None = None,
        llm_stage: str = "web_evidence_sufficiency",
        verified_findings: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[EvidenceBundle, SufficiencyResult]:
        combined = (
            merge_evidence_bundles(rag_evidence, web_evidence)
            if rag_evidence is not None
            else web_evidence
        )
        if finding_expected_output is not None:
            sufficiency = self.finding_extractor.evaluate(
                question,
                combined,
                expected_output=finding_expected_output,
                timeout_override=timeout_override,
                stage=llm_stage,
            )
        else:
            evaluator = sufficiency_evaluator_override or self.sufficiency_evaluator
            sufficiency = evaluator.evaluate(
                question,
                combined,
                answer_constraints=answer_constraints or (
                    "用中文简洁回答；仅引用给定证据；必须将 local_document 与 web 明确区分，"
                    "不得把 Web 内容描述为本地招股说明书或本地文档披露。"
                ),
                timeout_override=timeout_override,
                stage=llm_stage,
                verified_findings=verified_findings,
            )
        return combined, sufficiency

    def _evaluate_with_retry(
        self,
        question: str,
        rag_evidence: EvidenceBundle | None,
        web_evidence: EvidenceBundle,
        *,
        answer_constraints: str | None,
        finding_expected_output: str | None,
        transient_retry_limit: int,
        hard_deadline: float | None,
        sufficiency_evaluator_override: EvidenceSufficiencyEvaluator | None,
        llm_stage: str,
        verified_findings: Sequence[Mapping[str, Any]],
    ) -> tuple[EvidenceBundle, SufficiencyResult]:
        retries = 0
        while True:
            remaining = hard_deadline - time.monotonic() if hard_deadline is not None else None
            combined, result = self._evaluate(
                question,
                rag_evidence,
                web_evidence,
                answer_constraints=answer_constraints,
                finding_expected_output=finding_expected_output,
                timeout_override=remaining,
                sufficiency_evaluator_override=sufficiency_evaluator_override,
                llm_stage=llm_stage,
                verified_findings=verified_findings,
            )
            if not (
                result.generation_failure
                and is_retryable_llm_error(
                    result.error_type,
                    result.status_code,
                    provider_error_code=result.provider_error_code,
                )
                and retries < transient_retry_limit
                and (hard_deadline is None or time.monotonic() < hard_deadline)
            ):
                return combined, result
            retries += 1
            logger.debug(
                "WEB_LLM_RETRY: %s",
                json.dumps(
                    {
                        "stage": llm_stage,
                        "attempt": retries + 1,
                        "error_type": result.error_type,
                        "status_code": result.status_code,
                        "provider_error_code": result.provider_error_code,
                        "search_reused": True,
                    },
                    ensure_ascii=False,
                ),
            )

    async def _navigate_official_listings(
        self,
        question: str,
        fetched: Mapping[str, str],
        fetch: Callable[[str], Awaitable[str]],
        web_source_policy: WebSourcePolicy,
        *,
        execution_path: str,
        limit: int,
    ) -> dict[str, Any]:
        """Open the numbered pages of an official listing to find one notice.

        Only already-fetched pages are treated as listings, only sources that
        survive the active source policy are followed, and the number of extra
        detail fetches stays inside the caller's remaining fetch budget.
        """

        empty: dict[str, Any] = {"results": [], "fetched": {}}
        if limit <= 0 or self.max_list_pages <= 0:
            return empty
        tokens = date_tokens(question)
        if not tokens:
            return empty
        listing_pages = [
            url
            for url in fetched
            if is_listing_page(url) and _may_traverse_listing(url, web_source_policy)
        ]
        if not listing_pages:
            return empty

        results: list[dict[str, str]] = []
        extra_fetched: dict[str, str] = {}
        visited = set(fetched)
        for page_url in listing_pages[:2]:
            markup = await fetch_raw_html(page_url)
            if not markup:
                continue
            listing_markup = [(page_url, markup)]
            for candidate in pagination_urls(
                markup,
                page_url,
                limit=self.max_list_pages,
                exclude=visited,
            ):
                if not web_source_policy.permits_url(candidate):
                    continue
                visited.add(candidate)
                candidate_markup = await fetch_raw_html(candidate)
                if candidate_markup:
                    listing_markup.append((candidate, candidate_markup))
            for source_url, source_markup in listing_markup:
                remaining = limit - len(results)
                if remaining <= 0:
                    break
                for entry in target_entries(source_markup, source_url, tokens, limit=remaining):
                    if entry.url in visited or not web_source_policy.permits_url(entry.url):
                        continue
                    visited.add(entry.url)
                    fetch_started = time.perf_counter()
                    record_trace_event(
                        event_type="tool.call.started",
                        stage="tool_call",
                        status="started",
                        source="web",
                        attempt=1,
                        attributes={
                            "tool_name": "fetch",
                            "timeout": None,
                            "retry_count": 0,
                            "fetch_count": 1,
                            **_trace_io(
                                input_value={
                                    "url": entry.url,
                                    "reason": "official_listing_navigation",
                                    "execution_path": execution_path,
                                }
                            ),
                        },
                    )
                    try:
                        content = await fetch(entry.url)
                    except Exception as error:
                        record_trace_event(
                            event_type="tool.call.failed",
                            stage="tool_call",
                            status="failed",
                            source="web",
                            attempt=1,
                            duration_ms=round((time.perf_counter() - fetch_started) * 1000),
                            error_type=type(error).__name__,
                            attributes={"tool_name": "fetch", "retry_count": 0, "fetch_count": 1},
                        )
                        continue
                    record_trace_event(
                        event_type="tool.call.completed",
                        stage="tool_call",
                        status="completed",
                        source="web",
                        attempt=1,
                        duration_ms=round((time.perf_counter() - fetch_started) * 1000),
                        attributes={
                            "tool_name": "fetch",
                            "timeout": None,
                            "retry_count": 0,
                            "fetch_count": 1,
                            "result_count": 1,
                            "content_char_count": len(content or ""),
                            **_trace_io(
                                input_value={"url": entry.url},
                                output_value={"content": (content or "")[:2000]},
                            ),
                        },
                    )
                    results.append(entry.as_search_result())
                    if content:
                        extra_fetched[entry.url] = content
            if len(results) >= limit:
                break
        return {"results": results, "fetched": extra_fetched}

    async def _fetch_insufficient_results(
        self,
        search_results: Sequence[Mapping[str, Any]],
        fetch: Callable[[str], Awaitable[str]],
        *,
        query: str,
        execution_path: str,
        limit: int | None = None,
    ) -> dict[str, str]:
        fetched: dict[str, str] = {}
        fetch_limit = self.max_fetches if limit is None else max(0, limit)
        for raw in search_results[:fetch_limit]:
            if not isinstance(raw, Mapping):
                continue
            url = raw.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            logger.debug(
                "TOOL_CALL: %s",
                json.dumps(
                    {
                        "tool_name": "fetch",
                        "execution_path": execution_path,
                        "reason": "snippet_evidence_insufficient",
                        "url_preview": url[:_TRACE_PREVIEW_CHARS],
                    },
                    ensure_ascii=False,
                ),
            )
            fetch_started = time.perf_counter()
            record_trace_event(
                event_type="tool.call.started",
                stage="tool_call",
                status="started",
                source="web",
                attempt=1,
                attributes={
                    "tool_name": "fetch",
                    "timeout": None,
                    "retry_count": 0,
                    "fetch_count": 1,
                    **_trace_io(
                        input_value={
                            "url": url,
                            "reason": "snippet_evidence_insufficient",
                            "execution_path": execution_path,
                        }
                    ),
                },
            )
            try:
                content = await fetch(url)
            except Exception as error:
                record_trace_event(
                    event_type="tool.call.failed",
                    stage="tool_call",
                    status="failed",
                    source="web",
                    attempt=1,
                    duration_ms=round((time.perf_counter() - fetch_started) * 1000),
                    error_type=type(error).__name__,
                    attributes={
                        "tool_name": "fetch",
                        "timeout": None,
                        "retry_count": 0,
                        "fetch_count": 1,
                        "result_count": None,
                        **_trace_io(
                            input_value={"url": url},
                            output_value={
                                "error_type": type(error).__name__,
                                "error": str(error)[:2000],
                            },
                        ),
                    },
                )
                logger.debug(
                    "TOOL_RESULT: %s",
                    json.dumps(
                        {
                            "tool_name": "fetch",
                            "success": False,
                            "error_type": type(error).__name__,
                            "latency_ms": round((time.perf_counter() - fetch_started) * 1000),
                        },
                        ensure_ascii=False,
                    ),
                )
                continue
            if isinstance(content, str) and content.strip():
                fetched[url] = format_relevant_content(content, query)
            record_trace_event(
                event_type="tool.call.completed",
                stage="tool_call",
                status="completed",
                source="web",
                attempt=1,
                duration_ms=round((time.perf_counter() - fetch_started) * 1000),
                attributes={
                    "tool_name": "fetch",
                    "timeout": None,
                    "retry_count": 0,
                    "fetch_count": 1,
                    "result_count": 1 if isinstance(content, str) and content.strip() else 0,
                    "content_char_count": len(content) if isinstance(content, str) else None,
                    **_trace_io(
                        input_value={"url": url},
                        output_value={
                            "content": content[:6000] if isinstance(content, str) else content,
                            "truncated": isinstance(content, str) and len(content) > 6000,
                        },
                    ),
                },
            )
            logger.debug(
                "TOOL_RESULT: %s",
                json.dumps(
                    {
                        "tool_name": "fetch",
                        "success": bool(content),
                        "content_preview": content[:_TRACE_PREVIEW_CHARS] if isinstance(content, str) else "",
                        "latency_ms": round((time.perf_counter() - fetch_started) * 1000),
                    },
                    ensure_ascii=False,
                ),
            )
        return fetched

    def _finish_success(
        self,
        started: float,
        *,
        rag_evidence: EvidenceBundle | None,
        web_evidence: EvidenceBundle,
        combined: EvidenceBundle,
        sufficiency: SufficiencyResult,
        execution_path: str,
        finding_mode: bool = False,
        trace_final: bool = True,
    ) -> WebFallbackExecutionResult:
        assert sufficiency.answer is not None
        answer = _answer_with_mixed_provenance(
            sufficiency.answer,
            sufficiency.supported_evidence_ids,
            combined,
            claims=sufficiency.claims,
        )
        logger.debug(
            "%s: %s",
            "FINDING_EXTRACTION" if finding_mode else "SYNTHESIS",
            json.dumps(
                {
                    "status": "success",
                    "source_types": sorted({item.source_type.value for item in combined.items}),
                    "supported_evidence_ids": list(sufficiency.supported_evidence_ids),
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
            rag_evidence=rag_evidence,
            web_evidence=web_evidence,
            evidence=combined,
            sufficiency=sufficiency,
            execution_path=execution_path,
            trace_final=trace_final,
        )

    @staticmethod
    def _finish(
        started: float,
        *,
        status: str,
        rag_evidence: EvidenceBundle | None,
        answer: str | None = None,
        web_evidence: EvidenceBundle | None = None,
        evidence: EvidenceBundle | None = None,
        sufficiency: SufficiencyResult | None = None,
        missing_information: tuple[str, ...] = (),
        error_type: str | None = None,
        execution_path: str = "controlled_web_fallback",
        trace_final: bool = True,
    ) -> WebFallbackExecutionResult:
        result = WebFallbackExecutionResult(
            status=status,
            answer=answer,
            evidence=evidence or rag_evidence or web_evidence,
            web_evidence=web_evidence,
            sufficiency=sufficiency,
            missing_information=tuple(missing_information),
            error_type=error_type,
            latency_ms=round((time.perf_counter() - started) * 1000),
            execution_path=execution_path,
        )
        if trace_final:
            logger.debug(
                "SOURCE_COMPLETED: %s",
                json.dumps(
                    {
                        "source": "web",
                        "execution_path": execution_path,
                        "status": result.status,
                        "evidence_count": len(result.evidence.items) if result.evidence else 0,
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


class DirectWebExecutor:
    """Direct bounded path for one-hop factual Web questions."""

    def __init__(self, executor: ControlledWebFallbackExecutor | None = None) -> None:
        self.executor = executor or ControlledWebFallbackExecutor()

    async def execute(
        self,
        question: str,
        plan: SourcePlan,
        *,
        trace_final: bool = True,
    ) -> WebFallbackExecutionResult:
        if not is_direct_web_plan(question, plan):
            raise ValueError("DirectWebExecutor requires a simple single-source Web factual plan")
        return await self.executor.execute(
            question,
            execution_path="simple_web_factual",
            trace_final=trace_final,
            web_source_policy=plan.web_source_policy,
        )

    async def execute_for_source(
        self,
        question: str,
        plan: SourcePlan,
        *,
        trace_final: bool = False,
    ) -> WebFallbackExecutionResult:
        """Run the existing bounded Web executor inside an explicit hybrid plan."""

        if not (
            plan.mode is PlanMode.SINGLE_SOURCE
            and plan.primary_source is SourceType.WEB
            and plan.required_sources == (SourceType.WEB,)
        ):
            raise ValueError("hybrid Web projection must be a single Web source plan")
        return await self.executor.execute(
            question,
            execution_path="hybrid_web",
            trace_final=trace_final,
            web_source_policy=plan.web_source_policy,
        )


def is_direct_web_plan(question: str, plan: SourcePlan) -> bool:
    return (
        plan.mode is PlanMode.SINGLE_SOURCE
        and plan.primary_source is SourceType.WEB
        and plan.required_sources == (SourceType.WEB,)
        and is_simple_web_factual_query(question)
    )


def is_simple_web_factual_query(question: str) -> bool:
    """Return true only for deterministic one-hop factual Web questions."""

    return classify_web_task(question) is WebTaskComplexity.SIMPLE_FACTUAL


def _is_transient_llm_error(error_type: str | None, status_code: int | None = None) -> bool:
    """Keep transport/API retries distinct from parse and insufficiency outcomes."""

    return is_retryable_llm_error(error_type, status_code)


def _failure_status(result: SufficiencyResult) -> str:
    return "provider_content_block" if result.failure_kind == "provider_content_block" else "generation_failure"


def select_web_evidence(bundle: EvidenceBundle, *, max_items: int = 6) -> EvidenceBundle:
    """Select bounded, diverse Web evidence deterministically without an LLM call."""

    if max_items <= 0:
        raise ValueError("max_items must be positive")

    unique: list[tuple[int, Any, str]] = []
    seen_urls: set[str] = set()
    seen_texts: list[str] = []
    for rank, item in enumerate(bundle.items):
        canonical_url = _canonical_url(item.url or "")
        if canonical_url and canonical_url in seen_urls:
            continue
        normalized_text = _normalized_snippet(item.snippet or item.text)
        if normalized_text and any(_highly_repetitive(normalized_text, prior) for prior in seen_texts):
            continue
        if canonical_url:
            seen_urls.add(canonical_url)
        if normalized_text:
            seen_texts.append(normalized_text)
        unique.append((rank, item, _url_host(item.url or "")))

    # First keep the best-ranked result per source, then fill remaining slots in
    # rank order.  Sorting the chosen rows restores the provider's original rank.
    selected: list[tuple[int, Any, str]] = []
    selected_ranks: set[int] = set()
    seen_hosts: set[str] = set()
    for row in unique:
        host = row[2]
        if host and host in seen_hosts:
            continue
        selected.append(row)
        selected_ranks.add(row[0])
        if host:
            seen_hosts.add(host)
        if len(selected) >= max_items:
            break
    if len(selected) < max_items:
        for row in unique:
            if row[0] in selected_ranks:
                continue
            selected.append(row)
            selected_ranks.add(row[0])
            if len(selected) >= max_items:
                break
    selected.sort(key=lambda row: row[0])

    raw = dict(bundle.raw_tool_result)
    raw["synthesis_selection"] = {
        "input_count": len(bundle.items),
        "selected_count": len(selected),
        "max_items": max_items,
        "selected_evidence_ids": [row[1].retrieval_unit_id for row in selected],
    }
    logger.debug(
        "WEB_EVIDENCE_SELECTION: %s",
        json.dumps(raw["synthesis_selection"], ensure_ascii=False),
    )
    return replace(
        bundle,
        items=tuple(row[1] for row in selected),
        raw_tool_result=raw,
        execution_success=bool(selected),
    )


def _canonical_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return url.strip().casefold()
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
        )
    )
    return urlunsplit(
        (
            parts.scheme.casefold(),
            parts.netloc.casefold(),
            parts.path.rstrip("/") or "/",
            query,
            "",
        )
    )


def _url_host(url: str) -> str:
    try:
        return urlsplit(url).netloc.casefold().removeprefix("www.")
    except ValueError:
        return ""


def _may_traverse_listing(url: str, policy: WebSourcePolicy) -> bool:
    """Only open listing pagination on an allowlisted or official host."""

    host = _url_host(url)
    if not host:
        return False
    if policy.allowed_domains:
        return policy.permits_url(url)
    return host.endswith((".gov.cn", ".gov", ".edu.cn", ".org.cn"))


def _normalized_snippet(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value, flags=re.UNICODE).casefold()


def _highly_repetitive(left: str, right: str) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    # Short snippets routinely share boilerplate words; fuzzy-deduplicating them
    # would discard distinct facts. Exact duplicates above remain filtered.
    if len(shorter) < 50:
        return False
    if len(shorter) >= 60 and shorter in longer:
        return True
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.88


def _build_web_query(question: str, missing_information: tuple[str, ...]) -> str:
    question = _strip_dependency_binding(question)
    missing = "；".join(item.strip() for item in missing_information if item.strip())
    query = question if not missing else f"{question} 补充核实：{missing}"
    return _cap_search_query(query)


_DEPENDENCY_BINDING_MARKER = "以下 JSON 是已经执行的依赖任务结果"


def _strip_dependency_binding(question: str) -> str:
    """Search engines only need the clause, not the structured dependency dump."""

    head, _, _tail = question.partition(_DEPENDENCY_BINDING_MARKER)
    head = head.strip()
    return head or question


def _cap_search_query(query: str, *, max_chars: int = 900) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    if len(query) <= max_chars:
        return query
    # Prefer the clause head; add the trailing missing-information hint when the
    # provider would otherwise receive an over-long query and reject it.
    tail_marker = " 补充核实："
    head, separator, tail = query.partition(tail_marker)
    if separator and len(tail) < max_chars // 2:
        budget = max_chars - len(tail) - len(separator)
        return head[:budget] + separator + tail
    return query[:max_chars]


def _apply_source_policy_to_query(query: str, policy: WebSourcePolicy) -> str:
    # Provider support for site: syntax is inconsistent; policy enforcement is
    # performed on every returned URL instead of risking a false empty search.
    return query


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", "", query).casefold().rstrip("?？。！!")


def _merge_search_results(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    merged: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in (*first, *second):
        if not isinstance(item, Mapping):
            continue
        key = _canonical_url(str(item.get("url") or ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(item)
    return merged


def _answer_with_mixed_provenance(
    answer: str,
    supported_evidence_ids: tuple[str, ...],
    evidence: EvidenceBundle,
    *,
    claims: Sequence[Mapping[str, Any]] = (),
) -> str:
    by_id = {
        item.retrieval_unit_id: item
        for item in evidence.items
        if item.verified and item.source_type is not SourceType.MODEL_PRIOR
    }
    citations: list[str] = []
    for identifier in supported_evidence_ids:
        if identifier not in by_id:
            raise ValueError("final answer references unverified or model-prior evidence")
        item = by_id[identifier]
        if item.source_type.value == "rag":
            pages = "、".join(str(page) for page in item.page) if item.page else "未知"
            citations.append(
                f"- [local_document:{identifier}] {item.company_name or '未知公司'}；"
                f"{item.source_file or '未知文件'}；第{pages}页"
            )
        else:
            excerpt = re.sub(r"\s+", " ", item.text).strip()[:220]
            metadata = "；".join(
                part for part in (
                    f"级别={item.source_tier}" if item.source_tier else "",
                    f"发布日期={item.published_at}" if item.published_at else "",
                    f"检索时间={item.retrieved_at}" if item.retrieved_at else "",
                ) if part
            )
            citations.append(
                f"- [web:{identifier}] 结论依据原文：“{excerpt}”；来源："
                f"{item.title or '网页来源'}；{item.url or '未知 URL'}"
                + (f"；{metadata}" if metadata else "")
            )
    claim_lines = []
    for claim in claims:
        claim_lines.append(
            f"- {claim['statement']} "
            + " ".join(f"[{identifier}]" for identifier in claim["evidence_ids"])
        )
    claim_section = ""
    if claim_lines:
        claim_section = "\n\n结论—证据映射：\n" + "\n".join(claim_lines)
    return answer.strip() + claim_section + "\n\n证据来源：\n" + "\n".join(citations)


def _prefix_web_evidence(bundle: EvidenceBundle, prefix: str) -> EvidenceBundle:
    return replace(
        bundle,
        items=tuple(
            replace(item, retrieval_unit_id=f"{prefix}-{item.retrieval_unit_id}")
            for item in bundle.items
        ),
    )


async def _default_search(query: str, **kwargs: Any) -> Sequence[Mapping[str, Any]]:
    try:
        from search_provider import default_provider
    except ImportError:
        from ..search_provider import default_provider
    return await default_provider.search(query, **kwargs)


async def _default_fetch(url: str) -> str:
    try:
        from page_fetcher import fetch_page_content
    except ImportError:
        from ..page_fetcher import fetch_page_content
    return await fetch_page_content(url)
