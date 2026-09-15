"""Rule-first source planning without executing tools or touching index IDs."""

from __future__ import annotations

import re
from typing import Protocol, TYPE_CHECKING

from .entity_resolver import EntityResolver, normalize_entity_text
from .models import EntityResolution, PlanMode, RouterDecision, SourcePlan, SourceType
from .tool_registry import ToolRegistry
from .trace import record_trace_event

if TYPE_CHECKING:
    from .tool_necessity import ToolNecessityDecision


_SOURCE_ORDER = (SourceType.RAG, SourceType.SQL, SourceType.WEB)
_FUND_DOMAIN_SIGNALS = ("基金", "公募", "私募")
_EXPLICIT_DATABASE_SIGNALS = (
    "数据库",
    "结构化数据",
    "结构化持仓",
    "历史指标",
    "历史持仓数据",
)
_CATALOG_DOCUMENT_QUESTION_SIGNALS = (
    "经营风险",
    "核心风险",
    "主要风险",
    "风险因素",
    "公司业务",
    "主营业务",
    "竞争优势",
    "主要客户",
    "主要供应商",
    "募投项目",
    # Stable facts commonly disclosed in prospectuses/reports. Once the
    # entity resolves to a local catalog document, these should inherit the
    # scoped RAG source instead of paying for one LLM source-routing call per
    # decomposed requirement.
    "营业收入",
    "营业总收入",
    "净利润",
    "总资产",
    "净资产",
    "账面价值",
    "评估价值",
    "交易价格",
    "收购价款",
    "股权转让",
    "注册资本",
    "经营业绩",
)


class AmbiguousSourceRouter(Protocol):
    def route(
        self,
        question: str,
        entity: EntityResolution,
        registry: ToolRegistry,
        *,
        allowed_sources: tuple[SourceType, ...] | None = None,
        rejected_sources: tuple[SourceType, ...] = (),
        rejection_reason: str | None = None,
    ) -> RouterDecision: ...


class SourcePlanner:
    """Create a deterministic plan for clear queries; leave ambiguity as auto."""

    def __init__(
        self,
        resolver: EntityResolver,
        registry: ToolRegistry,
        *,
        ambiguous_router: AmbiguousSourceRouter | None = None,
    ) -> None:
        self.resolver = resolver
        self.registry = registry
        self.ambiguous_router = ambiguous_router

    def plan(self, question: str) -> SourcePlan:
        """Backward-compatible whole-query facade.

        The runtime decision layer uses ``select_external`` only after a
        requirement-level Tool Necessity decision.  This method remains for
        Phase A-D callers and deterministic regression tests.
        """
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be empty")

        entity = self.resolver.resolve(question)
        normalized = normalize_entity_text(question)
        matches = self._external_matches(normalized)
        if self.catalog_document_source(question) is SourceType.RAG:
            matches.add(SourceType.RAG)

        if not matches and _is_local_compute_query(normalized):
            return SourcePlan(
                mode=PlanMode.SINGLE_SOURCE,
                primary_source=SourceType.LOCAL_COMPUTE,
                required_sources=(SourceType.LOCAL_COMPUTE,),
                company_name=entity.company_name,
                catalog_doc_ids=(),
                document_type=entity.document_type,
                web_fallback_allowed=False,
                routing_method="deterministic_no_tool_rule",
                reason="matched deterministic local computation rule",
            )
        if not matches and _is_direct_stable_query(normalized):
            return SourcePlan(
                mode=PlanMode.SINGLE_SOURCE,
                primary_source=SourceType.DIRECT,
                required_sources=(SourceType.DIRECT,),
                company_name=entity.company_name,
                catalog_doc_ids=(),
                document_type=entity.document_type,
                web_fallback_allowed=False,
                routing_method="deterministic_no_tool_rule",
                reason="matched allowlisted stable direct-answer rule",
            )

        return self._external_plan(question, entity, matches)

    def select_external(
        self,
        question: str,
        necessity: "ToolNecessityDecision | None" = None,
    ) -> SourcePlan:
        """Select external sources after Tool Necessity has said they are needed."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if necessity is not None and not necessity.needs_tool:
            raise ValueError("select_external requires a tool-required decision")
        entity = self.resolver.resolve(question)
        normalized = normalize_entity_text(question)
        matches = self._external_matches(normalized)
        if not matches and self.catalog_document_source(question) is SourceType.RAG:
            matches.add(SourceType.RAG)
        if necessity is not None:
            matches.update(necessity.specified_sources)
        requested_document_type = _requested_document_type(normalized)
        if requested_document_type is not None:
            catalog_document_type = None
            if entity.catalog_doc_id is not None:
                catalog_document_type = self.resolver.catalog.get(
                    entity.catalog_doc_id
                ).document_type
            # A specifically named document is RAG only when that document
            # type exists in the scoped local catalog. Otherwise it is Web;
            # this invariant overrides a generic RAG routing hint.
            matches.discard(SourceType.RAG)
            matches.add(
                SourceType.RAG
                if catalog_document_type == requested_document_type
                else SourceType.WEB
            )
        allow_runtime_web_fallback = self._allows_sql_web_fallback(
            question,
            necessity,
            matches,
        )
        return self._external_plan(
            question,
            entity,
            matches,
            allow_optional_web_fallback=allow_runtime_web_fallback,
        )

    def catalog_document_source(self, question: str) -> SourceType | None:
        """Return RAG for stable document-domain facts about a catalogued entity."""

        normalized = normalize_entity_text(question)
        entity = self.resolver.resolve(question)
        if entity.catalog_doc_id is None or not _is_catalog_document_question(normalized):
            return None
        requested_document_type = _requested_document_type(normalized)
        if requested_document_type is not None:
            catalog_document_type = self.resolver.catalog.get(
                entity.catalog_doc_id
            ).document_type
            if catalog_document_type != requested_document_type:
                return None
        matches = self._external_matches(normalized)
        if SourceType.WEB in matches or SourceType.SQL in matches:
            return None
        return SourceType.RAG

    def deterministic_external_sources(self, question: str) -> tuple[SourceType, ...]:
        """Return only entity/schema-proven sources before semantic necessity routing.

        Generic nouns such as ``年报`` and ``新闻`` are deliberately excluded
        here: they can occur in model-answerable conceptual questions. Explicit
        source/freshness constraints remain owned by ToolNecessityPolicy.
        """

        matches: set[SourceType] = set()
        if self.catalog_document_source(question) is SourceType.RAG:
            matches.add(SourceType.RAG)
        else:
            normalized = normalize_entity_text(question)
            entity = self.resolver.resolve(question)
            requested_document_type = _requested_document_type(normalized)
            if requested_document_type is not None and entity.catalog_doc_id is not None:
                catalog_document_type = self.resolver.catalog.get(
                    entity.catalog_doc_id
                ).document_type
                if catalog_document_type != requested_document_type:
                    matches.add(SourceType.WEB)
        return tuple(source for source in _SOURCE_ORDER if source in matches)

    def guarded_external_sources(self, question: str) -> tuple[SourceType, ...]:
        """Return sources whose concrete scope contradicts a no-tool decision."""

        sources = set(self.deterministic_external_sources(question))
        if self._is_sql_query(normalize_entity_text(question)):
            sources.add(SourceType.SQL)
        return tuple(source for source in _SOURCE_ORDER if source in sources)

    @staticmethod
    def _allows_sql_web_fallback(
        question: str,
        necessity: "ToolNecessityDecision | None",
        matches: set[SourceType],
    ) -> bool:
        """Allow bounded recovery only for an ordinary SQL-only selection.

        ``required_sources`` stays SQL-only.  An explicit SQL/local-database
        *only* instruction is a user scope constraint and therefore wins over
        recovery convenience.  A Web match is instead planning-time hybrid and
        must never be reclassified as fallback permission.
        """
        if matches != {SourceType.SQL}:
            return False
        if necessity is not None and not necessity.needs_tool:
            return False
        return not _has_sql_only_constraint(normalize_entity_text(question))

    def _external_plan(
        self,
        question: str,
        entity: EntityResolution,
        matches: set[SourceType],
        *,
        allow_optional_web_fallback: bool = True,
    ) -> SourcePlan:
        catalog_doc_ids = (entity.catalog_doc_id,) if entity.catalog_doc_id else ()
        if not matches:
            if self.ambiguous_router is not None:
                decision = self.ambiguous_router.route(question, entity, self.registry)
                allowed_sources = self._allowed_ambiguous_sources(question, entity)
                invalid_sources = tuple(
                    source
                    for source in decision.required_sources
                    if source not in allowed_sources
                )
                if decision.primary_source not in allowed_sources or invalid_sources:
                    record_trace_event(
                        event_type="planning.source_route.rejected",
                        stage="planning",
                        status="rejected",
                        source=decision.primary_source.value,
                        attributes={
                            "reason": "source_capability_mismatch",
                            "rejected_primary_source": decision.primary_source.value,
                            "rejected_required_sources": [
                                source.value for source in decision.required_sources
                            ],
                            "allowed_sources": [source.value for source in allowed_sources],
                        },
                    )
                    rejected_sources = tuple(dict.fromkeys((
                        *((decision.primary_source,)
                          if decision.primary_source not in allowed_sources else ()),
                        *invalid_sources,
                    )))
                    decision = self.ambiguous_router.route(
                        question,
                        entity,
                        self.registry,
                        allowed_sources=allowed_sources,
                        rejected_sources=rejected_sources,
                        rejection_reason="source_capability_mismatch",
                    )
                    if (
                        decision.primary_source not in allowed_sources
                        or any(source not in allowed_sources for source in decision.required_sources)
                    ):
                        raise ValueError("LLM source reroute violated source capability constraints")
                return SourcePlan(
                    mode=(PlanMode.MULTI_SOURCE if len(decision.required_sources) > 1
                          else PlanMode.SINGLE_SOURCE),
                    primary_source=decision.primary_source,
                    required_sources=decision.required_sources,
                    company_name=entity.company_name,
                    catalog_doc_ids=catalog_doc_ids,
                    document_type=entity.document_type,
                    web_fallback_allowed=(
                        decision.web_fallback_allowed if allow_optional_web_fallback else False
                    ),
                    routing_method="llm",
                    reason=decision.reason or "lightweight source selection",
                )
            return SourcePlan(
                mode=PlanMode.AUTO,
                primary_source=None,
                required_sources=(),
                company_name=entity.company_name,
                catalog_doc_ids=catalog_doc_ids,
                document_type=entity.document_type,
                routing_method="auto_pending",
                reason="external capability is required but no source was selected",
            )

        required_sources = tuple(source for source in _SOURCE_ORDER if source in matches)
        mode = PlanMode.MULTI_SOURCE if len(required_sources) > 1 else PlanMode.SINGLE_SOURCE
        return SourcePlan(
            mode=mode,
            primary_source=required_sources[0],
            required_sources=required_sources,
            company_name=entity.company_name,
            catalog_doc_ids=catalog_doc_ids,
            document_type=entity.document_type,
            web_fallback_allowed=allow_optional_web_fallback,
            routing_method="rules",
            reason="matched explicit source signals: " + ", ".join(source.value for source in required_sources),
        )

    def _allowed_ambiguous_sources(
        self,
        question: str,
        entity: EntityResolution,
    ) -> tuple[SourceType, ...]:
        normalized = normalize_entity_text(question)
        allowed: set[SourceType] = {SourceType.WEB}
        if entity.catalog_doc_id is not None:
            allowed.add(SourceType.RAG)
        if self._is_sql_query(normalized) or any(
            signal in normalized for signal in _FUND_DOMAIN_SIGNALS
        ):
            allowed.add(SourceType.SQL)
        return tuple(source for source in _SOURCE_ORDER if source in allowed)

    def _external_matches(self, normalized_question: str) -> set[SourceType]:
        matches: set[SourceType] = set()
        if self._contains_hint(normalized_question, SourceType.RAG):
            matches.add(SourceType.RAG)
        if self._contains_hint(normalized_question, SourceType.WEB):
            matches.add(SourceType.WEB)
        if _has_explicit_web_request(normalized_question):
            matches.add(SourceType.WEB)
        if self._is_sql_query(normalized_question):
            matches.add(SourceType.SQL)
        return matches

    def _contains_hint(self, normalized_question: str, source_type: SourceType) -> bool:
        return any(
            normalize_entity_text(hint) in normalized_question
            for hint in self.registry.routing_hints(source_type)
        )

    def _is_sql_query(self, normalized_question: str) -> bool:
        if any(signal in normalized_question for signal in _EXPLICIT_DATABASE_SIGNALS):
            return True
        if not any(signal in normalized_question for signal in _FUND_DOMAIN_SIGNALS):
            return False
        hints = self.registry.routing_hints(SourceType.SQL)
        if any(normalize_entity_text(hint) in normalized_question for hint in hints):
            return True
        # A dated fund question with an explicit tabular comparison is also in
        # the verified database domain. Generic mentions of funds remain auto.
        has_year = bool(re.search(r"(?:19|20)\d{2}年?", normalized_question))
        has_comparison = any(term in normalized_question for term in ("前", "最高", "最低", "多少"))
        return has_year and has_comparison


def _has_sql_only_constraint(question: str) -> bool:
    """Recognize an explicit user prohibition on non-SQL recovery.

    Mentioning a database alone is not restrictive: it may simply identify the
    requested primary source.  The wording must also limit the execution scope
    (for example "只查本地数据库" or "finance.db 就行").
    """
    database = r"(?:finance\.db|本地数据库|结构化数据库|数据库|sql)"
    return bool(
        re.search(rf"(?:只|仅|仅限|只能|限于|只允许).{{0,12}}{database}", question)
        or re.search(rf"{database}.{{0,12}}(?:即可|就行|就好|不要联网|不联网|不查网页)", question)
    )


def _is_catalog_document_question(question: str) -> bool:
    """Recognize stable company facts covered by an already-scoped local document."""

    return any(signal in question for signal in _CATALOG_DOCUMENT_QUESTION_SIGNALS)


def _requested_document_type(question: str) -> str | None:
    if "招股说明书" in question or "招股书" in question:
        return "prospectus"
    if "年度报告" in question or "年报" in question:
        return "annual_report"
    if "财务报告" in question or "财报" in question:
        return "financial_report"
    return None


def _is_local_compute_query(question: str) -> bool:
    if any(term in question for term in ("最新", "最近", "近期", "今天", "新闻", "联网", "网页")):
        return False
    if re.search(r"\d+(?:\.\d+)?%?(?:转换|换算)(?:成|为)(?:百分比|小数)", question):
        return True
    if (
        any(currency in question for currency in ("美元", "人民币", "欧元", "日元"))
        and any(term in question for term in ("相加", "相减", "合计", "换算", "缺什么", "不能"))
    ):
        return True
    # Deterministic word problems must remain local computations too.  These
    # signals require numeric literals so that questions asking for a current
    # market return are still routed to evidence-backed sources.
    if len(re.findall(r"-?\d+(?:\.\d+)?%", question)) >= 2 and any(
        term in question
        for term in ("上涨", "下跌", "收益率", "增长率", "百分点", "平均", "复利", "累计")
    ):
        return True
    compact = question.rstrip("？?")
    return bool(re.fullmatch(r"[\d.()+\-*/×÷%\s]+(?:等于多少|是多少|计算)?", compact))


def _is_direct_stable_query(question: str) -> bool:
    if any(
        term in question
        for term in (
            "最新", "最近", "近期", "今天", "今日", "今年", "去年", "现在", "截至",
            "新闻", "当前", "实时", "来源", "搜索", "联网", "网页", "行情", "价格",
            "推荐买", "买入", "卖出", "投资建议", "诊断", "治疗", "用药", "法律建议",
        )
    ):
        return False
    if "圆周率" in question or re.search(r"(?:pi|π)(?:的)?(?:前|小数)", question, re.IGNORECASE):
        return True
    if len(question) > 80:
        return False
    stable_patterns = (
        r"^什么是.{1,40}[？?]?$",
        r"^.{1,40}(?:是什么|是什么意思|是谁|在哪里|是哪里|是多少)[？?]?$",
        r"^为什么.{1,40}[？?]?$",
        r"^.{1,40}(?:吗|么)[？?]?$",
        r"^(?:请)?(?:简单|简要)?(?:解释|说明|介绍).{1,40}[？?]?$",
    )
    return any(re.fullmatch(pattern, question) for pattern in stable_patterns)


def _has_explicit_web_request(question: str) -> bool:
    return any(
        term in question
        for term in ("联网", "网页搜索", "网络搜索", "web搜索", "外部来源", "提供来源")
    )
