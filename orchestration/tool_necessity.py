"""Requirement-level policy deciding whether external capability is necessary."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date
from enum import Enum
from typing import Callable, Iterable, Sequence

from .models import SourceType
from .task_decomposition import DecomposedRequirement
from .tool_registry import ToolRegistry


class ToolDecisionMode(str, Enum):
    NO_TOOL = "no_tool"
    TOOL_REQUIRED = "tool_required"
    EXPLORATORY = "exploratory"


class ToolNecessityError(ValueError):
    """Raised when necessity planning violates its structured contract."""


@dataclass(frozen=True)
class ToolNecessityDecision:
    task_id: str
    mode: ToolDecisionMode
    reason: str
    information_gap: str | None = None
    external_evidence_required: bool = False
    freshness_required: bool = False
    specified_sources: tuple[SourceType, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    decision_method: str = "llm"

    @property
    def needs_tool(self) -> bool:
        return self.mode is ToolDecisionMode.TOOL_REQUIRED

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "needs_tool": self.needs_tool,
            "reason": self.reason,
            "information_gap": self.information_gap,
            "external_evidence_required": self.external_evidence_required,
            "freshness_required": self.freshness_required,
            "specified_sources": [source.value for source in self.specified_sources],
            "required_capabilities": list(self.required_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "decision_method": self.decision_method,
        }


class ToolNecessityPolicy:
    """Apply hard external constraints, then one semantic decision for the rest."""

    _VERIFICATION = re.compile(r"(?:查证|核实|验证|提供|列出|附上).{0,8}(?:来源|证据|引用)|(?:来源|证据|引用).{0,8}(?:查证|核实|验证|提供|列出|附上)")
    _EXTERNAL_ACTION = re.compile(r"(?:发送|发布|创建|修改|删除|下单|购买|预订|上传|下载|提交)(?:邮件|消息|任务|日程|订单|文件|表单|记录)?")
    _EXPLICIT_DATABASE = re.compile(r"(?:finance\.db|本地数据库|结构化数据库|数据库中|查询数据库)", re.IGNORECASE)
    _EXPLICIT_LOCAL_DOCUMENT = re.compile(
        r"(?:本地|已上传|上传的|提供的|指定的|附件|文档库).{0,24}"
        r"(?:文档|文件|报告|招股书|招股说明书|年报|财报|pdf)",
        re.IGNORECASE,
    )
    _DOCUMENT_FACT_REFERENCE = re.compile(
        r"(?:(?:根据|来自|查阅|查看).{0,16}(?:招股说明书|招股书|年度报告|年报|财务报告|财报)"
        r"|(?:招股说明书|招股书|年度报告|年报|财务报告|财报).{0,16}(?:披露|记载|显示|报告的))",
        re.IGNORECASE,
    )
    _FRESHNESS = re.compile(r"(?:最新|最近|近期|今天|今日|当前|现在|实时|刚刚)")
    _CURRENT_CUTOFF = re.compile(r"截至\s*(?:当前|现在|今日|今天|目前)")
    _CURRENT_ANCHOR = re.compile(r"(?:当前|现在|今天|今日|至今|截至\s*(?:当前|现在|今日|今天|目前))")
    _FRESHNESS_CAPABILITY = re.compile(r"(?:实时|最新|当前|动态|fresh)", re.IGNORECASE)

    def __init__(
        self,
        registry: ToolRegistry,
        llm_call: Callable[..., str] | None = None,
        *,
        timeout: int = 30,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.registry = registry
        self._llm_call = llm_call
        self.timeout = timeout

    def decide(
        self,
        requirements: Sequence[DecomposedRequirement],
        *,
        conversation_context: str = "",
        reference_date: date | None = None,
    ) -> tuple[ToolNecessityDecision, ...]:
        if not requirements:
            raise ValueError("requirements must not be empty")
        temporal_reference = reference_date or date.today()
        decisions: dict[str, ToolNecessityDecision] = {}
        undecided: list[DecomposedRequirement] = []
        for requirement in requirements:
            hard = self._hard_constraint(requirement, temporal_reference)
            if hard is None:
                undecided.append(requirement)
            else:
                decisions[requirement.id] = hard

        if undecided:
            call = self._llm_call
            if call is None:
                from config import call_llm

                call = call_llm
            kwargs = {"temperature": 0.0, "timeout": self.timeout}
            if self._llm_call is None:
                kwargs["trace_operation"] = "tool_necessity"
            raw = call(self._prompt(undecided, conversation_context), **kwargs)
            requirements_by_id = {item.id: item for item in undecided}
            for decision in self.parse(raw, set(requirements_by_id)):
                decisions[decision.task_id] = self._normalize_historical_freshness(
                    decision,
                    requirements_by_id[decision.task_id],
                    temporal_reference,
                )
        return tuple(decisions[item.id] for item in requirements)

    def _hard_constraint(
        self,
        requirement: DecomposedRequirement,
        reference_date: date,
    ) -> ToolNecessityDecision | None:
        text = requirement.question.casefold()
        sources: list[SourceType] = []
        capabilities: list[str] = []
        # These rules only encode externally observable constraints.  Domain
        # nouns such as "年报" or "新闻" are intentionally not sufficient by
        # themselves; ordinary semantics stay with the lightweight policy.
        freshness = _requires_freshness(text, reference_date)
        explicit_rag = bool(self._EXPLICIT_LOCAL_DOCUMENT.search(text))
        document_fact_reference = bool(self._DOCUMENT_FACT_REFERENCE.search(text))
        explicit_web = bool(re.search(r"https?://|(?:联网|网页|网络搜索|web搜索)", text, re.IGNORECASE))
        explicit_sql = bool(self._EXPLICIT_DATABASE.search(text))

        if explicit_rag:
            sources.append(SourceType.RAG)
            capabilities.append("specified_local_document_evidence")
        if explicit_sql:
            sources.append(SourceType.SQL)
            capabilities.append("specified_structured_database_query")
        if document_fact_reference:
            capabilities.append("specified_document_evidence")
        if freshness or explicit_web:
            sources.append(SourceType.WEB)
            capabilities.append("fresh_or_specified_web_information")
        verification = bool(self._VERIFICATION.search(text))
        action = bool(self._EXTERNAL_ACTION.search(text))
        if verification:
            capabilities.append("external_verification_evidence")
        if action:
            capabilities.append("external_action")
        if not sources and not verification and not action and not document_fact_reference:
            return None
        return ToolNecessityDecision(
            task_id=requirement.id,
            mode=ToolDecisionMode.TOOL_REQUIRED,
            reason="the task contains an explicit external-source, freshness, evidence, or action constraint",
            information_gap=(
                "external action capability is required"
                if action
                else "the requested external information or evidence is outside model-only execution"
            ),
            external_evidence_required=bool(sources or verification),
            freshness_required=freshness,
            specified_sources=tuple(dict.fromkeys(sources)),
            required_capabilities=tuple(dict.fromkeys(capabilities)),
            decision_method="deterministic_hard_constraint",
        )

    @staticmethod
    def _prompt(
        requirements: Iterable[DecomposedRequirement],
        conversation_context: str,
    ) -> str:
        items = [item.to_dict() for item in requirements]
        return (
            "你是 FinAgent 的 Tool Necessity Policy。对每个原子任务只判断是否必须使用外部能力，"
            "不要选择具体工具，不要回答任务。默认最小充分能力集合：如果当前对话上下文、稳定模型知识和推理"
            "足以可靠完成，选择 no_tool；只有存在明确的信息缺口、外部证据要求、时效要求、私有数据访问或外部动作"
            "时选择 tool_required。开放研究、依赖动态探索且无法预先形成可靠工具计划时才选择 exploratory。"
            "不要用主观 confidence 数值，不要为了保险选择工具。optional_capabilities 不会被默认执行。"
            "严格区分历史时间约束和 freshness：绝对历史时间（如 2021年、2025年、2024Q3、"
            "截至2025-12-31）以及已经解析为确定历史年份的相对时间，只限定查询范围，"
            "本身不要求 freshness；此类任务可仍为 tool_required，但 freshness_required 必须为 false。"
            "只有用户要求当前/动态状态（当前、现在、今天、最新、实时、最近或近期动态、截至当前）"
            "时 freshness_required 才为 true。freshness 不决定是否允许运行时 source fallback。"
            "当 mode=tool_required 时，required_capabilities 必须至少包含一项；"
            "如无法给出更具体的能力名，使用 external_information_lookup，不得返回空数组。"
            "返回严格 JSON：{\"decisions\":[{\"task_id\":\"T1\","
            "\"mode\":\"no_tool|tool_required|exploratory\",\"reason\":\"...\","
            "\"information_gap\":null,\"external_evidence_required\":false,"
            "\"freshness_required\":false,\"required_capabilities\":[],"
            "\"optional_capabilities\":[]}]}。\n"
            f"当前对话上下文：{conversation_context or '(none supplied)'}\n"
            f"原子任务：{json.dumps(items, ensure_ascii=False)}"
        )

    @classmethod
    def parse(cls, raw: str, expected_ids: set[str]) -> tuple[ToolNecessityDecision, ...]:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolNecessityError("necessity output must be non-empty")
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ToolNecessityError("necessity output must be valid JSON") from error
        records = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ToolNecessityError("decisions must be a list")
        decisions: list[ToolNecessityDecision] = []
        for record in records:
            if not isinstance(record, dict):
                raise ToolNecessityError("each decision must be an object")
            task_id = record.get("task_id")
            reason = record.get("reason")
            gap = record.get("information_gap")
            evidence = record.get("external_evidence_required")
            freshness = record.get("freshness_required")
            required = _string_list(record.get("required_capabilities"), "required_capabilities")
            optional = _string_list(record.get("optional_capabilities"), "optional_capabilities")
            try:
                mode = ToolDecisionMode(record.get("mode"))
            except ValueError as error:
                raise ToolNecessityError("invalid necessity mode") from error
            if task_id not in expected_ids or not isinstance(reason, str) or not reason.strip():
                raise ToolNecessityError("decision task_id/reason is invalid")
            if gap is not None and not isinstance(gap, str):
                raise ToolNecessityError("information_gap must be string or null")
            if type(evidence) is not bool or type(freshness) is not bool:
                raise ToolNecessityError("evidence and freshness fields must be boolean")
            if mode is ToolDecisionMode.NO_TOOL and (required or evidence or freshness):
                raise ToolNecessityError("no_tool cannot require external capabilities")
            repaired_missing_capability = mode is ToolDecisionMode.TOOL_REQUIRED and not required
            if repaired_missing_capability:
                # Capability names are planning metadata; source selection still
                # happens in SourcePlanner.  A model occasionally returns the
                # semantically usable combination ``tool_required`` + an empty
                # list.  Repair that one omission instead of dropping the whole
                # request into the much broader legacy ReAct fallback.
                required = ("external_information_lookup",)
            decisions.append(
                ToolNecessityDecision(
                    task_id=task_id,
                    mode=mode,
                    reason=reason.strip(),
                    information_gap=gap.strip() if isinstance(gap, str) and gap.strip() else None,
                    external_evidence_required=evidence,
                    freshness_required=freshness,
                    required_capabilities=required,
                    optional_capabilities=optional,
                    decision_method=(
                        "llm_semantic_policy_repaired"
                        if repaired_missing_capability
                        else "llm_semantic_policy"
                    ),
                )
            )
        if {item.task_id for item in decisions} != expected_ids or len(decisions) != len(expected_ids):
            raise ToolNecessityError("decisions must cover every requirement exactly once")
        return tuple(decisions)

    @classmethod
    def _normalize_historical_freshness(
        cls,
        decision: ToolNecessityDecision,
        requirement: DecomposedRequirement,
        reference_date: date,
    ) -> ToolNecessityDecision:
        """Normalize only semantic claims contradicted by temporal scope."""
        scope = _temporal_scope(requirement.question, reference_date)
        if scope == "current" and decision.needs_tool and not decision.freshness_required:
            return replace(
                decision,
                freshness_required=True,
                required_capabilities=tuple(dict.fromkeys(
                    (*decision.required_capabilities, "current_period_external_data_lookup")
                )),
            )
        if not decision.freshness_required or scope != "historical":
            return decision
        required = tuple(
            capability
            for capability in decision.required_capabilities
            if not cls._FRESHNESS_CAPABILITY.search(capability)
        )
        if decision.mode is ToolDecisionMode.TOOL_REQUIRED and not required:
            required = ("historical_external_data_lookup",)
        return replace(
            decision,
            freshness_required=False,
            required_capabilities=required,
        )


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ToolNecessityError(f"{name} must be a list of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _requires_freshness(text: str, reference_date: date) -> bool:
    """Return true for current anchors or a temporal interval overlapping now."""
    scope = _temporal_scope(text, reference_date)
    if scope == "current":
        return True
    if scope == "historical":
        return False
    return bool(
        ToolNecessityPolicy._FRESHNESS.search(text)
        or ToolNecessityPolicy._CURRENT_CUTOFF.search(text)
    )


def _temporal_scope(text: str, reference_date: date) -> str | None:
    """Classify explicit time without making a particular year special."""
    if ToolNecessityPolicy._CURRENT_ANCHOR.search(text):
        return "current"
    intervals = _absolute_intervals(text)
    if not intervals:
        return None
    if all(end < reference_date for _start, end in intervals):
        return "historical"
    if any(start <= reference_date <= end for start, end in intervals):
        return "current"
    return "nonhistorical"


def _absolute_intervals(text: str) -> tuple[tuple[date, date], ...]:
    """Return bounded year/quarter/month/day intervals found in the text."""
    intervals: list[tuple[date, date]] = []
    consumed: list[tuple[int, int]] = []
    patterns = (
        re.compile(r"(?P<year>(?:19|20)\d{2})\s*[-/.年]\s*(?P<month>\d{1,2})\s*[-/.月]\s*(?P<day>\d{1,2})日?"),
        re.compile(r"(?P<year>(?:19|20)\d{2})\s*年?\s*(?:Q(?P<quarter>[1-4])|第?(?P<cquarter>[一二三四])季度)", re.IGNORECASE),
        re.compile(r"(?P<year>(?:19|20)\d{2})年\s*(?P<month>\d{1,2})月"),
        re.compile(r"(?P<year>(?:19|20)\d{2})年"),
    )
    for index, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            if any(match.start() >= start and match.end() <= end for start, end in consumed):
                continue
            try:
                year = int(match.group("year"))
                if index == 0:
                    point = date(year, int(match.group("month")), int(match.group("day")))
                    interval = (point, point)
                elif index == 1:
                    quarter_text = match.group("quarter") or match.group("cquarter")
                    quarter = int(quarter_text) if quarter_text.isdigit() else "一二三四".index(quarter_text) + 1
                    start_month = (quarter - 1) * 3 + 1
                    interval = (date(year, start_month, 1), date(year, start_month + 2, _month_end(year, start_month + 2)))
                elif index == 2:
                    month = int(match.group("month"))
                    interval = (date(year, month, 1), date(year, month, _month_end(year, month)))
                else:
                    interval = (date(year, 1, 1), date(year, 12, 31))
            except ValueError:
                continue
            intervals.append(interval)
            consumed.append((match.start(), match.end()))
    return tuple(intervals)


def _month_end(year: int, month: int) -> int:
    if month == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    return 30 if month in {4, 6, 9, 11} else 31
