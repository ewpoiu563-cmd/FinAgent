"""Web task complexity classification and strict multi-hop plan parsing."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class WebTaskComplexity(str, Enum):
    SIMPLE_FACTUAL = "simple_factual"
    STRUCTURED_RESEARCH = "structured_research"
    LEGACY_RESEARCH = "legacy_research"


@dataclass(frozen=True)
class WebSubquestion:
    id: str
    subquestion: str
    depends_on: tuple[str, ...]
    expected_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subquestion": self.subquestion,
            "depends_on": list(self.depends_on),
            "expected_output": self.expected_output,
        }


@dataclass(frozen=True)
class WebResearchPlan:
    subquestions: tuple[WebSubquestion, ...]
    planning_method: str = "llm"
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "subquestions": [item.to_dict() for item in self.subquestions],
            "planning_method": self.planning_method,
            "latency_ms": self.latency_ms,
        }


class WebResearchPlanningError(ValueError):
    pass


class WebResearchPlanner:
    """Use one light LLM call to create a bounded dependency-ordered plan."""

    def __init__(
        self,
        llm_call: Callable[..., str] | None = None,
        *,
        timeout: int = 45,
        max_subquestions: int = 5,
    ) -> None:
        if timeout <= 0 or max_subquestions < 2:
            raise ValueError("invalid Web planner limits")
        self._llm_call = llm_call
        self.timeout = timeout
        self.max_subquestions = max_subquestions

    def plan(self, question: str) -> WebResearchPlan:
        if classify_web_task(question) is not WebTaskComplexity.STRUCTURED_RESEARCH:
            raise WebResearchPlanningError("question is not eligible for structured Web planning")
        call = self._llm_call
        if call is None:
            from config import call_llm

            call = call_llm
        started = time.perf_counter()
        kwargs = {"temperature": 0.0, "timeout": self.timeout}
        if self._llm_call is None:
            kwargs["trace_operation"] = "web_planning"
        raw = call(_planner_prompt(question, self.max_subquestions), **kwargs)
        plan = self.parse_result(raw, original_question=question, max_subquestions=self.max_subquestions)
        return WebResearchPlan(
            subquestions=plan.subquestions,
            planning_method="llm",
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def parse_result(
        raw: str,
        *,
        original_question: str,
        max_subquestions: int = 5,
    ) -> WebResearchPlan:
        if not isinstance(raw, str) or not raw.strip():
            raise WebResearchPlanningError("Web plan output must be non-empty JSON")
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise WebResearchPlanningError("Web plan output must be valid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"subquestions"}:
            raise WebResearchPlanningError("Web plan must contain only subquestions")
        raw_steps = payload["subquestions"]
        if not isinstance(raw_steps, list) or not 2 <= len(raw_steps) <= max_subquestions:
            raise WebResearchPlanningError(f"Web plan must contain 2 through {max_subquestions} subquestions")

        steps: list[WebSubquestion] = []
        seen: set[str] = set()
        original_normalized = _normalize(original_question)
        for position, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict) or set(raw_step) != {
                "id", "subquestion", "depends_on", "expected_output"
            }:
                raise WebResearchPlanningError(f"subquestion {position} has an invalid schema")
            identifier = raw_step["id"]
            subquestion = raw_step["subquestion"]
            depends_on = raw_step["depends_on"]
            expected_output = raw_step["expected_output"]
            if not isinstance(identifier, str) or not re.fullmatch(r"SQ[1-9]\d*", identifier):
                raise WebResearchPlanningError(f"subquestion {position} has an invalid id")
            if identifier in seen:
                raise WebResearchPlanningError(f"duplicate subquestion id: {identifier}")
            if not isinstance(subquestion, str) or not subquestion.strip():
                raise WebResearchPlanningError(f"subquestion {identifier} must be non-empty")
            if _normalize(subquestion) == original_normalized:
                raise WebResearchPlanningError("a subquestion must not repeat the full original question")
            if not isinstance(expected_output, str) or not expected_output.strip():
                raise WebResearchPlanningError(f"subquestion {identifier}.expected_output must be non-empty")
            if not isinstance(depends_on, list) or any(not isinstance(item, str) for item in depends_on):
                raise WebResearchPlanningError(f"subquestion {identifier}.depends_on must be a list[str]")
            if len(set(depends_on)) != len(depends_on) or any(item not in seen for item in depends_on):
                raise WebResearchPlanningError(
                    f"subquestion {identifier} dependencies must be unique earlier steps"
                )
            raw_placeholders = set(re.findall(r"\{([^{}]+)\}", subquestion))
            if any(not re.fullmatch(r"SQ[1-9]\d*", item) for item in raw_placeholders):
                raise WebResearchPlanningError(
                    f"subquestion {identifier} contains an unsupported placeholder"
                )
            placeholders = raw_placeholders
            if placeholders != set(depends_on):
                raise WebResearchPlanningError(
                    f"subquestion {identifier} placeholders must exactly match depends_on"
                )
            steps.append(
                WebSubquestion(
                    id=identifier,
                    subquestion=subquestion.strip(),
                    depends_on=tuple(depends_on),
                    expected_output=expected_output.strip(),
                )
            )
            seen.add(identifier)
        if not any(step.depends_on for step in steps):
            raise WebResearchPlanningError("structured Web plan must contain at least one dependency")
        return WebResearchPlan(tuple(steps))


def classify_web_task(question: str) -> WebTaskComplexity:
    normalized = " ".join(question.strip().split()) if isinstance(question, str) else ""
    if not normalized:
        return WebTaskComplexity.LEGACY_RESEARCH
    broad_or_news = (
        "新闻", "最近", "近期", "最新", "当前动态", "全面研究", "开放式研究",
    )
    if any(signal in normalized for signal in broad_or_news):
        return WebTaskComplexity.LEGACY_RESEARCH
    dependency_signals = (
        "出生那年", "去世那年", "同一年", "该年份", "那一年", "在他", "在她",
        "其出生", "其去世", "之后又", "然后", "基于上述", "前者", "后者",
        "原著作者", "那部作品", "该作品", "任教的大学", "所在的大学",
        "大学位于", "学校位于", "公司总部所在", "作者获得",
        "的那部", "的那个", "的该",
    )
    structured_signals = (
        "对比", "比较", "分析", "综合", "为什么", "原因", "如何影响", "分别",
        "以及", "并且", "同时", "之间的关系", "共同点", "历届", "证明",
    )
    if (
        any(signal in normalized for signal in dependency_signals)
        or any(signal in normalized for signal in structured_signals)
        or normalized.count("？") + normalized.count("?") > 1
        or len(normalized) > 80
    ):
        return WebTaskComplexity.STRUCTURED_RESEARCH
    factual_signals = (
        "谁", "哪", "何时", "什么时候", "何地", "哪里", "在哪", "多少",
        "几", "是什么", "是否", "哪年", "哪天",
    )
    if any(signal in normalized for signal in factual_signals):
        return WebTaskComplexity.SIMPLE_FACTUAL
    return WebTaskComplexity.LEGACY_RESEARCH


def materialize_subquestion(step: WebSubquestion, findings: dict[str, str]) -> str:
    if any(dependency not in findings for dependency in step.depends_on):
        missing = [dependency for dependency in step.depends_on if dependency not in findings]
        raise WebResearchPlanningError(f"unresolved dependencies for {step.id}: {missing}")
    query = step.subquestion
    for dependency in step.depends_on:
        query = query.replace("{" + dependency + "}", findings[dependency].strip())
    if re.search(r"\{[^{}]+\}", query):
        raise WebResearchPlanningError(f"unresolved placeholder in {step.id}")
    return query.strip()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold().rstrip("?？。！!")


def _planner_prompt(question: str, max_subquestions: int) -> str:
    return (
        "你是 Web 多跳研究规划器，不执行搜索也不回答问题。把原问题拆成有向无环的子问题。"
        "模型内部知识只能用于分解、候选假设和查询措辞，不能当作已验证事实写入计划或最终证据；"
        "任何人物、年份、地点或数字结论都必须由后续 Search/Web evidence 验证。"
        "只有已完成依赖后才能构造下一步查询。依赖结果必须用 {SQ1} 形式的占位符引用；"
        "depends_on 必须与 subquestion 中的占位符完全一致。第一步不得复述整道原题。"
        f"最多 {max_subquestions} 步。只输出 JSON，且顶层只能有 subquestions；每步字段严格为 "
        "id、subquestion、depends_on、expected_output。示例："
        '{"subquestions":['
        '{"id":"SQ1","subquestion":"事件 A 的人物是谁？","depends_on":[],"expected_output":"person_name"},'
        '{"id":"SQ2","subquestion":"{SQ1}的出生年份是哪一年？","depends_on":["SQ1"],"expected_output":"year"},'
        '{"id":"SQ3","subquestion":"{SQ2}年发生的 B 是什么？","depends_on":["SQ2"],"expected_output":"event"}'
        "]}\n原问题：" + question
    )
