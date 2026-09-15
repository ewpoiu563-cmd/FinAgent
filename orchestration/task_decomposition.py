"""Lightweight requirement decomposition before tool-necessity decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Sequence


class TaskDecompositionError(ValueError):
    """Raised when a semantic decomposition cannot satisfy its contract."""


@dataclass(frozen=True)
class DecomposedRequirement:
    id: str
    question: str
    required: bool = True
    depends_on: tuple[str, ...] = ()
    expected_output: str | None = None
    retrieval_query: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "question": self.question,
            "required": self.required,
            "depends_on": list(self.depends_on),
            "expected_output": self.expected_output,
            "retrieval_query": self.retrieval_query,
        }


class LightweightTaskDecomposer:
    """Use one bounded semantic call to produce independently executable requirements."""

    def __init__(self, llm_call: Callable[..., str] | None = None, *, timeout: int = 30) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._llm_call = llm_call
        self.timeout = timeout

    def decompose(self, question: str) -> tuple[DecomposedRequirement, ...]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        question = question.strip()
        call = self._llm_call
        if call is None:
            from config import call_llm

            call = call_llm
        kwargs = {"temperature": 0.0, "timeout": self.timeout}
        if self._llm_call is None:
            kwargs["trace_operation"] = "task_decomposition"
        raw = call(self._prompt(question), **kwargs)
        return self.parse(raw)

    @staticmethod
    def _prompt(question: str) -> str:
        return (
            "你是 FinAgent 的轻量 Requirement Decomposer。每个 requirement 必须是一个能够独立执行、"
            "独立获得 evidence、独立判断 success/insufficient 的原子信息需求。只要用户要求分别得到结果，"
            "即使它们使用相同来源和指标，只在时间、实体、条件或比较对象上不同，也必须拆成不同 requirement。"
            "省略表达必须继承原问题中的共享指标、实体和约束，使每个 requirement 的 question 脱离原句后仍然完整。"
            "年份、日期、年末/截至等时间点、指标口径，以及用户指定的数据库、文档或 Web 来源约束必须原样保留，"
            "不得改写成其他时间聚合、其他指标或其他来源。"
            "不要根据连接词或领域关键词机械拆分；单一交付目标即使需要多步推理也保持一个 requirement。"
            "同一句披露中的并列事实（如同一句里的起始时间、并列做法、并列物料、认定或名次）属于同一个交付目标，"
            "必须保留在同一个 requirement 中，不得按并列词逐项拆分。"
            "由两个端点共同计算的指标（增长率、差额、差值、比值、倍数）必须保留在同一个 requirement 中，"
            "不得拆成“首个交易日的增长率”“最后一个交易日的增长率”这类无法独立计算的子问题。"
            "用户用顿号、和、以及列举的每一项（如“未来5年、2035年和本世纪中叶的目标”）必须各自成为独立 requirement；"
            "不得用“全文是什么”“全称是什么”“全文在哪里可以找到”这类获取原文的脚手架问题替代用户列举的问题，"
            "也不得添加用户没有要求的内容。"
            "针对同一份文档或同一条公告的一组属性（如数值、日期、原文链接、页面名称、文号），"
            "必须保留在同一个 requirement 中，不得拆成多个各自检索的子问题。"
            "此阶段不要解析相对时间，保留其在对应 requirement 中；不要决定工具或来源，不要回答问题。"
            "如果该 requirement 明显需要从文档中检索证据，同时生成 retrieval_query：使用文档中可能真实出现的"
            "章节标题、指标名、实体名、年份和同义词，去掉口语化问法；否则为 null。"
            "返回严格 JSON：{\"requirements\":[{\"id\":\"T1\",\"question\":\"...\","
            "\"required\":true,\"depends_on\":[],\"expected_output\":\"...\","
            "\"retrieval_query\":null}]}。"
            "ID 唯一；依赖只能引用已列出的 ID；不得添加用户未要求的任务。\n"
            f"用户问题：{question}"
        )

    @classmethod
    def parse(cls, raw: str) -> tuple[DecomposedRequirement, ...]:
        payload = _json_object(raw, "task decomposition")
        records = payload.get("requirements")
        if not isinstance(records, list) or not records:
            raise TaskDecompositionError("requirements must be a non-empty list")
        requirements: list[DecomposedRequirement] = []
        for record in records:
            if not isinstance(record, dict):
                raise TaskDecompositionError("each requirement must be an object")
            identifier = record.get("id")
            question = record.get("question")
            required = record.get("required", True)
            depends_on = record.get("depends_on", [])
            expected = record.get("expected_output")
            retrieval_query = record.get("retrieval_query")
            if not isinstance(identifier, str) or not identifier.strip():
                raise TaskDecompositionError("requirement id must be non-empty")
            if not isinstance(question, str) or not question.strip():
                raise TaskDecompositionError("requirement question must be non-empty")
            if type(required) is not bool:
                raise TaskDecompositionError("requirement required must be boolean")
            if not isinstance(depends_on, list) or any(not isinstance(item, str) for item in depends_on):
                raise TaskDecompositionError("depends_on must be a list of strings")
            if expected is not None and not isinstance(expected, str):
                raise TaskDecompositionError("expected_output must be a string or null")
            if retrieval_query is not None and (
                not isinstance(retrieval_query, str)
                or not retrieval_query.strip()
                or len(retrieval_query) > 500
            ):
                raise TaskDecompositionError("retrieval_query must be a non-empty bounded string or null")
            requirements.append(
                DecomposedRequirement(
                    id=identifier.strip(),
                    question=question.strip(),
                    required=required,
                    depends_on=tuple(depends_on),
                    expected_output=expected.strip() if isinstance(expected, str) else None,
                    retrieval_query=(
                        retrieval_query.strip()
                        if isinstance(retrieval_query, str)
                        else None
                    ),
                )
            )
        cls._validate_graph(requirements)
        return tuple(requirements)

    @staticmethod
    def _validate_graph(requirements: Sequence[DecomposedRequirement]) -> None:
        identifiers = [item.id for item in requirements]
        if len(set(identifiers)) != len(identifiers):
            raise TaskDecompositionError("requirement ids must be unique")
        known = set(identifiers)
        for item in requirements:
            if item.id in item.depends_on or not set(item.depends_on).issubset(known):
                raise TaskDecompositionError("requirement dependencies are invalid")
        pending = {item.id: set(item.depends_on) for item in requirements}
        while pending:
            ready = {identifier for identifier, dependencies in pending.items() if not dependencies}
            if not ready:
                raise TaskDecompositionError("requirement dependencies contain a cycle")
            pending = {identifier: dependencies - ready for identifier, dependencies in pending.items()
                       if identifier not in ready}


def _json_object(raw: str, label: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise TaskDecompositionError(f"{label} output must be non-empty")
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise TaskDecompositionError(f"{label} output must be valid JSON") from error
    if not isinstance(payload, dict):
        raise TaskDecompositionError(f"{label} output must be an object")
    return payload
