"""No-tool LLM answers and deterministic local computation."""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from .llm_errors import diagnose_llm_error
from .models import PlanMode, SourcePlan, SourceType


logger = logging.getLogger(__name__)
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass(frozen=True)
class DirectExecutionResult:
    status: str
    answer: str | None
    route: str
    latency_ms: int = 0
    error_type: str | None = None

    def public_answer(self) -> str:
        if self.status == "success" and self.answer:
            return self.answer
        if self.status == "clarification_needed":
            return self.answer or "clarification_needed: 需要补充计算口径。"
        return "generation_failure: 无工具直接回答或本地计算未完成。"


class DirectNoToolExecutor:
    """Use an LLM for direct answers while exposing no external tools."""

    def __init__(
        self,
        llm_call: Callable[..., str] | None = None,
        *,
        timeout: int = 60,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._llm_call = llm_call
        self.timeout = timeout

    def execute(self, question: str, plan: SourcePlan) -> DirectExecutionResult:
        if not is_no_tool_plan(plan):
            raise ValueError("DirectNoToolExecutor requires direct or local_compute plan")
        started = time.perf_counter()
        try:
            if plan.primary_source is SourceType.LOCAL_COMPUTE:
                answer = _local_compute(question)
            else:
                answer = self._direct_llm_answer(question)
            return DirectExecutionResult(
                status="success",
                answer=answer,
                route=plan.primary_source.value,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        except Exception as error:
            diagnostic = diagnose_llm_error(error)
            logger.debug(
                "NO_TOOL_DIRECT: %s",
                json.dumps(
                    {
                        "status": "failure",
                        "error_type": type(error).__name__,
                        "failure_kind": diagnostic.failure_kind,
                    },
                    ensure_ascii=False,
                ),
            )
            return DirectExecutionResult(
                status=(
                    "provider_content_block"
                    if diagnostic.failure_kind == "provider_content_block"
                    else "generation_failure"
                ),
                answer=None,
                route=plan.primary_source.value,
                latency_ms=round((time.perf_counter() - started) * 1000),
                error_type=type(error).__name__,
            )

    def _direct_llm_answer(self, question: str) -> str:
        call = self._llm_call
        if call is None:
            from config import call_llm

            call = call_llm
        prompt = (
            "你是 FinAgent 的无工具直接回答器。当前原子任务已经过 Tool Necessity Policy 判断，"
            "当前对话上下文、稳定模型知识与推理能力足以完成它，且没有必须满足的外部能力要求。"
            "直接回答问题；不得声称调用了搜索、数据库或文档工具，"
            "不得伪造引用。保持简洁准确。若问题实际依赖最新信息或外部证据，只说明无法在无工具模式下确认。\n"
            f"Question: {question.strip()}"
        )
        logger.debug(
            "NO_TOOL_DIRECT: %s",
            json.dumps({"status": "started", "question": question.strip()}, ensure_ascii=False),
        )
        kwargs = {"temperature": 0.0, "timeout": self.timeout}
        if self._llm_call is None:
            kwargs["trace_operation"] = "direct_answer"
        answer = call(prompt, **kwargs)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("direct LLM answer must be a non-empty string")
        logger.debug(
            "NO_TOOL_DIRECT: %s",
            json.dumps({"status": "success", "answer_preview": answer[:300]}, ensure_ascii=False),
        )
        return answer.strip()


def is_no_tool_plan(plan: SourcePlan) -> bool:
    return (
        plan.mode is PlanMode.SINGLE_SOURCE
        and plan.primary_source in {SourceType.DIRECT, SourceType.LOCAL_COMPUTE}
        and plan.required_sources == (plan.primary_source,)
    )


_CHAINED_UP_DOWN = re.compile(
    r"(?:先|首先)?(?:上涨|涨|上升|增长)(\d+(?:\.\d+)?)%.*?"
    r"(?:随后|然后|接着|之后|再|又|后)(?:下跌|跌|下降|回落)(\d+(?:\.\d+)?)%"
)
_CHAINED_DOWN_UP = re.compile(
    r"(?:先|首先)?(?:下跌|跌|下降|回落)(\d+(?:\.\d+)?)%.*?"
    r"(?:随后|然后|接着|之后|再|又|后)(?:上涨|涨|上升|增长)(\d+(?:\.\d+)?)%"
)
_PRINCIPAL = re.compile(r"(?:本金|初始金额|起始金额|投入|初始值|起初|原始值)(\d+(?:\.\d+)?)")


def _chained_percentage_move(question: str) -> tuple[Decimal, Decimal] | None:
    """Return the two signed percentage moves of a chained up/down question."""

    match = _CHAINED_UP_DOWN.search(question)
    if match:
        return Decimal(match.group(1)), -Decimal(match.group(2))
    match = _CHAINED_DOWN_UP.search(question)
    if match:
        return -Decimal(match.group(1)), Decimal(match.group(2))
    return None


def _principal_match(question: str):
    return _PRINCIPAL.search(question)


def _local_compute(question: str) -> str:
    normalized = "".join(question.split())
    currencies = {currency for currency in ("美元", "人民币", "欧元", "日元") if currency in normalized}
    if len(currencies) >= 2 and any(term in normalized for term in ("相加", "相减", "合计", "换算")):
        return "无法直接计算：不同币种必须先提供同一时点、同一报价方向的汇率及换算目标币种。"
    to_percent = re.search(r"(\d+(?:\.\d+)?)(?:转换|换算)(?:成|为)百分比", normalized)
    if to_percent:
        value = Decimal(to_percent.group(1)) * Decimal(100)
        return f"本地确定性换算结果：{_render_decimal(value)}%。"
    to_decimal = re.search(r"(\d+(?:\.\d+)?)%(?:转换|换算)(?:成|为)小数", normalized)
    if to_decimal:
        value = Decimal(to_decimal.group(1)) / Decimal(100)
        return f"本地确定性换算结果：{_render_decimal(value)}。"

    # Finite percentage word problems are deterministic only when every input
    # is explicitly present in the request.  They never consult market data.
    percentages = [Decimal(value) for value in re.findall(r"(-?\d+(?:\.\d+)?)%", normalized)]
    # A chained move ("先上涨20%，随后下跌20%") is a compounding question, not a
    # difference between two rates.  "相对初始值" must not be read as "compare
    # the two percentages", which is what the point-difference branch assumes.
    chained_move = _chained_percentage_move(normalized)
    if (
        len(percentages) == 2
        and chained_move is None
        and any(term in normalized for term in ("百分点", "相对", "高", "低", "比较"))
    ):
        point_difference = percentages[0] - percentages[1]
        relative = (percentages[0] / percentages[1] - Decimal(1)) * Decimal(100) if percentages[1] else None
        relative_text = "无法定义（比较基准为 0）" if relative is None else f"{_render_decimal(relative)}%"
        return (
            f"本地确定性计算结果：收益率相差{_render_decimal(point_difference)}个百分点；"
            f"以前者相对后者计算，前者高{relative_text}。"
            "前者是两个收益率的绝对差，后者以乙的收益率为基准计算相对增幅，因此口径不同。"
        )
    if chained_move is not None:
        principal = (
            Decimal(_principal_match(normalized).group(1))
            if _principal_match(normalized)
            else Decimal(100)
        )
        final = principal
        for rate in chained_move:
            final *= Decimal(1) + rate / Decimal(100)
        change = (final / principal - Decimal(1)) * Decimal(100)
        described = "、".join(
            f"{'上涨' if rate > 0 else '下跌'}{_render_decimal(abs(rate))}%"
            for rate in chained_move
        )
        return (
            f"本地确定性计算结果：以初始值{_render_decimal(principal)}演示，"
            f"依次{described}后，"
            f"最终值={_render_decimal(final)}；相对初始值变化={_render_decimal(change)}%。"
        )
    principal_match = _principal_match(normalized)
    if principal_match and percentages and any(term in normalized for term in ("复利", "连续", "累计", "最终")):
        principal = Decimal(principal_match.group(1))
        unit = "万元" if re.search(
            r"(?:本金|初始金额|起始金额|投入|初始值|起初|演示)\d+(?:\.\d+)?万元", normalized
        ) else ""
        final = principal
        for rate in percentages:
            final *= Decimal(1) + rate / Decimal(100)
        cumulative = (final / principal - Decimal(1)) * Decimal(100)
        return (
            f"本地确定性计算结果：期末值={_render_decimal(final)}{unit}；"
            f"累计收益率={_render_decimal(cumulative)}%。"
        )
    if len(percentages) >= 2 and any(term in normalized for term in ("平均", "平均收益率")):
        average = sum(percentages) / Decimal(len(percentages))
        if any(term in normalized for term in ("累计", "复利")):
            cumulative_factor = Decimal(1)
            for rate in percentages:
                cumulative_factor *= Decimal(1) + rate / Decimal(100)
            cumulative = (cumulative_factor - Decimal(1)) * Decimal(100)
            return (
                f"本地确定性计算结果：算术平均收益率={_render_decimal(average)}%；"
                f"累计收益率={_render_decimal(cumulative)}%。"
                "前者逐年收益率直接平均，后者按各年收益率连乘，二者口径不同。"
            )
        return f"本地确定性计算结果：算术平均收益率={_render_decimal(average)}%。"
    move = re.search(r"(?:从|由)?(\d+(?:\.\d+)?)(?:元|点)?(?:先)?上涨(\d+(?:\.\d+)?)%.*?(?:再|后)?下跌(\d+(?:\.\d+)?)%", normalized)
    if move:
        initial, up, down = (Decimal(value) for value in move.groups())
        final = initial * (Decimal(1) + up / Decimal(100)) * (Decimal(1) - down / Decimal(100))
        change = (final / initial - Decimal(1)) * Decimal(100)
        return f"本地确定性计算结果：最终值={_render_decimal(final)}；累计变动={_render_decimal(change)}%。"

    expression = normalized
    for token in ("请计算", "计算", "等于多少", "是多少", "等于", "？", "?"):
        expression = expression.replace(token, "")
    expression = expression.replace("×", "*").replace("÷", "/")
    if not expression:
        raise ValueError("empty arithmetic expression")
    value = _evaluate_expression(expression)
    rendered = str(value)
    if isinstance(value, float) and value.is_integer():
        rendered = str(int(value))
    return f"本地确定性计算结果：{rendered}。"


def _render_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _evaluate_expression(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent is outside the local compute bound")
            value = _BINARY_OPERATORS[type(node.op)](left, right)
            if abs(value) > 10**100:
                raise ValueError("arithmetic result is outside the local compute bound")
            return value
        raise ValueError("expression contains unsupported syntax")

    return evaluate(tree)
