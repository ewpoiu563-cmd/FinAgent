"""Execute financial arithmetic from a bounded, evidence-referenced plan."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .financial_amounts import AmountSemanticError, MoneyAmount, amount_ratio, normalize_scale, subtract_money


class CalculationPlanError(ValueError):
    """A calculation plan is malformed, unsupported, or lacks compatible evidence."""


@dataclass(frozen=True)
class DeterministicCalculation:
    operation: str
    left: Mapping[str, Any]
    right: Mapping[str, Any]
    display_unit: str | None = None


@dataclass(frozen=True)
class CalculationResult:
    text: str
    evidence_ids: tuple[str, ...]


def execute_calculation(
    calculation: DeterministicCalculation,
    evidence: Mapping[str, Mapping[str, Any]],
) -> CalculationResult:
    left, left_ids = _resolve_operand(calculation.left, evidence)
    right, right_ids = _resolve_operand(calculation.right, evidence)
    operation = calculation.operation
    unit = calculation.display_unit
    try:
        if operation in {"add", "subtract"}:
            left_money, right_money = _money(left), _money(right)
            if left_money.currency != right_money.currency:
                raise AmountSemanticError("不同币种缺少汇率和汇率日期，不能相加或相减")
            target_unit = unit or left_money.display_unit
            if operation == "add":
                normalized_unit, multiplier = normalize_scale(target_unit)
                result = MoneyAmount(
                    value=(left_money.base_value + right_money.base_value) / multiplier,
                    currency=left_money.currency,
                    display_unit=normalized_unit,
                    multiplier=multiplier,
                    base_value=left_money.base_value + right_money.base_value,
                    original_text=f"{left_money.original_text} + {right_money.original_text}",
                )
                symbol = "+"
            else:
                result = subtract_money(left_money, right_money, display_unit=target_unit)
                symbol = "−"
            text = f"{_render_money(left_money)} {symbol} {_render_money(right_money)} = {_render_money(result)}"
        elif operation == "multiply":
            money, factor, money_on_left = _money_and_scalar(left, right)
            result = _scale_money(money, factor, unit)
            text = f"{_render_money(money)} × {_render_decimal(factor)} = {_render_money(result)}"
        elif operation == "divide":
            if isinstance(left, MoneyAmount) and isinstance(right, MoneyAmount):
                ratio = amount_ratio(left, right)
                text = f"{_render_money(left)} ÷ {_render_money(right)} = {_render_decimal(ratio)}"
            else:
                money, divisor, _ = _money_and_scalar(left, right, require_money_left=True)
                if divisor == 0:
                    raise CalculationPlanError("除数不能为 0")
                result = _scale_money(money, Decimal(1) / divisor, unit)
                text = f"{_render_money(money)} ÷ {_render_decimal(divisor)} = {_render_money(result)}"
        elif operation == "growth_rate":
            base, current = _money(left), _money(right)
            if base.currency != current.currency:
                raise AmountSemanticError("不同币种缺少汇率和汇率日期，不能计算增长率")
            if base.base_value == 0:
                raise CalculationPlanError("增长率的基期金额不能为 0")
            rate = (current.base_value - base.base_value) / base.base_value * Decimal(100)
            text = (
                f"({_render_money(current)} − {_render_money(base)}) ÷ {_render_money(base)} "
                f"= {_render_decimal(rate)}%"
            )
        else:
            raise CalculationPlanError(f"unsupported operation: {operation}")
    except AmountSemanticError as error:
        raise CalculationPlanError(str(error)) from error
    return CalculationResult(text=text, evidence_ids=tuple(dict.fromkeys(left_ids + right_ids)))


def parse_calculation(record: Any) -> DeterministicCalculation:
    if not isinstance(record, Mapping):
        raise CalculationPlanError("calculation must be an object")
    allowed = {"operation", "left", "right", "display_unit"}
    if set(record).difference(allowed):
        raise CalculationPlanError("calculation contains unsupported fields")
    operation = record.get("operation")
    left, right = record.get("left"), record.get("right")
    display_unit = record.get("display_unit")
    if operation not in {"add", "subtract", "multiply", "divide", "growth_rate"}:
        raise CalculationPlanError("unsupported calculation operation")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise CalculationPlanError("calculation operands must be objects")
    if display_unit is not None:
        if not isinstance(display_unit, str):
            raise CalculationPlanError("display_unit must be a string or null")
        # growth_rate already renders a percentage.  Models commonly emit
        # display_unit="%" for that operation; accept the equivalent intent
        # instead of treating percent as a monetary scale.
        if operation == "growth_rate" and display_unit in {"%", "％"}:
            display_unit = None
        else:
            normalize_scale(display_unit)
    return DeterministicCalculation(operation, dict(left), dict(right), display_unit)


def requires_deterministic_calculation(question: str) -> bool:
    return any(token in question for token in (
        "计算", "相加", "加上", "相减", "减去", "相差", "差额", "差值", "合计", "总和", "之和",
        "乘以", "相乘", "除以", "相除", "比值", "增长率", "增幅", "同比", "环比", "百分点",
    ))


def _resolve_operand(
    operand: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[MoneyAmount | Decimal, tuple[str, ...]]:
    if set(operand) == {"literal"}:
        try:
            return Decimal(str(operand["literal"])), ()
        except (InvalidOperation, ValueError) as error:
            raise CalculationPlanError("literal must be a finite decimal") from error
    if set(operand) != {"evidence_id", "amount_index"}:
        raise CalculationPlanError("operand must be a literal or an evidence amount reference")
    evidence_id = operand["evidence_id"]
    index = operand["amount_index"]
    if not isinstance(evidence_id, str) or type(index) is not int or index < 0:
        raise CalculationPlanError("invalid evidence amount reference")
    item = evidence.get(evidence_id)
    values = item.get("monetary_values") if item else None
    if not isinstance(values, list) or index >= len(values) or not isinstance(values[index], Mapping):
        raise CalculationPlanError(f"evidence {evidence_id} has no monetary_values[{index}]")
    value = values[index]
    try:
        amount = MoneyAmount(
            value=Decimal(str(value["value"])),
            currency=str(value["currency"]),
            display_unit=str(value["display_unit"]),
            multiplier=Decimal(str(value["multiplier"])),
            base_value=Decimal(str(value["base_value"])),
            original_text=str(value.get("original_text", evidence_id)),
        )
    except (KeyError, InvalidOperation, ValueError) as error:
        raise CalculationPlanError(f"evidence {evidence_id} has invalid monetary value") from error
    return amount, (evidence_id,)


def _money(value: MoneyAmount | Decimal) -> MoneyAmount:
    if not isinstance(value, MoneyAmount):
        raise CalculationPlanError("operation requires an evidence monetary amount")
    return value


def _money_and_scalar(
    left: MoneyAmount | Decimal,
    right: MoneyAmount | Decimal,
    *,
    require_money_left: bool = False,
) -> tuple[MoneyAmount, Decimal, bool]:
    if isinstance(left, MoneyAmount) and isinstance(right, Decimal):
        return left, right, True
    if not require_money_left and isinstance(right, MoneyAmount) and isinstance(left, Decimal):
        return right, left, False
    raise CalculationPlanError("operation requires one evidence monetary amount and one literal")


def _scale_money(amount: MoneyAmount, factor: Decimal, display_unit: str | None) -> MoneyAmount:
    unit = display_unit or amount.display_unit
    normalized_unit, multiplier = normalize_scale(unit)
    base_value = amount.base_value * factor
    return MoneyAmount(
        value=base_value / multiplier,
        currency=amount.currency,
        display_unit=normalized_unit,
        multiplier=multiplier,
        base_value=base_value,
        original_text=amount.original_text,
    )


def _render_money(amount: MoneyAmount) -> str:
    unit = "" if amount.display_unit == "base" else amount.display_unit
    currency_name = {"CNY": "人民币", "USD": "美元"}.get(amount.currency, amount.currency)
    return f"{_render_decimal(amount.value)} {unit}{currency_name}"


def _render_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
