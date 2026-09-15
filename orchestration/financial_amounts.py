"""Deterministic currency and amount-scale semantics for financial evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


_CURRENCY_ALIASES = {
    "人民币": "CNY",
    "人民币元": "CNY",
    "RMB": "CNY",
    "CNY": "CNY",
    "￥": "CNY",
    "¥": "CNY",
    "美元": "USD",
    "美金": "USD",
    "USD": "USD",
    "US$": "USD",
    "$": "USD",
}

_SCALE_ALIASES = {
    "base": ("base", Decimal("1")),
    "千": ("千", Decimal("1000")),
    "万": ("万", Decimal("10000")),
    "百万": ("百万", Decimal("1000000")),
    "亿": ("亿", Decimal("100000000")),
    "元": ("元", Decimal("1")),
    "千元": ("千元", Decimal("1000")),
    "万元": ("万元", Decimal("10000")),
    "百万元": ("百万元", Decimal("1000000")),
    "亿元": ("亿元", Decimal("100000000")),
    "thousand": ("thousand", Decimal("1000")),
    "million": ("million", Decimal("1000000")),
    "billion": ("billion", Decimal("1000000000")),
    "k": ("thousand", Decimal("1000")),
    "m": ("million", Decimal("1000000")),
    "bn": ("billion", Decimal("1000000000")),
}

_CURRENCY_TOKEN = r"人民币元|人民币|RMB|CNY|美元|美金|USD|US\$|[￥¥$]"
_SCALE_TOKEN = r"亿元|百万元|万元|千元|百万|亿|万|千|元|billion|million|thousand|bn|[km]"
_AMOUNT_PATTERN = re.compile(
    rf"(?P<prefix>{_CURRENCY_TOKEN})?\s*"
    r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    rf"(?P<scale>{_SCALE_TOKEN})?\s*(?P<suffix>{_CURRENCY_TOKEN})?",
    re.IGNORECASE,
)
_PARENTHETICAL_UNIT = re.compile(
    rf"[（(]\s*(?:(?P<currency>{_CURRENCY_TOKEN})\s*)?(?P<scale>{_SCALE_TOKEN})\s*[）)]",
    re.IGNORECASE,
)
_MONEY_COLUMN = re.compile(
    r"金额|收入|营收|利润|成本|费用|现金|资产净值|市值|价格|估值|规模|余额|薪酬|报酬|分红|股息|融资",
    re.IGNORECASE,
)
_CURRENCY_COLUMN = re.compile(r"币种|currency", re.IGNORECASE)
_UNIT_COLUMN = re.compile(r"金额单位|货币单位|计量单位|^单位$|unit", re.IGNORECASE)


class AmountSemanticError(ValueError):
    """Raised when an amount operation would silently mix incompatible semantics."""


@dataclass(frozen=True)
class MoneyAmount:
    value: Decimal
    currency: str
    display_unit: str
    multiplier: Decimal
    base_value: Decimal
    original_text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "value": _decimal_text(self.value),
            "currency": self.currency,
            "display_unit": self.display_unit,
            "multiplier": _decimal_text(self.multiplier),
            "base_value": _decimal_text(self.base_value),
            "original_text": self.original_text,
        }

    def convert_to(self, display_unit: str) -> "MoneyAmount":
        unit, multiplier = normalize_scale(display_unit)
        return MoneyAmount(
            value=self.base_value / multiplier,
            currency=self.currency,
            display_unit=unit,
            multiplier=multiplier,
            base_value=self.base_value,
            original_text=self.original_text,
        )


@dataclass(frozen=True)
class ExchangeRate:
    """A dated quote: one base-currency unit equals ``rate`` quote-currency units."""

    base_currency: str
    quote_currency: str
    rate: Decimal
    rate_date: str

    def __post_init__(self) -> None:
        if self.base_currency == self.quote_currency:
            raise AmountSemanticError("exchange-rate currencies must differ")
        if self.rate <= 0:
            raise AmountSemanticError("exchange rate must be positive")
        if not self.rate_date.strip():
            raise AmountSemanticError("exchange rate requires a rate date")


def normalize_currency(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    token = value.strip()
    return _CURRENCY_ALIASES.get(token) or _CURRENCY_ALIASES.get(token.upper())


def normalize_scale(value: str | None) -> tuple[str, Decimal]:
    if not isinstance(value, str) or not value.strip():
        return "base", Decimal("1")
    token = value.strip()
    normalized = _SCALE_ALIASES.get(token) or _SCALE_ALIASES.get(token.casefold())
    if normalized is None:
        raise AmountSemanticError(f"unsupported amount scale: {value}")
    return normalized


def extract_money_amounts(text: str) -> tuple[MoneyAmount, ...]:
    """Extract explicit monetary amounts; bare numbers are intentionally ignored."""
    if not isinstance(text, str) or not text:
        return ()
    amounts: list[MoneyAmount] = []
    for match in _AMOUNT_PATTERN.finditer(text):
        prefix, suffix, scale_token = match.group("prefix", "suffix", "scale")
        if not prefix and not suffix and not scale_token:
            continue
        currency = normalize_currency(prefix) or normalize_currency(suffix)
        if currency is None and scale_token and scale_token.endswith("元"):
            currency = "CNY"
        if currency is None:
            continue
        display_unit, multiplier = normalize_scale(scale_token)
        try:
            value = Decimal(match.group("number").replace(",", ""))
        except InvalidOperation:
            continue
        original = match.group(0).strip()
        amounts.append(MoneyAmount(value, currency, display_unit, multiplier, value * multiplier, original))
    return tuple(amounts)


def extract_sql_money_amounts(rows: tuple[Mapping[str, Any], ...]) -> tuple[MoneyAmount, ...]:
    """Extract amounts from SQL cells using explicit row/column currency and unit metadata."""
    amounts: list[MoneyAmount] = []
    for row in rows:
        row_currency = next(
            (normalize_currency(str(value)) for key, value in row.items() if _CURRENCY_COLUMN.search(str(key))),
            None,
        )
        row_unit = next(
            (str(value).strip() for key, value in row.items() if _UNIT_COLUMN.search(str(key)) and value is not None),
            None,
        )
        for key, raw_value in row.items():
            column = str(key)
            if _CURRENCY_COLUMN.search(column) or _UNIT_COLUMN.search(column):
                continue
            explicit = extract_money_amounts(str(raw_value)) if isinstance(raw_value, str) else ()
            if explicit:
                amounts.extend(explicit)
                continue
            unit_match = _PARENTHETICAL_UNIT.search(column)
            column_currency = normalize_currency(unit_match.group("currency")) if unit_match else None
            unit_token = unit_match.group("scale") if unit_match else row_unit
            currency = column_currency or row_currency
            if currency is None and isinstance(unit_token, str) and unit_token.endswith("元"):
                currency = "CNY"
            if currency is None or not (_MONEY_COLUMN.search(column) or unit_match):
                continue
            try:
                value = Decimal(str(raw_value).replace(",", ""))
                display_unit, multiplier = normalize_scale(unit_token)
            except (InvalidOperation, AmountSemanticError, AttributeError):
                continue
            original = f"{column}={raw_value}" + (f" {unit_token}" if unit_token else "")
            amounts.append(MoneyAmount(value, currency, display_unit, multiplier, value * multiplier, original))
    return tuple(amounts)


def subtract_money(left: MoneyAmount, right: MoneyAmount, *, display_unit: str | None = None) -> MoneyAmount:
    if left.currency != right.currency:
        raise AmountSemanticError(
            f"cannot subtract {left.currency} and {right.currency} without an exchange rate and rate date"
        )
    unit = display_unit or left.display_unit
    normalized_unit, multiplier = normalize_scale(unit)
    base_value = left.base_value - right.base_value
    return MoneyAmount(
        value=base_value / multiplier,
        currency=left.currency,
        display_unit=normalized_unit,
        multiplier=multiplier,
        base_value=base_value,
        original_text=f"{left.original_text} - {right.original_text}",
    )


def convert_currency(
    amount: MoneyAmount,
    target_currency: str,
    exchange_rate: ExchangeRate,
    *,
    display_unit: str = "base",
) -> MoneyAmount:
    target = normalize_currency(target_currency) or target_currency.strip().upper()
    if amount.currency == target:
        return amount.convert_to(display_unit)
    if amount.currency == exchange_rate.base_currency and target == exchange_rate.quote_currency:
        base_value = amount.base_value * exchange_rate.rate
    elif amount.currency == exchange_rate.quote_currency and target == exchange_rate.base_currency:
        base_value = amount.base_value / exchange_rate.rate
    else:
        raise AmountSemanticError("exchange rate does not cover the requested currency pair")
    unit, multiplier = normalize_scale(display_unit)
    return MoneyAmount(
        value=base_value / multiplier,
        currency=target,
        display_unit=unit,
        multiplier=multiplier,
        base_value=base_value,
        original_text=(
            f"{amount.original_text}; FX {exchange_rate.base_currency}/{exchange_rate.quote_currency}="
            f"{_decimal_text(exchange_rate.rate)} @ {exchange_rate.rate_date}"
        ),
    )


def amount_ratio(numerator: MoneyAmount, denominator: MoneyAmount) -> Decimal:
    if numerator.currency != denominator.currency:
        raise AmountSemanticError(
            f"cannot divide {numerator.currency} by {denominator.currency} without an exchange rate and rate date"
        )
    if denominator.base_value == 0:
        raise AmountSemanticError("cannot divide by a zero monetary amount")
    return numerator.base_value / denominator.base_value


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
