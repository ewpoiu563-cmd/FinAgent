"""Deterministic agreement checks between confirmed SQLIntent and generated SQL."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .semantic_planner import SQLIntent


@dataclass(frozen=True)
class SQLSemanticValidation:
    valid: bool
    errors: tuple[str, ...] = ()

    @property
    def error(self) -> str | None:
        return "; ".join(self.errors) if self.errors else None


def validate_sql_against_intent(sql: str, intent: SQLIntent) -> SQLSemanticValidation:
    if not isinstance(sql, str) or not sql.strip():
        return SQLSemanticValidation(False, ("generated SQL is empty",))
    errors: list[str] = []
    folded = sql.casefold()

    if intent.metric_column:
        metric_identifier = _identifier(intent.metric_column)
        if not re.search(metric_identifier, sql, re.IGNORECASE):
            errors.append(f"metric mismatch: expected column {intent.metric_column}")
        if intent.aggregation in {"max", "min", "avg", "sum"}:
            function = {"max": "MAX", "min": "MIN", "avg": "AVG", "sum": "SUM"}[intent.aggregation]
            if not re.search(rf"\b{function}\s*\(\s*(?:\w+\.)?{metric_identifier}\s*\)", sql, re.IGNORECASE):
                errors.append(f"aggregation mismatch: expected {function}({intent.metric_column})")

    for column in intent.select_columns:
        if not re.search(_identifier(column), sql, re.IGNORECASE):
            errors.append(f"selected column mismatch: expected {column}")

    for entity in intent.grounded_entities:
        if not _has_entity_predicate(sql, entity.entity_type, entity.code, entity.name):
            errors.append(f"grounded entity missing from SQL: {entity.name}/{entity.code}")

    temporal = intent.temporal_constraint
    if temporal:
        if not re.search(_identifier(temporal.field), sql, re.IGNORECASE):
            errors.append(f"temporal field mismatch: expected {temporal.field}")
        if temporal.precision == "quarter":
            start, end = temporal.storage_value.split("-", 1)
            if start not in sql or end not in sql:
                errors.append(f"temporal constraint missing: expected quarter {temporal.value}")
        elif temporal.storage_value not in sql:
            errors.append(f"temporal constraint missing: expected {temporal.storage_value}")
        if intent.temporal_semantics == "period_end" and temporal.precision == "year":
            year_end_values = (f"{temporal.value}1231", f"{temporal.value}-12-31 00:00:00")
            if not any(value in sql for value in year_end_values):
                errors.append("temporal semantics mismatch: period_end requires an explicit year-end date")

    for item in intent.filters:
        if item.field in {"股票代码", "对应股票代码", "基金代码", "基金类型", "报告类型"}:
            if not re.search(_identifier(item.field), sql, re.IGNORECASE) or item.value not in sql:
                errors.append(f"filter mismatch: expected {item.field}={item.value}")

    if intent.group_by:
        group_match = re.search(r"\bGROUP\s+BY\b(.+?)(?:\bORDER\s+BY\b|\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL)
        if not group_match:
            errors.append("group_by mismatch: GROUP BY is required")
        else:
            group_sql = group_match.group(1)
            for field in intent.group_by:
                if not re.search(_identifier(field), group_sql, re.IGNORECASE):
                    errors.append(f"group_by mismatch: expected {field}")

    for ordering in intent.ordering:
        order_match = re.search(r"\bORDER\s+BY\b(.+?)(?:\bLIMIT\b|$)", sql, re.IGNORECASE | re.DOTALL)
        if not order_match or not re.search(_identifier(ordering.field), order_match.group(1), re.IGNORECASE):
            errors.append(f"ordering mismatch: expected {ordering.field} {ordering.direction}")
        elif ordering.direction.casefold() not in order_match.group(1).casefold():
            errors.append(f"ordering direction mismatch: expected {ordering.direction}")

    if (
        intent.top_k is not None
        and intent.top_k_scope == "global_top_k"
        and not (intent.target_domain == "fund_holdings" and intent.aggregation == "sum")
    ):
        if not re.search(rf"\bLIMIT\s+{intent.top_k}\b", folded, re.IGNORECASE):
            errors.append(f"global_top_k mismatch: expected LIMIT {intent.top_k}")
    elif intent.top_k is not None and intent.top_k_scope == "per_group_top_k":
        rank_filter = re.search(
            rf"{_identifier('第N大重仓股')}\s*<=\s*{intent.top_k}\b",
            sql,
            re.IGNORECASE,
        )
        window_rank = re.search(
            r"ROW_NUMBER\s*\(\s*\)\s*OVER\s*\(\s*PARTITION\s+BY\b",
            sql,
            re.IGNORECASE,
        )
        if not rank_filter and not window_rank:
            errors.append(
                f"per_group_top_k mismatch: expected 第N大重仓股 <= {intent.top_k} "
                "or ROW_NUMBER() OVER (PARTITION BY ...)"
            )

    if intent.result_shape == "time_series":
        limit = re.search(r"\bLIMIT\s+(\d+)\b", folded, re.IGNORECASE)
        temporal = intent.temporal_constraint
        minimum_safe_limit = 31 if temporal and temporal.precision == "month" else 366
        if limit and int(limit.group(1)) < minimum_safe_limit:
            errors.append(
                f"time_series truncation risk: LIMIT must be at least {minimum_safe_limit} "
                "for the requested complete period"
            )

    return SQLSemanticValidation(not errors, tuple(errors))


def _identifier(column: str) -> str:
    return rf"(?:\[[^\]]*\]\.)?(?:\[{re.escape(column)}\]|\b{re.escape(column)}\b)"


def _has_entity_predicate(sql: str, entity_type: str, code: str, name: str) -> bool:
    fields_and_values = (
        (("股票代码", code), ("对应股票代码", code), ("股票名称", name))
        if entity_type == "stock"
        else (("基金代码", code), ("基金简称", name), ("基金全称", name))
    )
    for field, value in fields_and_values:
        identifier = _identifier(field)
        quoted_value = re.escape(value)
        comparison = rf"{identifier}\s*(?:=|==|LIKE)\s*['\"]?{quoted_value}['\"]?"
        in_clause = rf"{identifier}\s+IN\s*\([^)]*['\"]?{quoted_value}['\"]?[^)]*\)"
        if re.search(comparison, sql, re.IGNORECASE) or re.search(in_clause, sql, re.IGNORECASE):
            return True
    return False
