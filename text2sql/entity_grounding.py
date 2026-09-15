"""Deterministic local financial entity grounding and semantic SQL literal guard."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .schema_provider import DEFAULT_DB_PATH


@dataclass(frozen=True)
class GroundedFinancialEntity:
    entity_type: str
    name: str
    code: str
    matched_text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "entity_type": self.entity_type,
            "name": self.name,
            "code": self.code,
            "matched_text": self.matched_text,
        }


@dataclass(frozen=True)
class FinancialEntityGrounding:
    entities: tuple[GroundedFinancialEntity, ...] = ()

    @property
    def matched(self) -> bool:
        return bool(self.entities)

    def augment_question(self, question: str) -> str:
        if not self.entities:
            return question
        lines = [
            question.strip(),
            "",
            "[LOCAL_DATABASE_ENTITY_GROUNDING]",
            "下列名称和代码由本地 finance.db 确定。SQL 必须使用这些精确值，不得猜测或替换代码：",
        ]
        for entity in self.entities:
            lines.append(
                f"- entity_type={entity.entity_type}; name={entity.name}; code={entity.code}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SemanticLiteralValidation:
    valid: bool
    error: str | None = None


def ground_financial_entities(
    question: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> FinancialEntityGrounding:
    """Resolve stock/fund names or explicit six-digit codes from finance.db read-only."""

    if not isinstance(question, str) or not question.strip():
        return FinancialEntityGrounding()
    connection = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        matches: list[GroundedFinancialEntity] = []
        matches.extend(_match_stock_entities(connection, question))
        matches.extend(_match_fund_entities(connection, question))
        explicit_codes = set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", question))
        if explicit_codes:
            matches.extend(_match_explicit_codes(connection, explicit_codes))
    finally:
        connection.close()

    unique: dict[tuple[str, str, str], GroundedFinancialEntity] = {}
    for entity in matches:
        unique[(entity.entity_type, entity.name, entity.code)] = entity
    return FinancialEntityGrounding(tuple(unique.values()))


def validate_semantic_literals(
    sql: str,
    question: str,
    grounding: FinancialEntityGrounding,
) -> SemanticLiteralValidation:
    """Reject stock/fund identity literals not supported by input or local grounding."""

    if not isinstance(sql, str) or not sql.strip():
        return SemanticLiteralValidation(False, "SQL is empty")
    allowed = {
        "stock_code": {e.code for e in grounding.entities if e.entity_type == "stock"},
        "stock_name": {e.name for e in grounding.entities if e.entity_type == "stock"},
        "fund_code": {e.code for e in grounding.entities if e.entity_type == "fund"},
        "fund_name": {e.name for e in grounding.entities if e.entity_type == "fund"},
    }
    question_folded = question.casefold()
    checks = (
        ("stock_code", ("股票代码", "对应股票代码")),
        ("stock_name", ("股票名称",)),
        ("fund_code", ("基金代码",)),
        ("fund_name", ("基金简称", "基金全称")),
    )
    for kind, columns in checks:
        literals: list[str] = []
        for column in columns:
            literals.extend(_identity_literals_for_column(sql, column))
        for literal in literals:
            normalized = literal.strip().strip("%_")
            if not normalized:
                continue
            if allowed[kind]:
                if normalized in allowed[kind]:
                    continue
            elif normalized.casefold() in question_folded:
                continue
            return SemanticLiteralValidation(
                False,
                f"semantic literal guard rejected unsupported {kind} literal: {normalized}",
            )
    return SemanticLiteralValidation(True)


def _match_stock_entities(
    connection: sqlite3.Connection,
    question: str,
) -> list[GroundedFinancialEntity]:
    rows = connection.execute(
        "SELECT DISTINCT [股票名称], [股票代码] FROM [基金股票持仓明细] "
        "WHERE length(trim([股票名称])) >= 2 AND instr(?, trim([股票名称])) > 0 "
        "ORDER BY length([股票名称]) DESC, [股票名称], [股票代码]",
        (question,),
    ).fetchall()
    return [
        GroundedFinancialEntity("stock", str(name), str(code), str(name))
        for name, code in rows
        if name and code
    ]


def _match_fund_entities(
    connection: sqlite3.Connection,
    question: str,
) -> list[GroundedFinancialEntity]:
    rows = connection.execute(
        "SELECT DISTINCT [基金简称], [基金全称], [基金代码] FROM [基金基本信息] "
        "WHERE (length(trim([基金简称])) >= 2 AND instr(?, trim([基金简称])) > 0) "
        "OR (length(trim([基金全称])) >= 2 AND instr(?, trim([基金全称])) > 0) "
        "ORDER BY length(coalesce([基金全称], [基金简称])) DESC, [基金代码]",
        (question, question),
    ).fetchall()
    matches = []
    for short_name, full_name, code in rows:
        name = full_name or short_name
        matched = full_name if full_name and full_name in question else short_name
        if name and code and matched:
            matches.append(GroundedFinancialEntity("fund", str(name), str(code), str(matched)))
    return matches


def _match_explicit_codes(
    connection: sqlite3.Connection,
    codes: set[str],
) -> list[GroundedFinancialEntity]:
    placeholders = ",".join("?" for _ in codes)
    ordered_codes = tuple(sorted(codes))
    stock_rows = connection.execute(
        f"SELECT DISTINCT [股票名称], [股票代码] FROM [基金股票持仓明细] "
        f"WHERE [股票代码] IN ({placeholders}) ORDER BY [股票代码], [股票名称]",
        ordered_codes,
    ).fetchall()
    known_stock_codes = {str(code) for _, code in stock_rows if code}
    market_rows = connection.execute(
        f"SELECT [股票代码] FROM [A股票日行情表] WHERE [股票代码] IN ({placeholders}) "
        f"UNION SELECT [股票代码] FROM [港股票日行情表] WHERE [股票代码] IN ({placeholders}) "
        f"UNION SELECT [股票代码] FROM [A股公司行业划分表] WHERE [股票代码] IN ({placeholders}) "
        f"UNION SELECT [对应股票代码] FROM [基金可转债持仓明细] WHERE [对应股票代码] IN ({placeholders})",
        ordered_codes * 4,
    ).fetchall()
    fund_rows = connection.execute(
        f"SELECT DISTINCT coalesce([基金全称], [基金简称]), [基金代码] FROM [基金基本信息] "
        f"WHERE [基金代码] IN ({placeholders}) ORDER BY [基金代码]",
        ordered_codes,
    ).fetchall()
    return [
        GroundedFinancialEntity("stock", str(name), str(code), str(code))
        for name, code in stock_rows
        if name and code
    ] + [
        GroundedFinancialEntity("stock", str(code), str(code), str(code))
        for (code,) in market_rows
        if code and str(code) not in known_stock_codes
    ] + [
        GroundedFinancialEntity("fund", str(name), str(code), str(code))
        for name, code in fund_rows
        if name and code
    ]


def _identity_literals_for_column(sql: str, column: str) -> list[str]:
    identifier = rf"(?:\[{re.escape(column)}\]|\b{re.escape(column)}\b)"
    comparison = re.compile(
        identifier
        + r'''\s*(?:=|==|LIKE)\s*(?:'((?:''|[^'])*)'|"((?:""|[^"])*)"|(?<!\d)(\d{5,6})(?!\d))''',
        re.IGNORECASE,
    )
    in_clause = re.compile(
        identifier + r"\s+IN\s*\(([^)]*)\)",
        re.IGNORECASE | re.DOTALL,
    )
    literals = []
    for single_quoted, double_quoted, numeric in comparison.findall(sql):
        value = single_quoted or double_quoted.replace('""', '"') or numeric
        literals.append(value.replace("''", "'"))
    for clause in in_clause.findall(sql):
        literals.extend(value.replace("''", "'") for value in re.findall(r"'((?:''|[^'])*)'", clause))
        literals.extend(value.replace('""', '"') for value in re.findall(r'"((?:""|[^"])*)"', clause))
        literals.extend(re.findall(r"(?<!\d)\d{5,6}(?!\d)", clause))
    return literals
