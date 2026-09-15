"""Validated business semantics for fields exposed by the local finance DB."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .schema_provider import DEFAULT_DB_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEMANTIC_CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "financial_schema_semantics.json"
_DATE_PATTERNS = {
    "YYYYMMDD": re.compile(r"^\d{8}$"),
    "YYYY-MM-DD HH:MM:SS": re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
}


@dataclass(frozen=True)
class SemanticField:
    semantic_id: str
    table: str
    column: str
    sqlite_type: str
    business_meaning: str = ""
    synonyms: tuple[str, ...] = ()
    storage_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "table": self.table,
            "column": self.column,
            "sqlite_type": self.sqlite_type,
            "business_meaning": self.business_meaning,
            "synonyms": list(self.synonyms),
            "storage_format": self.storage_format,
        }


@dataclass(frozen=True)
class SchemaSemanticCatalog:
    metrics: Mapping[str, SemanticField]
    dimensions: Mapping[str, SemanticField]
    temporal_fields: Mapping[str, SemanticField]
    database: str
    verified_at: str

    def metric(self, semantic_id: str) -> SemanticField | None:
        return self.metrics.get(semantic_id)

    def temporal_for_table(self, table: str) -> SemanticField | None:
        preferred = {
            "基金股票持仓明细": "holding_date",
            "基金债券持仓明细": "bond_holding_date",
            "基金可转债持仓明细": "convertible_holding_date",
            "基金日行情表": "trading_date",
            "基金规模变动表": "scale_period_end",
            "基金份额持有人结构": "holder_period_end",
            "基金基本信息": "fund_inception_date",
            "A股票日行情表": "a_stock_trading_date",
            "港股票日行情表": "hk_stock_trading_date",
            "A股公司行业划分表": "stock_industry_date",
        }.get(table)
        return self.temporal_fields.get(preferred) if preferred else None


def _field(semantic_id: str, value: Any) -> SemanticField:
    if not isinstance(value, dict):
        raise ValueError(f"semantic field {semantic_id!r} must be an object")
    for required in ("table", "column", "sqlite_type"):
        if not isinstance(value.get(required), str) or not value[required].strip():
            raise ValueError(f"semantic field {semantic_id!r}.{required} must be a non-empty string")
    synonyms = value.get("synonyms", [])
    if not isinstance(synonyms, list) or any(not isinstance(item, str) for item in synonyms):
        raise ValueError(f"semantic field {semantic_id!r}.synonyms must be a list[str]")
    return SemanticField(
        semantic_id=semantic_id,
        table=value["table"],
        column=value["column"],
        sqlite_type=value["sqlite_type"].upper(),
        business_meaning=str(value.get("business_meaning") or ""),
        synonyms=tuple(synonyms),
        storage_format=value.get("storage_format"),
    )


def _section(payload: Mapping[str, Any], name: str) -> dict[str, SemanticField]:
    raw = payload.get(name)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"semantic catalog {name!r} must be a non-empty object")
    return {semantic_id: _field(semantic_id, value) for semantic_id, value in raw.items()}


@lru_cache(maxsize=4)
def load_schema_semantic_catalog(
    path: str | Path = DEFAULT_SEMANTIC_CATALOG_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> SchemaSemanticCatalog:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported financial schema semantic catalog")
    catalog = SchemaSemanticCatalog(
        metrics=_section(payload, "metrics"),
        dimensions=_section(payload, "dimensions"),
        temporal_fields=_section(payload, "temporal_fields"),
        database=str(payload.get("database") or ""),
        verified_at=str(payload.get("verified_at") or ""),
    )
    _validate_against_database(catalog, Path(db_path))
    return catalog


def _validate_against_database(catalog: SchemaSemanticCatalog, db_path: Path) -> None:
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        all_fields = [*catalog.metrics.values(), *catalog.dimensions.values(), *catalog.temporal_fields.values()]
        schemas: dict[str, dict[str, str]] = {}
        for field in all_fields:
            if field.table not in schemas:
                schemas[field.table] = {
                    str(row[1]): str(row[2]).upper()
                    for row in connection.execute(f"PRAGMA table_info([{field.table}])")
                }
            actual_type = schemas[field.table].get(field.column)
            if actual_type is None:
                raise ValueError(f"semantic catalog column is absent: {field.table}.{field.column}")
            if actual_type != field.sqlite_type:
                raise ValueError(
                    f"semantic catalog type mismatch for {field.table}.{field.column}: "
                    f"catalog={field.sqlite_type}, database={actual_type}"
                )
        for field in catalog.temporal_fields.values():
            pattern = _DATE_PATTERNS.get(field.storage_format or "")
            if pattern is None:
                raise ValueError(f"unsupported date storage format: {field.storage_format!r}")
            values = connection.execute(
                f"SELECT DISTINCT [{field.column}] FROM [{field.table}] "
                f"WHERE [{field.column}] IS NOT NULL LIMIT 1000"
            ).fetchall()
            if any(not pattern.fullmatch(str(row[0])) for row in values):
                raise ValueError(
                    f"database values do not match {field.storage_format}: {field.table}.{field.column}"
                )
    finally:
        connection.close()
