"""Rule-first SQL semantic intent planning over the verified finance schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from .entity_grounding import (
    FinancialEntityGrounding,
    GroundedFinancialEntity,
    ground_financial_entities,
)
from .schema_semantics import SchemaSemanticCatalog, SemanticField, load_schema_semantic_catalog


_CHINESE_NUMBERS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


@dataclass(frozen=True)
class SQLFilter:
    field: str
    operator: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "operator": self.operator, "value": self.value}


@dataclass(frozen=True)
class SQLOrdering:
    field: str
    direction: str

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "direction": self.direction}


@dataclass(frozen=True)
class TemporalConstraint:
    field: str
    precision: str
    value: str
    storage_value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "precision": self.precision,
            "value": self.value,
            "storage_value": self.storage_value,
        }


@dataclass(frozen=True)
class SQLIntent:
    target_entity: str
    target_domain: str
    grounded_entities: tuple[GroundedFinancialEntity, ...]
    metric: str | None
    metric_table: str | None
    metric_column: str | None
    aggregation: str | None
    filters: tuple[SQLFilter, ...]
    temporal_constraint: TemporalConstraint | None
    temporal_semantics: str | None
    group_by: tuple[str, ...]
    ordering: tuple[SQLOrdering, ...]
    top_k: int | None
    result_shape: str = "records"
    top_k_scope: str | None = None
    select_columns: tuple[str, ...] = ()
    ambiguity: tuple[str, ...] = ()
    clarification_needed: bool = False
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_entity": self.target_entity,
            "target_domain": self.target_domain,
            "grounded_entities": [entity.to_dict() for entity in self.grounded_entities],
            "metric": self.metric,
            "metric_table": self.metric_table,
            "metric_column": self.metric_column,
            "aggregation": self.aggregation,
            "filters": [item.to_dict() for item in self.filters],
            "temporal_constraint": self.temporal_constraint.to_dict() if self.temporal_constraint else None,
            "temporal_semantics": self.temporal_semantics,
            "group_by": list(self.group_by),
            "ordering": [item.to_dict() for item in self.ordering],
            "top_k": self.top_k,
            "result_shape": self.result_shape,
            "top_k_scope": self.top_k_scope,
            "select_columns": list(self.select_columns),
            "ambiguity": list(self.ambiguity),
            "clarification_needed": self.clarification_needed,
            "clarification_question": self.clarification_question,
        }

    def generator_payload(self, question: str) -> str:
        return json.dumps(
            {"original_question": question, "confirmed_sql_intent": self.to_dict()},
            ensure_ascii=False,
            separators=(",", ":"),
        )


class SQLSemanticPlanner:
    """Resolve critical metric/entity/time semantics before SQL generation."""

    def __init__(self, catalog: SchemaSemanticCatalog | None = None) -> None:
        self.catalog = catalog or load_schema_semantic_catalog()

    def plan(
        self,
        question: str,
        grounding: FinancialEntityGrounding | None = None,
    ) -> SQLIntent:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        normalized = "".join(question.split())
        grounding = grounding or ground_financial_entities(question)
        metric = self._metric(normalized)
        metric_field = self.catalog.metric(metric) if metric else None
        domain = self._domain(normalized, metric_field)
        relevant_grounding = self._relevant_grounding(normalized, domain, grounding)
        target_entity = "fund" if "基金" in normalized else "record"
        temporal = self._temporal(normalized, metric_field, domain)
        top_k = _top_k(normalized)
        result_shape, top_k_scope = self._result_shape(normalized, domain, top_k)
        temporal_semantics, ambiguity, clarification = self._temporal_semantics(
            normalized, metric, temporal, domain
        )
        if domain == "fund_scale_reports" and self._requests_monthly_snapshots(normalized):
            year = temporal.value if temporal is not None and temporal.precision == "year" else "所选年份"
            clarification = (
                f"当前数据库的基金规模变动表是定期报告粒度，不含{year}完整的逐月月末数据，"
                "无法得出每月末排名。是否改为查询各可用报告期末（3月、6月、9月和12月）？"
            )
        if temporal is not None and temporal.precision == "year" and temporal_semantics == "period_end":
            temporal = self._resolve_year_end(temporal)
        aggregation = self._aggregation(normalized, metric, temporal_semantics)
        group_by = self._group_by(normalized, target_entity, aggregation)
        if result_shape == "fund_profile_summary":
            group_by = ()
        filters = self._filters(normalized, relevant_grounding, temporal, domain)
        ordering = self._ordering(normalized, metric, metric_field, aggregation, group_by, result_shape)
        if result_shape == "time_series" and temporal is not None:
            ordering = (SQLOrdering(temporal.field, "asc"),)
        elif result_shape == "per_group_top_k":
            ordering = (SQLOrdering("持仓日期", "asc"), SQLOrdering("第N大重仓股", "asc"))
        return SQLIntent(
            target_entity=target_entity,
            target_domain=domain,
            grounded_entities=relevant_grounding.entities,
            metric=metric,
            metric_table=metric_field.table if metric_field else None,
            metric_column=metric_field.column if metric_field else None,
            aggregation=aggregation,
            filters=filters,
            temporal_constraint=temporal,
            temporal_semantics=temporal_semantics,
            group_by=group_by,
            ordering=ordering,
            top_k=top_k,
            result_shape=result_shape,
            top_k_scope=top_k_scope,
            select_columns=self._select_columns(normalized, domain),
            ambiguity=ambiguity,
            clarification_needed=bool(clarification),
            clarification_question=clarification,
        )

    def _resolve_year_end(self, temporal: TemporalConstraint) -> TemporalConstraint:
        field = next(
            (
                candidate
                for candidate in self.catalog.temporal_fields.values()
                if candidate.column == temporal.field
            ),
            None,
        )
        if field is None:
            raise ValueError(f"temporal field is absent from semantic catalog: {temporal.field}")
        iso = f"{temporal.value}-12-31"
        storage = iso.replace("-", "") if field.storage_format == "YYYYMMDD" else iso + " 00:00:00"
        return TemporalConstraint(temporal.field, "day", iso, storage)

    def _metric(self, question: str) -> str | None:
        if "可转债" in question:
            if any(signal in question for signal in ("净值占比", "持仓占比", "可转债占比")):
                return "convertible_holding_nav_ratio"
            if any(signal in question for signal in ("市值", "金额")):
                return "convertible_holding_market_value"
            if any(signal in question for signal in ("数量", "持有多少")):
                return "convertible_holding_quantity"
        if any(signal in question for signal in ("债券持仓", "持债", "债券名称", "债券类型")):
            if any(signal in question for signal in ("净值占比", "持仓占比", "持债占比")):
                return "bond_holding_nav_ratio"
            if "市值" in question:
                return "bond_holding_market_value"
            if "数量" in question:
                return "bond_holding_quantity"
        if any(signal in question for signal in ("机构投资者", "机构持有")):
            return "institutional_holder_ratio" if any(signal in question for signal in ("比例", "占比")) else "institutional_holder_shares"
        if any(signal in question for signal in ("个人投资者", "个人持有")):
            return "individual_holder_ratio" if any(signal in question for signal in ("比例", "占比")) else "individual_holder_shares"
        market_prefix = "hk_stock" if "港股" in question else "a_stock" if "A股" in question else None
        if market_prefix:
            for suffix, signals in (
                ("turnover", ("成交金额",)), ("volume", ("成交量",)),
                ("high", ("最高价",)), ("low", ("最低价",)),
                ("open", ("开盘价", "开盘")), ("close", ("收盘价", "收盘")),
            ):
                if any(signal in question for signal in signals):
                    return f"{market_prefix}_{suffix}"
        # A requested ranking by holding market value is not changed into a
        # NAV-ratio query merely because the response also asks for NAV ratio.
        if any(signal in question for signal in ("按该股票市值", "市值降序", "市值升序")):
            return "holding_market_value"
        precedence = (
            ("holding_nav_ratio", ("市值占基金资产净值比", "净值占比", "持仓占比")),
            ("holding_quantity", ("持有数量", "持仓数量", "持有多少股", "数量最多")),
            ("holding_market_value", ("持有市值", "持仓市值", "市值最高", "总市值")),
            ("period_end_shares", ("期末基金总份额", "期末总份额", "基金总份额")),
            ("asset_net_value", ("资产净值", "基金规模", "资产规模")),
            ("unit_net_value", ("单位净值",)),
        )
        for semantic_id, signals in precedence:
            # "数量最多" can describe a manager's fund count, not a holding
            # quantity.  Holding metrics require an explicit holding context.
            if semantic_id == "holding_quantity" and not any(
                marker in question for marker in ("持仓", "持有", "重仓股", "股票")
        ):
                continue
            if any(signal in question for signal in signals):
                return semantic_id
        if "市值合计" in question or "市值总和" in question:
            return "holding_market_value"
        if "持有" in question and "市值" in question:
            return "holding_market_value"
        if "持有" in question and any(signal in question for signal in ("数量", "最多")):
            return "holding_quantity"
        return None

    @staticmethod
    def _relevant_grounding(
        question: str,
        domain: str,
        grounding: FinancialEntityGrounding,
    ) -> FinancialEntityGrounding:
        entities = grounding.entities
        stock_only_domains = {"a_stock_daily_market", "hk_stock_daily_market", "stock_industry"}
        stock_aware_domains = {"fund_holdings", "fund_convertible_bond_holdings", *stock_only_domains}
        if domain in stock_only_domains:
            entities = tuple(item for item in entities if item.entity_type == "stock")
        elif domain not in stock_aware_domains:
            entities = tuple(item for item in entities if item.entity_type != "stock")
        if domain in {"fund_holdings", "fund_convertible_bond_holdings"}:
            explicit_fund_codes = set(re.findall(r"基金(?:代码)?[:：]?\s*(\d{6})", question))
            explicit_stock_codes = set(re.findall(
                r"(?:股票代码|对应股票代码|正股代码|股票)[:：]?\s*(\d{5,6})", question
            ))
            if explicit_fund_codes:
                entities = tuple(
                    item for item in entities
                    if item.entity_type != "stock" or item.code not in explicit_fund_codes
                )
            if explicit_stock_codes:
                entities = tuple(
                    item for item in entities
                    if item.entity_type != "fund" or item.code not in explicit_stock_codes
                )
        return FinancialEntityGrounding(entities)

    @staticmethod
    def _domain(question: str, metric: SemanticField | None) -> str:
        if metric is not None:
            return {
                "基金股票持仓明细": "fund_holdings",
                "基金债券持仓明细": "fund_bond_holdings",
                "基金可转债持仓明细": "fund_convertible_bond_holdings",
                "基金日行情表": "fund_daily_market",
                "基金规模变动表": "fund_scale_reports",
                "基金份额持有人结构": "fund_holder_structure",
                "基金基本信息": "fund_profile",
                "A股票日行情表": "a_stock_daily_market",
                "港股票日行情表": "hk_stock_daily_market",
            }.get(metric.table, "finance_db")
        if any(term in question for term in ("行业划分", "一级行业", "二级行业", "行业分类")):
            return "stock_industry"
        if "可转债" in question:
            return "fund_convertible_bond_holdings"
        if any(term in question for term in ("债券持仓", "持债", "债券名称", "债券类型")):
            return "fund_bond_holdings"
        if any(term in question for term in ("持有人结构", "机构投资者", "个人投资者", "机构持有", "个人持有", "机构占比", "个人占比")):
            return "fund_holder_structure"
        # These phrases name physical report domains.  They must win over the
        # generic word "规模" and over a bare annual-report filter.
        if "规模变动" in question:
            return "fund_scale_reports"
        if "年报(含半年报)" in question and any(
            term in question for term in ("持仓记录", "不同基金数", "重仓股")
        ):
            return "fund_holdings"
        if "港股" in question and any(term in question for term in ("行情", "开盘", "收盘", "最高价", "最低价", "成交量", "成交金额")):
            return "hk_stock_daily_market"
        if any(term in question for term in ("A股行情", "A股开盘", "A股收盘", "A股最高价", "A股最低价", "A股成交")):
            return "a_stock_daily_market"
        if any(term in question for term in ("持仓", "重仓股", "股票", "持有")):
            return "fund_holdings"
        if any(term in question for term in ("行情", "净值", "规模", "交易日")):
            return "fund_daily_market"
        return "fund_profile"

    def _temporal(
        self,
        question: str,
        metric: SemanticField | None,
        domain: str,
    ) -> TemporalConstraint | None:
        table = metric.table if metric else {
            "fund_holdings": "基金股票持仓明细",
            "fund_bond_holdings": "基金债券持仓明细",
            "fund_convertible_bond_holdings": "基金可转债持仓明细",
            "fund_daily_market": "基金日行情表",
            "fund_scale_reports": "基金规模变动表",
            "fund_holder_structure": "基金份额持有人结构",
            "fund_profile": "基金基本信息",
            "a_stock_daily_market": "A股票日行情表",
            "hk_stock_daily_market": "港股票日行情表",
            "stock_industry": "A股公司行业划分表",
        }.get(domain)
        temporal = self.catalog.temporal_for_table(table or "")
        if temporal is None:
            return None
        def day_constraint(year: int, month: int, day: int) -> TemporalConstraint:
            iso = date(year, month, day).isoformat()
            storage = iso.replace("-", "") if temporal.storage_format == "YYYYMMDD" else iso + " 00:00:00"
            return TemporalConstraint(temporal.column, "day", iso, storage)

        # Parse machine-style exact dates before years.  Previously 20190110
        # was accepted as the year 2019, producing a broad query and often an
        # unnecessary clarification request.
        timestamp = re.search(
            r"(?<!\d)((?:19|20)\d{2})-(\d{2})-(\d{2})(?:[ T]?\d{2}:\d{2}:\d{2})?(?!\d)",
            question,
        )
        if timestamp:
            return day_constraint(*(int(value) for value in timestamp.groups()))
        compact_day = re.search(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)", question)
        if compact_day:
            return day_constraint(*(int(value) for value in compact_day.groups()))
        exact = re.search(r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日", question)
        if exact:
            return day_constraint(*(int(value) for value in exact.groups()))
        quarter = re.search(r"((?:19|20)\d{2})年第?([一二三四1-4])季度", question)
        if quarter:
            year, raw_quarter = quarter.groups()
            quarter_number = (
                {"一": 1, "二": 2, "三": 3, "四": 4}[raw_quarter]
                if raw_quarter in "一二三四"
                else int(raw_quarter)
            )
            start_month = (quarter_number - 1) * 3 + 1
            end_month = start_month + 2
            storage = f"{year}{start_month:02d}-{year}{end_month:02d}"
            return TemporalConstraint(temporal.column, "quarter", f"{year}-Q{quarter_number}", storage)
        month = re.search(r"((?:19|20)\d{2})年(\d{1,2})月", question)
        if month:
            year, month_value = month.groups()
            storage = f"{year}{int(month_value):02d}" if temporal.storage_format == "YYYYMMDD" else f"{year}-{int(month_value):02d}"
            return TemporalConstraint(temporal.column, "month", f"{year}-{int(month_value):02d}", storage)
        # A bare year is valid only when it is not the prefix of a longer
        # numeric date/code.  An explicit 年/年度 suffix is always accepted.
        year = re.search(r"(?<!\d)((?:19|20)\d{2})(?:年度?|(?=\D|$))", question)
        if year:
            if domain in {"fund_scale_reports", "fund_holder_structure"} and "年度" in question:
                return TemporalConstraint("定期报告所属年度", "year", year.group(1), year.group(1))
            return TemporalConstraint(temporal.column, "year", year.group(1), year.group(1))
        return None

    @staticmethod
    def _temporal_semantics(
        question: str,
        metric: str | None,
        temporal: TemporalConstraint | None,
        domain: str,
    ) -> tuple[str | None, tuple[str, ...], str | None]:
        if temporal is None:
            return None, (), None
        if temporal.precision == "day":
            return "as_of_date", (), None
        if temporal.precision == "month":
            return "within_period", (), None
        if temporal.precision == "quarter":
            return "within_period", (), None
        if any(signal in question for signal in ("年末", "年底", "年终", "截至")):
            return "period_end", (), None
        if any(signal in question for signal in ("全年最高", "年内最高", "年度峰值", "全年最大", "年内最大")):
            return "period_max", (), None
        if any(signal in question for signal in ("全年最低", "年内最低", "全年最小", "年内最小")):
            return "period_min", (), None
        if any(signal in question for signal in ("全年平均", "年内平均", "年度平均", "日均", "平均")):
            return "period_average", (), None
        snapshot_metrics = {
            "asset_net_value",
            "holding_quantity",
            "holding_market_value",
            "holding_nav_ratio",
            "bond_holding_quantity", "bond_holding_market_value", "bond_holding_nav_ratio",
            "convertible_holding_quantity", "convertible_holding_market_value", "convertible_holding_nav_ratio",
            "institutional_holder_shares", "institutional_holder_ratio",
            "individual_holder_shares", "individual_holder_ratio",
        }
        if metric in snapshot_metrics:
            if any(signal in question for signal in ("最高", "最大", "最多")):
                return "period_max", (), None
            if any(signal in question for signal in ("最低", "最小", "最少")):
                return "period_min", (), None
            ambiguity = ("period_end", "period_max", "period_average")
            subject = "基金规模" if metric == "asset_net_value" else "基金持仓快照"
            return (
                "ambiguous",
                ambiguity,
                f"“{temporal.value}年{subject}”需要明确时间口径：按年末快照、全年最高，还是全年平均排名？",
            )
        return "within_period", (), None

    @staticmethod
    def _result_shape(question: str, domain: str, top_k: int | None) -> tuple[str, str | None]:
        if (
            domain == "fund_daily_market"
            and all(marker in question for marker in ("交易日数量", "平均单位净值", "最高单位净值", "最低单位净值"))
        ):
            return "period_metric_summary", None
        if (
            domain == "fund_profile"
            and "不同基金类型数量" in question
            and any(marker in question for marker in ("每个管理人", "每位管理人", "各管理人"))
        ):
            return "manager_distinct_fund_type_count", None
        if domain == "fund_profile" and "管理费率" in question and any(marker in question for marker in ("为空", "空字符串", "缺失")):
            return "missing_management_fee_by_fund_type", None
        if domain == "fund_profile" and "按基金类型统计基金数量" in question:
            return "fund_type_count", None
        if domain == "fund_daily_market" and "不同基金数量" in question and "总记录数" in question:
            return "distinct_fund_and_record_count", None
        if domain == "fund_daily_market" and "每个月末" in question and "基金数量" in question:
            return "monthly_last_trading_day_fund_count", None
        if domain == "fund_holder_structure" and "不同基金数" in question:
            return "holder_distinct_fund_count", None
        if domain == "fund_daily_market" and "平均单位净值" in question and "至少" in question and "基金类型" in question:
            return "fund_type_daily_average", None
        if domain == "fund_daily_market" and "各基金类型" in question and "资产净值记录" in question and "基金数" in question:
            return "fund_type_daily_count", None
        if domain == "fund_holdings" and "每种基金类型" in question and "持仓记录数" in question:
            return "holding_count_by_fund_type", None
        if domain == "fund_scale_reports" and "规模变动记录数" in question and "不同基金数" in question:
            return "scale_count_by_fund_type", None
        if domain == "fund_scale_reports" and "份额净减少量" in question:
            return "scale_share_net_decrease", None
        if domain == "fund_scale_reports" and "期末总份额最高" in question:
            return "scale_top_period_end_shares", "global_top_k"
        if domain == "fund_daily_market" and "基金数量" in question and "资产净值合计" in question:
            return "daily_count_and_asset_sum", None
        if domain == "fund_holdings" and "每个管理人" in question and "不同基金数" in question:
            return "holding_distinct_fund_by_manager", None
        if domain == "fund_daily_market" and "各管理人" in question and "基金数" in question:
            return "daily_fund_count_by_manager", None
        # An existence check for one concrete calendar date must report the
        # database's coverage boundary; otherwise an out-of-range date simply
        # returns zero rows and looks like a missing record.
        if (
            domain == "fund_daily_market"
            and any(marker in question for marker in ("是否有", "有没有", "是否存在", "是否包含"))
            and re.search(r"(?:19|20)\d{2}年\d{1,2}月\d{1,2}日", question)
        ):
            return "date_coverage", None
        if (
            domain == "fund_daily_market"
            and "增长率" in question
            and any(signal in question for signal in ("第一个和最后一个", "首末交易日", "首个和最后一个"))
        ):
            return "period_endpoints_growth", None
        if (
            domain == "fund_daily_market"
            and "资产净值" in question
            and any(marker in question for marker in ("差额", "差值", "相差", "之差"))
            and len(re.findall(r"(?<!\d)\d{6}(?!\d)", question)) >= 2
        ):
            # Keep both values and their difference in one SQL result.  A
            # decomposed follow-up otherwise has to re-interpret two bare
            # database numbers and can fail before Decimal computation.
            return "fund_metric_difference", None
        if (
            domain == "fund_profile"
            and "基金总数" in question
            and "管理人去重数" in question
            and any(signal in question for signal in ("平均每位管理人", "每位管理人平均"))
        ):
            return "fund_profile_summary", None
        per_group = (
            top_k is not None
            and domain in {"fund_holdings", "fund_bond_holdings", "fund_convertible_bond_holdings"}
            and any(signal in question for signal in ("各持仓报告期", "每个持仓日期", "各报告期", "每期"))
        )
        if per_group:
            return "per_group_top_k", "per_group_top_k"
        if (
            top_k is not None
            and domain in {"fund_holdings", "fund_bond_holdings", "fund_convertible_bond_holdings"}
            and ("排名前" in question or re.search(r"前\d+大(?:债券|可转债)", question))
        ):
            return "rank_threshold", "per_group_top_k"
        if domain in {"fund_bond_holdings", "fund_convertible_bond_holdings"} and "记录数" in question and "市值合计" in question:
            return "holding_record_count_and_sum", None
        if top_k is not None:
            return "global_top_k", "global_top_k"
        if any(signal in question for signal in ("每个交易日", "每一交易日", "逐日", "每日", "时间序列")):
            return "time_series", None
        return "records", None

    @staticmethod
    def _select_columns(question: str, domain: str) -> tuple[str, ...]:
        columns: list[str] = []
        if domain == "fund_daily_market":
            if "日期" in question or re.search(r"哪(?:一)?天", question):
                columns.append("交易日期")
            for signal, column in (
                ("单位净值", "单位净值"),
                ("累计单位净值", "累计单位净值"),
                ("资产净值", "资产净值"),
            ):
                if signal in question and column not in columns:
                    columns.append(column)
        if domain == "fund_holdings":
            for signal, column in (
                ("股票代码", "股票代码"),
                ("股票名称", "股票名称"),
                ("名称", "股票名称"),
                ("基金代码", "基金代码"),
                ("基金简称", "基金简称"),
                ("市值", "市值"),
                ("净值占比", "市值占基金资产净值比"),
                ("排名", "第N大重仓股"),
            ):
                if signal in question and column not in columns:
                    columns.append(column)
        if domain == "fund_bond_holdings":
            for signal, column in (("债券类型", "债券类型"), ("债券名称", "债券名称"),
                                   ("持债数量", "持债数量"), ("持债市值", "持债市值"),
                                   ("持债占比", "持债市值占基金资产净值比"), ("排名", "第N大重仓股")):
                if signal in question and column not in columns:
                    columns.append(column)
        if domain == "fund_convertible_bond_holdings":
            for signal, column in (("对应股票代码", "对应股票代码"), ("债券名称", "债券名称"),
                                   ("数量", "数量"), ("市值", "市值"),
                                   ("净值占比", "市值占基金资产净值比"), ("排名", "第N大重仓股")):
                if signal in question and column not in columns:
                    columns.append(column)
        if domain == "fund_holder_structure":
            if "机构" in question and "份额" in question:
                columns.append("机构投资者持有的基金份额")
            if "机构" in question and any(signal in question for signal in ("比例", "占比")):
                columns.append("机构投资者持有的基金份额占总份额比例")
            if "个人" in question and "份额" in question:
                columns.append("个人投资者持有的基金份额")
            if "个人" in question and any(signal in question for signal in ("比例", "占比")):
                columns.append("个人投资者持有的基金份额占总份额比例")
        if domain in {"a_stock_daily_market", "hk_stock_daily_market"}:
            if "日期" in question or "交易日" in question:
                columns.append("交易日")
            for signal, column in (("昨收盘", "昨收盘(元)"), ("开盘", "今开盘(元)"),
                                   ("最高价", "最高价(元)"), ("最低价", "最低价(元)"),
                                   ("收盘价", "收盘价(元)"), ("成交量", "成交量(股)"),
                                   ("成交金额", "成交金额(元)")):
                if signal in question and column not in columns:
                    columns.append(column)
        if domain == "stock_industry":
            for signal, column in (("行业划分标准", "行业划分标准"), ("一级行业", "一级行业名称"),
                                   ("二级行业", "二级行业名称")):
                if signal in question and column not in columns:
                    columns.append(column)
        return tuple(columns)

    @staticmethod
    def _requests_monthly_snapshots(question: str) -> bool:
        return any(signal in question for signal in ("每月末", "每个月末", "逐月月末", "月末期"))

    @staticmethod
    def _aggregation(question: str, metric: str | None, temporal_semantics: str | None) -> str | None:
        if temporal_semantics == "period_max":
            return "max"
        if temporal_semantics == "period_min":
            return "min"
        if temporal_semantics == "period_average":
            return "avg"
        if metric == "holding_market_value" and any(term in question for term in ("市值合计", "市值总和")):
            return "sum"
        if "统计" in question and any(marker in question for marker in ("每", "各", "按")):
            return "count"
        return None

    @staticmethod
    def _group_by(question: str, target_entity: str, aggregation: str | None) -> tuple[str, ...]:
        if aggregation in {"max", "min", "avg"} and target_entity == "fund":
            return ("基金代码", "基金简称")
        if "每种基金类型" in question or "每类基金" in question or "各基金类型" in question or "按基金类型" in question:
            return ("基金类型",)
        if aggregation == "count" and any(marker in question for marker in ("每位管理人", "每个管理人", "各管理人")):
            return ("管理人",)
        return ()

    @staticmethod
    def _filters(
        question: str,
        grounding: FinancialEntityGrounding,
        temporal: TemporalConstraint | None,
        domain: str,
    ) -> tuple[SQLFilter, ...]:
        filters: list[SQLFilter] = []
        for entity in grounding.entities:
            field = (
                "对应股票代码"
                if entity.entity_type == "stock" and domain == "fund_convertible_bond_holdings"
                else "股票代码" if entity.entity_type == "stock" else "基金代码"
            )
            filters.append(SQLFilter(field, "=", entity.code))
        for fund_type in ("股票型", "混合型", "债券型", "货币型", "其他型"):
            if fund_type in question:
                filters.append(SQLFilter("基金类型", "=", fund_type))
        report_type = None
        for candidate in ("年报(含半年报)", "季报", "其他"):
            if candidate in question:
                report_type = candidate
                break
        if report_type is None and ("年报" in question or "半年报" in question):
            report_type = "年报(含半年报)"
        if report_type:
            filters.append(SQLFilter("报告类型", "=", report_type))
        if temporal:
            operator = (
                "=" if temporal.precision == "day"
                else "year_prefix" if temporal.precision == "year"
                else "quarter_range" if temporal.precision == "quarter"
                else "month_prefix"
            )
            filters.append(SQLFilter(temporal.field, operator, temporal.storage_value))
        at_least = re.search(r"至少(\d+)(?:只|个|类)?基金", question)
        if at_least:
            filters.append(SQLFilter("聚合基金数", ">=", at_least.group(1)))
        type_at_least = re.search(r"不少于(\d+)类", question)
        if type_at_least:
            filters.append(SQLFilter("聚合基金类型数", ">=", type_at_least.group(1)))
        return tuple(filters)

    @staticmethod
    def _ordering(
        question: str,
        metric: str | None,
        metric_field: SemanticField | None,
        aggregation: str | None,
        group_by: tuple[str, ...],
        result_shape: str,
    ) -> tuple[SQLOrdering, ...]:
        # A scalar comparison/difference is not a ranking request even when a
        # child phrase contains words such as "higher".  Injecting ORDER BY
        # changes its result shape and can make an otherwise valid aggregate
        # fail semantic validation.
        if any(term in question for term in ("差额", "差值", "相差", "之差", "比较")):
            return ()
        if "最高价" in question and "最低价" in question:
            return ()
        if aggregation == "sum" and not group_by:
            return ()
        if "份额净减少量" in question:
            return (SQLOrdering("份额净减少量", "desc"), SQLOrdering("基金代码", "asc"))
        if result_shape in {"period_metric_summary", "distinct_fund_and_record_count", "daily_count_and_asset_sum"}:
            return ()
        if result_shape == "missing_management_fee_by_fund_type":
            return (SQLOrdering("管理费率缺失数", "desc"), SQLOrdering("基金类型", "asc"))
        if result_shape == "manager_distinct_fund_type_count":
            return (SQLOrdering("基金类型数", "desc"), SQLOrdering("管理人", "asc"))
        if result_shape == "fund_type_count":
            return (SQLOrdering("基金数量", "desc"), SQLOrdering("基金类型", "asc"))
        if result_shape == "rank_threshold":
            return (SQLOrdering("第N大重仓股", "asc"),)
        if result_shape in {"holding_count_by_fund_type", "scale_count_by_fund_type"}:
            return (SQLOrdering("持仓记录数" if result_shape == "holding_count_by_fund_type" else "记录数", "desc"), SQLOrdering("基金类型", "asc"))
        if group_by == ("基金类型",) and any(term in question for term in ("基金数降序", "数量降序", "记录数降序", "基金数最多")):
            return (SQLOrdering("基金数", "desc"), SQLOrdering("基金类型", "asc"))
        if group_by == ("管理人",) and any(term in question for term in ("基金数降序", "数量降序", "最多", "前")):
            return (SQLOrdering("基金数", "desc"), SQLOrdering("管理人", "asc"))
        # Multi-aggregate summaries are scalar outputs; words such as 最高 and
        # 最低 describe selected aggregate columns, not row ordering.
        if aggregation == "count" and not group_by:
            return ()
        if "第" in question and "大重仓股" in question:
            return (SQLOrdering("第N大重仓股", "asc"), SQLOrdering("股票代码", "asc"))
        if metric and metric_field and any(term in question for term in ("最高", "最大", "最多", "前")):
            return (SQLOrdering(metric_field.column, "desc"),)
        if metric and metric_field and any(term in question for term in ("最低", "最小", "最少")):
            return (SQLOrdering(metric_field.column, "asc"),)
        if "升序" in question:
            if "交易日期" in question:
                return (SQLOrdering("交易日期", "asc"),)
            if "成立日期" in question:
                return (SQLOrdering("成立日期", "asc"),)
        if any(term in question for term in ("首个交易日", "第一个交易日", "最早交易日")):
            return (SQLOrdering("交易日期", "asc"),)
        if any(term in question for term in ("最后一个交易日", "末个交易日", "最晚交易日")):
            return (SQLOrdering("交易日期", "desc"),)
        return ()


def _top_k(question: str) -> int | None:
    match = re.search(r"(?:前|最高的?|最大的?|最多的?|top)\s*(\d{1,3})", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"前([一二三四五六七八九十])(?:个|只|名|条)?", question)
    if match:
        return _CHINESE_NUMBERS[match.group(1)]
    match = re.search(r"第([一二三四五六七八九十])大重仓股", question)
    if match:
        return _CHINESE_NUMBERS[match.group(1)]
    return None


def render_confirmed_sql_request(question: str, intent: SQLIntent) -> str:
    """Structured generator input; critical semantics are declarative, not inferred twice."""

    return intent.generator_payload(question)
