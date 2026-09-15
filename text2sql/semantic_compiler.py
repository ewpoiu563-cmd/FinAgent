"""Small deterministic SQL compiler for stable, high-frequency intent shapes."""

from __future__ import annotations

from .semantic_planner import SQLIntent


def compile_confirmed_intent(intent: SQLIntent) -> str | None:
    """Compile the common fund-holding ranking shape without reinterpreting intent.

    Unsupported shapes deliberately return ``None`` so the existing LLM generator
    remains available for complex Text-to-SQL requests.
    """

    if intent.result_shape == "period_endpoints_growth":
        return _compile_period_endpoints_growth(intent)

    if intent.result_shape == "fund_metric_difference":
        return _compile_fund_metric_difference(intent)

    if intent.result_shape == "fund_profile_summary":
        return _compile_fund_profile_summary(intent)

    if intent.result_shape == "date_coverage":
        return _compile_date_coverage(intent)

    deterministic_compilers = {
        "period_metric_summary": _compile_period_metric_summary,
        "manager_distinct_fund_type_count": _compile_manager_distinct_fund_type_count,
        "missing_management_fee_by_fund_type": _compile_missing_management_fee_by_fund_type,
        "fund_type_count": _compile_fund_type_count,
        "distinct_fund_and_record_count": _compile_distinct_fund_and_record_count,
        "monthly_last_trading_day_fund_count": _compile_monthly_last_trading_day_fund_count,
        "holder_distinct_fund_count": _compile_holder_distinct_fund_count,
        "fund_type_daily_average": _compile_fund_type_daily_average,
        "fund_type_daily_count": _compile_fund_type_daily_count,
        "holding_count_by_fund_type": _compile_holding_count_by_fund_type,
        "scale_count_by_fund_type": _compile_scale_count_by_fund_type,
        "scale_top_period_end_shares": _compile_scale_top_period_end_shares,
        "daily_count_and_asset_sum": _compile_daily_count_and_asset_sum,
        "holding_distinct_fund_by_manager": _compile_holding_distinct_fund_by_manager,
        "daily_fund_count_by_manager": _compile_daily_fund_count_by_manager,
    }
    deterministic = deterministic_compilers.get(intent.result_shape)
    if deterministic is not None:
        return deterministic(intent)

    if (
        intent.target_domain == "fund_daily_market"
        and intent.temporal_constraint is not None
        and intent.temporal_constraint.precision == "quarter"
        and intent.ordering
        and "交易日期" in intent.select_columns
    ):
        return _compile_quarter_extreme(intent)

    if (
        intent.target_domain == "fund_daily_market"
        and intent.temporal_constraint is not None
        and intent.temporal_constraint.precision == "year"
        and intent.ordering
        and intent.ordering[0].field == "交易日期"
        and "交易日期" in intent.select_columns
    ):
        return _compile_daily_period_boundary(intent)

    if intent.result_shape == "time_series":
        return _compile_fund_daily_time_series(intent)

    if intent.result_shape == "per_group_top_k":
        return _compile_holding_per_group_top_k(intent)

    if intent.result_shape == "rank_threshold":
        return _compile_rank_threshold(intent)

    if intent.result_shape == "holding_record_count_and_sum":
        return _compile_holding_record_count_and_sum(intent)

    if (
        intent.target_domain == "fund_holder_structure"
        and intent.metric_column == "机构投资者持有的基金份额占总份额比例"
        and intent.top_k is not None
    ):
        return _compile_holder_ratio_ranking(intent)

    if (
        intent.target_domain in {"fund_bond_holdings", "fund_convertible_bond_holdings"}
        and intent.metric_column in {"持债市值", "市值"}
        and intent.ordering
        and intent.ordering[0].direction == "desc"
    ):
        return _compile_highest_debt_holding(intent)

    if _is_fund_type_count_and_average(intent):
        return _compile_fund_type_count_and_average(intent)

    if _is_scale_share_net_decrease(intent):
        return _compile_scale_share_net_decrease(intent)

    if _is_holding_top_k_sum(intent):
        return _compile_holding_top_k_sum(intent)

    if (
        not intent.clarification_needed
        and intent.target_domain == "fund_profile"
        and intent.aggregation == "count"
        and intent.group_by == ("管理人",)
    ):
        top_k = intent.top_k if intent.top_k is not None else 100
        return (
            "SELECT [管理人], COUNT(*) AS [基金数量] FROM [基金基本信息] "
            "GROUP BY [管理人] ORDER BY [基金数量] DESC, [管理人] ASC "
            f"LIMIT {top_k}"
        )

    if (
        not intent.clarification_needed
        and intent.target_domain == "fund_daily_market"
        and intent.metric_table == "基金日行情表"
        and intent.metric_column == "资产净值"
        and intent.temporal_constraint is not None
        and intent.temporal_constraint.precision == "year"
        and intent.temporal_semantics in {"period_max", "period_min", "period_average"}
        and intent.aggregation in {"max", "min", "avg"}
    ):
        return _compile_fund_daily_ranking(intent)

    if (
        intent.clarification_needed
        or intent.target_domain != "fund_holdings"
        or intent.metric_table != "基金股票持仓明细"
        or not intent.metric_column
        or intent.temporal_constraint is None
        or intent.temporal_semantics not in {"as_of_date", "period_end", "period_max", "period_min", "period_average"}
        or intent.aggregation not in {None, "max", "min", "avg"}
    ):
        return None

    alias = "h"
    metric = _column(alias, intent.metric_column)
    aggregation = intent.aggregation.upper() if intent.aggregation else None
    metric_expression = f"{aggregation}({metric})" if aggregation else metric
    metric_alias = {
        "max": f"最大{intent.metric_column}",
        "min": f"最小{intent.metric_column}",
        "avg": f"平均{intent.metric_column}",
    }.get(intent.aggregation, intent.metric_column)

    select_fields = [f"{alias}.[基金代码]", f"{alias}.[基金简称]"]
    for column in intent.select_columns:
        field = f"{alias}.[{column}]"
        if field not in select_fields:
            select_fields.append(field)
    if f"{alias}.[{metric_alias}]" not in select_fields:
        select_fields.append(f"{metric_expression} AS [{metric_alias}]")

    predicates: list[str] = []
    for item in intent.filters:
        if item.field not in {"股票代码", "股票名称", "基金代码", "基金简称", "报告类型", "持仓日期"}:
            return None
        if item.operator == "=":
            predicates.append(f"{_column(alias, item.field)} = {_literal(item.value)}")
        elif item.operator in {"year_prefix", "month_prefix"}:
            predicates.append(f"{_column(alias, item.field)} LIKE {_literal(item.value + '%')}")
        else:
            return None

    clauses = [
        "SELECT " + ", ".join(select_fields),
        "FROM [基金股票持仓明细] h",
    ]
    if predicates:
        clauses.append("WHERE " + " AND ".join(predicates))
    if intent.group_by:
        clauses.append("GROUP BY " + ", ".join(_column(alias, field) for field in intent.group_by))

    if intent.ordering:
        order_parts: list[str] = []
        for item in intent.ordering:
            expression = metric_expression if item.field == intent.metric_column else _column(alias, item.field)
            order_parts.append(f"{expression} {item.direction.upper()}")
        order_parts.append(f"{alias}.[基金代码] ASC")
        clauses.append("ORDER BY " + ", ".join(order_parts))
    clauses.append(f"LIMIT {intent.top_k if intent.top_k is not None else 100}")
    return " ".join(clauses)


def _compile_period_endpoints_growth(intent: SQLIntent) -> str | None:
    if intent.temporal_constraint is None or intent.temporal_constraint.precision != "year":
        return None
    fund = next((item.value for item in intent.filters if item.field == "基金代码"), None)
    if fund is None:
        return None
    year = intent.temporal_constraint.value
    fund_literal = _literal(fund)
    period_literal = _literal(year + "%")
    return " ".join((
        "SELECT f.[交易日期] AS first_date, f.[单位净值] AS first_nav,",
        "l.[交易日期] AS last_date, l.[单位净值] AS last_nav,",
        "ROUND((l.[单位净值] - f.[单位净值]) / f.[单位净值] * 100.0, 2) AS growth_rate_pct",
        f"FROM [基金日行情表] f JOIN [基金日行情表] l ON l.[基金代码] = {fund_literal}",
        "AND l.[交易日期] = (SELECT MAX([交易日期]) FROM [基金日行情表]",
        f"WHERE [基金代码] = {fund_literal} AND [交易日期] LIKE {period_literal})",
        f"WHERE f.[基金代码] = {fund_literal}",
        "AND f.[交易日期] = (SELECT MIN([交易日期]) FROM [基金日行情表]",
        f"WHERE [基金代码] = {fund_literal} AND [交易日期] LIKE {period_literal})",
    ))


def _compile_fund_metric_difference(intent: SQLIntent) -> str | None:
    """Compile two same-date fund values and their signed difference.

    The order follows the grounded entities in the request, so the difference
    is deterministic and the output remains a single typed SQL observation.
    """

    if (
        intent.target_domain != "fund_daily_market"
        or intent.metric_column != "资产净值"
        or intent.temporal_constraint is None
        or intent.temporal_constraint.precision != "day"
        or len(intent.grounded_entities) != 2
    ):
        return None
    first, second = intent.grounded_entities
    date_value = _literal(intent.temporal_constraint.storage_value)
    first_code, second_code = _literal(first.code), _literal(second.code)
    first_value = f"MAX(CASE WHEN [基金代码] = {first_code} THEN [资产净值] END)"
    second_value = f"MAX(CASE WHEN [基金代码] = {second_code} THEN [资产净值] END)"
    return " ".join((
        "SELECT",
        f"{first_value} AS [{first.code}资产净值],",
        f"{second_value} AS [{second.code}资产净值],",
        f"({first_value} - {second_value}) AS [资产净值差额]",
        "FROM [基金日行情表]",
        f"WHERE [基金代码] IN ({first_code}, {second_code})",
        f"AND [交易日期] = {date_value}",
    ))


def _compile_fund_profile_summary(intent: SQLIntent) -> str | None:
    if intent.target_domain != "fund_profile":
        return None
    return (
        "SELECT COUNT(*) AS [基金总数], COUNT(DISTINCT [管理人]) AS [管理人去重数], "
        "ROUND(CAST(COUNT(*) AS REAL) / NULLIF(COUNT(DISTINCT [管理人]), 0), 2) "
        "AS [平均每位管理人基金数] FROM [基金基本信息]"
    )


def _compile_date_coverage(intent: SQLIntent) -> str | None:
    fund = next((item.value for item in intent.filters if item.field == "基金代码"), None)
    temporal = intent.temporal_constraint
    if fund is None or temporal is None or temporal.precision != "day":
        return None
    fund_literal = _literal(fund)
    target = _literal(temporal.storage_value)
    return (
        "SELECT MIN([交易日期]) AS [数据最早日期], MAX([交易日期]) AS [数据最晚日期], "
        f"SUM(CASE WHEN [交易日期] = {target} THEN 1 ELSE 0 END) AS [目标日期行情记录数] "
        f"FROM [基金日行情表] WHERE [基金代码] = {fund_literal}"
    )


def _filter_value(intent: SQLIntent, field: str, operator: str | None = None) -> str | None:
    return next(
        (item.value for item in intent.filters if item.field == field and (operator is None or item.operator == operator)),
        None,
    )


def _compile_period_metric_summary(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    fund = _filter_value(intent, "基金代码", "=")
    if temporal is None or temporal.precision != "month" or fund is None:
        return None
    return (
        "SELECT COUNT(*) AS [交易日数], ROUND(AVG([单位净值]), 6) AS [平均单位净值], "
        "MAX([单位净值]) AS [最高单位净值], MIN([单位净值]) AS [最低单位净值] "
        "FROM [基金日行情表] "
        f"WHERE [基金代码] = {_literal(fund)} AND [交易日期] LIKE {_literal(temporal.storage_value + '%')} "
        "AND [单位净值] IS NOT NULL"
    )


def _compile_manager_distinct_fund_type_count(intent: SQLIntent) -> str:
    minimum = int(_filter_value(intent, "聚合基金类型数", ">=") or "1")
    limit = intent.top_k if intent.top_k is not None else 100
    return (
        "SELECT [管理人], COUNT(DISTINCT [基金类型]) AS [基金类型数] FROM [基金基本信息] "
        "GROUP BY [管理人] "
        f"HAVING COUNT(DISTINCT [基金类型]) >= {minimum} "
        "ORDER BY [基金类型数] DESC, [管理人] ASC "
        f"LIMIT {limit}"
    )


def _compile_missing_management_fee_by_fund_type(intent: SQLIntent) -> str:
    return (
        "SELECT [基金类型], COUNT(*) AS [管理费率缺失数] FROM [基金基本信息] "
        "WHERE [管理费率] IS NULL OR TRIM([管理费率]) = '' "
        "GROUP BY [基金类型] ORDER BY [管理费率缺失数] DESC, [基金类型] ASC"
    )


def _compile_fund_type_count(intent: SQLIntent) -> str:
    limit = intent.top_k if intent.top_k is not None else 100
    return (
        "SELECT [基金类型], COUNT(*) AS [基金数量] FROM [基金基本信息] "
        "GROUP BY [基金类型] ORDER BY [基金数量] DESC, [基金类型] ASC "
        f"LIMIT {limit}"
    )


def _compile_distinct_fund_and_record_count(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "year":
        return None
    return (
        "SELECT COUNT(DISTINCT [基金代码]) AS [基金数], COUNT(*) AS [记录数] "
        "FROM [基金日行情表] "
        f"WHERE [交易日期] LIKE {_literal(temporal.value + '%')} AND [单位净值] IS NOT NULL"
    )


def _compile_monthly_last_trading_day_fund_count(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "year":
        return None
    year = _literal(temporal.value + "%")
    # Some source rows carry weekend month-end snapshots.  Exclude Saturdays
    # and Sundays, then use the final observed business day of each month.
    return (
        "SELECT [月份], [基金数] FROM (SELECT daily.[交易日期], daily.[月份], daily.[基金数], "
        "ROW_NUMBER() OVER (PARTITION BY daily.[月份] ORDER BY daily.[交易日期] DESC) AS rn "
        "FROM (SELECT [交易日期], SUBSTR([交易日期], 1, 6) AS [月份], "
        "COUNT(DISTINCT [基金代码]) AS [基金数] FROM [基金日行情表] "
        f"WHERE [交易日期] LIKE {year} AND STRFTIME('%w', SUBSTR([交易日期],1,4)||'-'||"
        "SUBSTR([交易日期],5,2)||'-'||SUBSTR([交易日期],7,2)) NOT IN ('0','6') GROUP BY [交易日期]) daily) ranked "
        "WHERE rn = 1 ORDER BY [月份] ASC"
    )


def _compile_holder_distinct_fund_count(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "year":
        return None
    return (
        "SELECT COUNT(DISTINCT [基金代码]) AS [基金数] FROM [基金份额持有人结构] "
        f"WHERE [定期报告所属年度] = {int(temporal.value)}"
    )


def _compile_fund_type_daily_average(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "day":
        return None
    minimum = int(_filter_value(intent, "聚合基金数", ">=") or "1")
    return " ".join((
        "SELECT b.[基金类型], COUNT(*) AS [基金数], ROUND(AVG(d.[单位净值]), 6) AS [平均单位净值]",
        "FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码]",
        f"WHERE d.[交易日期] = {_literal(temporal.storage_value)} AND d.[单位净值] IS NOT NULL",
        "GROUP BY b.[基金类型]",
        f"HAVING COUNT(*) >= {minimum}",
        "ORDER BY [平均单位净值] DESC, b.[基金类型] ASC",
    ))


def _compile_fund_type_daily_count(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "day":
        return None
    limit = intent.top_k if intent.top_k is not None else 100
    return " ".join((
        "SELECT b.[基金类型], COUNT(*) AS [基金数]",
        "FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码]",
        f"WHERE d.[交易日期] = {_literal(temporal.storage_value)} AND d.[资产净值] IS NOT NULL",
        "GROUP BY b.[基金类型] ORDER BY [基金数] DESC, b.[基金类型] ASC",
        f"LIMIT {limit}",
    ))


def _compile_holding_count_by_fund_type(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    report_type = _filter_value(intent, "报告类型", "=")
    if temporal is None or temporal.precision != "day" or report_type is None:
        return None
    limit = intent.top_k if intent.top_k is not None else 100
    return " ".join((
        "SELECT b.[基金类型], COUNT(*) AS [持仓记录数]",
        "FROM [基金股票持仓明细] h JOIN [基金基本信息] b ON b.[基金代码] = h.[基金代码]",
        f"WHERE h.[持仓日期] = {_literal(temporal.storage_value)} AND h.[报告类型] = {_literal(report_type)}",
        "GROUP BY b.[基金类型] ORDER BY [持仓记录数] DESC, b.[基金类型] ASC",
        f"LIMIT {limit}",
    ))


def _compile_scale_count_by_fund_type(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "year":
        return None
    limit = intent.top_k if intent.top_k is not None else 100
    return " ".join((
        "SELECT b.[基金类型], COUNT(*) AS [记录数], COUNT(DISTINCT s.[基金代码]) AS [基金数]",
        "FROM [基金规模变动表] s JOIN [基金基本信息] b ON b.[基金代码] = s.[基金代码]",
        f"WHERE s.[定期报告所属年度] = {int(temporal.value)} GROUP BY b.[基金类型]",
        "ORDER BY [记录数] DESC, b.[基金类型] ASC",
        f"LIMIT {limit}",
    ))


def _compile_scale_top_period_end_shares(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "year" or intent.top_k is None:
        return None
    return " ".join((
        "SELECT s.[基金代码], s.[基金简称], b.[基金类型], s.[截止日期], s.[报告期期末基金总份额]",
        "FROM [基金规模变动表] s LEFT JOIN [基金基本信息] b ON s.[基金代码] = b.[基金代码]",
        f"WHERE s.[定期报告所属年度] = {int(temporal.value)}",
        "ORDER BY s.[报告期期末基金总份额] DESC, s.[基金代码] ASC",
        f"LIMIT {intent.top_k}",
    ))


def _compile_daily_count_and_asset_sum(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    fund_type = _filter_value(intent, "基金类型", "=")
    if temporal is None or temporal.precision != "day" or fund_type is None:
        return None
    return " ".join((
        "SELECT COUNT(*) AS [基金数], ROUND(SUM(d.[资产净值]), 2) AS [资产净值合计]",
        "FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码]",
        f"WHERE d.[交易日期] = {_literal(temporal.storage_value)} AND b.[基金类型] = {_literal(fund_type)}",
        "AND d.[资产净值] IS NOT NULL",
    ))


def _compile_holding_distinct_fund_by_manager(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    report_type = _filter_value(intent, "报告类型", "=")
    if temporal is None or temporal.precision != "day" or report_type is None:
        return None
    limit = intent.top_k if intent.top_k is not None else 100
    return " ".join((
        "SELECT b.[管理人], COUNT(DISTINCT h.[基金代码]) AS [基金数]",
        "FROM [基金股票持仓明细] h JOIN [基金基本信息] b ON b.[基金代码] = h.[基金代码]",
        f"WHERE h.[持仓日期] = {_literal(temporal.storage_value)} AND h.[报告类型] = {_literal(report_type)}",
        "GROUP BY b.[管理人] ORDER BY [基金数] DESC, b.[管理人] ASC",
        f"LIMIT {limit}",
    ))


def _compile_daily_fund_count_by_manager(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "day":
        return None
    limit = intent.top_k if intent.top_k is not None else 100
    return " ".join((
        "SELECT b.[管理人], COUNT(*) AS [基金数]",
        "FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码]",
        f"WHERE d.[交易日期] = {_literal(temporal.storage_value)} AND d.[资产净值] IS NOT NULL",
        "GROUP BY b.[管理人] ORDER BY [基金数] DESC, b.[管理人] ASC",
        f"LIMIT {limit}",
    ))


def _compile_quarter_extreme(intent: SQLIntent) -> str | None:
    fund = next((item.value for item in intent.filters if item.field == "基金代码"), None)
    temporal = intent.temporal_constraint
    if fund is None or temporal is None or "-" not in temporal.storage_value or not intent.metric_column:
        return None
    start, end = temporal.storage_value.split("-", 1)
    direction = intent.ordering[0].direction.upper()
    return (
        f"SELECT [交易日期], [{intent.metric_column}] FROM [基金日行情表] "
        f"WHERE [基金代码] = {_literal(fund)} AND [交易日期] >= {_literal(start + '01')} "
        f"AND [交易日期] <= {_literal(end + '31')} "
        f"ORDER BY [{intent.metric_column}] {direction}, [交易日期] ASC LIMIT 1"
    )


def _compile_daily_period_boundary(intent: SQLIntent) -> str | None:
    fund = next((item.value for item in intent.filters if item.field == "基金代码"), None)
    temporal = intent.temporal_constraint
    if fund is None or temporal is None:
        return None
    columns = list(dict.fromkeys(("交易日期", *intent.select_columns)))
    direction = intent.ordering[0].direction.upper()
    return (
        "SELECT " + ", ".join(f"[{column}]" for column in columns)
        + " FROM [基金日行情表] "
        + f"WHERE [基金代码] = {_literal(fund)} AND [交易日期] LIKE {_literal(temporal.value + '%')} "
        + f"ORDER BY [交易日期] {direction} LIMIT 1"
    )


def _compile_fund_daily_time_series(intent: SQLIntent) -> str | None:
    if (
        intent.clarification_needed
        or intent.target_domain != "fund_daily_market"
        or intent.metric_table != "基金日行情表"
        or intent.temporal_constraint is None
        or intent.temporal_semantics != "within_period"
        or not intent.select_columns
    ):
        return None
    predicates: list[str] = []
    for item in intent.filters:
        if item.field == "基金代码" and item.operator == "=":
            predicates.append(f"d.[基金代码] = {_literal(item.value)}")
        elif item.field == "交易日期" and item.operator in {"year_prefix", "month_prefix"}:
            predicates.append(f"d.[交易日期] LIKE {_literal(item.value + '%')}")
        else:
            return None
    selected = ["d.[交易日期]"] + [f"d.[{column}]" for column in intent.select_columns]
    return " ".join((
        "SELECT " + ", ".join(selected),
        "FROM [基金日行情表] d",
        "WHERE " + " AND ".join(predicates),
        "ORDER BY d.[交易日期] ASC",
    ))


def _compile_holding_per_group_top_k(intent: SQLIntent) -> str | None:
    if (
        intent.clarification_needed
        or intent.target_domain != "fund_holdings"
        or intent.temporal_constraint is None
        or intent.top_k is None
    ):
        return None
    predicates: list[str] = []
    for item in intent.filters:
        if item.field in {"基金代码", "股票代码", "报告类型"} and item.operator == "=":
            predicates.append(f"h.[{item.field}] = {_literal(item.value)}")
        elif item.field == "持仓日期" and item.operator in {"year_prefix", "month_prefix"}:
            predicates.append(f"h.[持仓日期] LIKE {_literal(item.value + '%')}")
        else:
            return None
    predicates.append(f"h.[第N大重仓股] <= {intent.top_k}")
    return " ".join((
        "SELECT h.[持仓日期], h.[第N大重仓股], h.[股票代码], h.[股票名称], "
        "h.[数量], h.[市值], h.[市值占基金资产净值比]",
        "FROM [基金股票持仓明细] h",
        "WHERE " + " AND ".join(predicates),
        "ORDER BY h.[持仓日期] ASC, h.[第N大重仓股] ASC, h.[股票代码] ASC",
    ))


def _compile_rank_threshold(intent: SQLIntent) -> str | None:
    configuration = {
        "fund_holdings": (
            "基金股票持仓明细",
            ("第N大重仓股", "股票代码", "股票名称", "市值", "市值占基金资产净值比"),
            "股票代码",
        ),
        "fund_bond_holdings": (
            "基金债券持仓明细",
            ("第N大重仓股", "债券类型", "债券名称", "持债数量", "持债市值"),
            "债券名称",
        ),
        "fund_convertible_bond_holdings": (
            "基金可转债持仓明细",
            ("第N大重仓股", "对应股票代码", "债券名称", "数量", "市值"),
            "债券名称",
        ),
    }.get(intent.target_domain)
    temporal = intent.temporal_constraint
    if configuration is None or temporal is None or temporal.precision != "day" or intent.top_k is None:
        return None
    table, columns, tie_breaker = configuration
    predicates = []
    for item in intent.filters:
        if item.field in {"基金代码", "报告类型", temporal.field} and item.operator == "=":
            predicates.append(f"[{item.field}] = {_literal(item.value)}")
    predicates.append(f"[第N大重仓股] <= {intent.top_k}")
    return " ".join((
        "SELECT " + ", ".join(f"[{column}]" for column in columns),
        f"FROM [{table}] WHERE " + " AND ".join(predicates),
        f"ORDER BY [第N大重仓股] ASC, [{tie_breaker}] ASC",
    ))


def _compile_holding_record_count_and_sum(intent: SQLIntent) -> str | None:
    table_and_metric = {
        "fund_bond_holdings": ("基金债券持仓明细", "持债市值", "持债市值合计"),
        "fund_convertible_bond_holdings": ("基金可转债持仓明细", "市值", "可转债市值合计"),
    }.get(intent.target_domain)
    temporal = intent.temporal_constraint
    if table_and_metric is None or temporal is None or temporal.precision != "day":
        return None
    table, metric, alias = table_and_metric
    predicates = [
        f"[{item.field}] = {_literal(item.value)}"
        for item in intent.filters
        if item.field in {"基金代码", "报告类型", temporal.field} and item.operator == "="
    ]
    return (
        f"SELECT COUNT(*) AS [记录数], ROUND(SUM([{metric}]), 2) AS [{alias}] "
        f"FROM [{table}] WHERE " + " AND ".join(predicates)
    )


def _compile_holder_ratio_ranking(intent: SQLIntent) -> str | None:
    temporal = intent.temporal_constraint
    if temporal is None or temporal.precision != "day" or intent.top_k is None:
        return None
    metric = "机构投资者持有的基金份额占总份额比例"
    return (
        f"SELECT [基金代码], [基金简称], [{metric}] FROM [基金份额持有人结构] "
        f"WHERE [{temporal.field}] = {_literal(temporal.storage_value)} "
        f"ORDER BY [{metric}] DESC, [基金代码] ASC LIMIT {intent.top_k}"
    )


def _compile_highest_debt_holding(intent: SQLIntent) -> str | None:
    configuration = {
        "fund_bond_holdings": (
            "基金债券持仓明细",
            ("债券类型", "债券名称", "持债市值", "持债市值占基金资产净值比"),
            "持债市值",
        ),
        "fund_convertible_bond_holdings": (
            "基金可转债持仓明细",
            ("对应股票代码", "债券名称", "市值", "市值占基金资产净值比"),
            "市值",
        ),
    }.get(intent.target_domain)
    temporal = intent.temporal_constraint
    if configuration is None or temporal is None or temporal.precision != "day":
        return None
    table, columns, metric = configuration
    predicates = [
        f"[{item.field}] = {_literal(item.value)}"
        for item in intent.filters
        if item.field in {"基金代码", "报告类型", temporal.field} and item.operator == "="
    ]
    return " ".join((
        "SELECT " + ", ".join(f"[{column}]" for column in columns),
        f"FROM [{table}] WHERE " + " AND ".join(predicates),
        f"ORDER BY [{metric}] DESC, [债券名称] ASC LIMIT 1",
    ))


def _compile_fund_daily_ranking(intent: SQLIntent) -> str | None:
    metric = f"d.[{intent.metric_column}]"
    function = intent.aggregation.upper()
    metric_expression = f"{function}({metric})"
    metric_alias = {
        "max": "最高资产净值",
        "min": "最低资产净值",
        "avg": "平均资产净值",
    }[intent.aggregation]
    predicates: list[str] = []
    for item in intent.filters:
        if item.field == "交易日期" and item.operator == "year_prefix":
            predicates.append(f"d.[交易日期] LIKE {_literal(item.value + '%')}")
        elif item.field == "基金类型" and item.operator == "=":
            predicates.append(f"b.[基金类型] = {_literal(item.value)}")
        elif item.field == "基金代码" and item.operator == "=":
            predicates.append(f"d.[基金代码] = {_literal(item.value)}")
        else:
            return None
    direction = intent.ordering[0].direction.upper() if intent.ordering else (
        "ASC" if intent.aggregation == "min" else "DESC"
    )
    top_k = intent.top_k if intent.top_k is not None else 100
    return " ".join(
        (
            f"SELECT b.[基金代码], b.[基金简称], {metric_expression} AS [{metric_alias}]",
            "FROM [基金日行情表] d JOIN [基金基本信息] b ON d.[基金代码] = b.[基金代码]",
            "WHERE " + " AND ".join(predicates),
            "GROUP BY b.[基金代码], b.[基金简称]",
            f"ORDER BY {metric_expression} {direction}, b.[基金代码] ASC",
            f"LIMIT {top_k}",
        )
    )


def _is_fund_type_count_and_average(intent: SQLIntent) -> bool:
    return (
        not intent.clarification_needed
        and intent.target_domain == "fund_daily_market"
        and intent.metric_column == "资产净值"
        and intent.aggregation == "count"
        and intent.group_by == ("基金类型",)
        and intent.temporal_constraint is not None
        and intent.temporal_constraint.precision == "day"
    )


def _compile_fund_type_count_and_average(intent: SQLIntent) -> str:
    date = _literal(intent.temporal_constraint.storage_value)
    limit = intent.top_k if intent.top_k is not None else 100
    return " ".join((
        "SELECT b.[基金类型], COUNT(*) AS [基金数], ROUND(AVG(d.[资产净值]), 2) AS [平均资产净值]",
        "FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码]",
        f"WHERE d.[交易日期] = {date} AND d.[资产净值] IS NOT NULL",
        "GROUP BY b.[基金类型]",
        "ORDER BY [基金数] DESC, b.[基金类型] ASC",
        f"LIMIT {limit}",
    ))


def _is_scale_share_net_decrease(intent: SQLIntent) -> bool:
    return (
        not intent.clarification_needed
        and intent.result_shape == "scale_share_net_decrease"
        and intent.target_domain == "fund_scale_reports"
        and intent.metric_column == "报告期期末基金总份额"
        and intent.top_k is not None
        and intent.temporal_constraint is not None
        and intent.temporal_constraint.precision == "year"
    )


def _compile_scale_share_net_decrease(intent: SQLIntent) -> str:
    year = _literal(intent.temporal_constraint.value)
    limit = intent.top_k if intent.top_k is not None else 100
    return " ".join((
        "SELECT [基金代码], [基金简称], [截止日期], [报告期期初基金总份额], [报告期基金总申购份额],",
        "[报告期基金总赎回份额], [报告期期末基金总份额],",
        "ROUND([报告期期初基金总份额] - [报告期期末基金总份额], 2) AS [份额净减少量]",
        "FROM [基金规模变动表]",
        f"WHERE [定期报告所属年度] = {year}",
        "AND [报告期期末基金总份额] < [报告期期初基金总份额]",
        "AND [报告期基金总赎回份额] > [报告期基金总申购份额]",
        "ORDER BY [份额净减少量] DESC, [基金代码] ASC",
        f"LIMIT {limit}",
    ))


def _is_holding_top_k_sum(intent: SQLIntent) -> bool:
    return (
        not intent.clarification_needed
        and intent.target_domain == "fund_holdings"
        and intent.metric_column == "市值"
        and intent.aggregation == "sum"
        and intent.top_k is not None
        and intent.temporal_constraint is not None
    )


def _compile_holding_top_k_sum(intent: SQLIntent) -> str:
    predicates: list[str] = []
    for item in intent.filters:
        if item.field in {"基金代码", "股票代码", "报告类型", "持仓日期"} and item.operator == "=":
            predicates.append(f"h.[{item.field}] = {_literal(item.value)}")
        else:
            return None
    predicates.append(f"h.[第N大重仓股] BETWEEN 1 AND {intent.top_k}")
    return " ".join((
        "SELECT COUNT(*) AS [记录数], ROUND(SUM(h.[市值]), 2) AS [前十大市值合计]",
        "FROM [基金股票持仓明细] h",
        "WHERE " + " AND ".join(predicates),
    ))


def _column(alias: str, field: str) -> str:
    if "]" in field:
        raise ValueError("invalid SQL identifier in confirmed intent")
    return f"{alias}.[{field}]"


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
