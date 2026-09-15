"""Build the frozen 100-case Text-to-SQL Gold set for testrun003.

The cases are authored as natural-language questions plus read-only reference
SQL.  Their answers are materialized directly from the pinned SQLite database;
no agent answer is used to author Gold.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from materialize_sql_gold import materialize


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "finance.db"
OUT = ROOT / "eval_v2" / "testruns" / "testrun003" / "inputs" / "gold"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case(case_id: int, difficulty: str, question: str, sql: str, *, ordered: bool = True,
          keys: list[str] | None = None, tolerance: float = 0.0) -> dict[str, Any]:
    return {
        "case_id": f"v3-t2sql-{case_id:03d}",
        "difficulty": difficulty,
        "question": "只使用本地数据库，" + question,
        "gold_sql": sql,
        "answer_semantics": {
            "kind": "rows", "ordered": ordered, "key_columns": keys or [],
            "numeric_tolerance": tolerance,
        },
        "derived_results": [],
    }


def _empty(case_id: int, question: str, sql: str) -> dict[str, Any]:
    row = _case(case_id, "boundary", question, sql)
    row["answer_semantics"] = {"kind": "empty_rows", "ordered": True, "key_columns": []}
    return row


def build_cases(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Create 100 distinct, executable cases from stable database anchors."""

    basic_codes = [r[0] for r in connection.execute(
        "SELECT [基金代码] FROM [基金基本信息] ORDER BY [基金代码] LIMIT 30"
    )]
    daily_codes = [r[0] for r in connection.execute(
        "SELECT [基金代码] FROM [基金日行情表] WHERE [交易日期] LIKE '2021%' "
        "GROUP BY [基金代码] HAVING COUNT(*) >= 100 ORDER BY [基金代码] LIMIT 25"
    )]
    holding_anchors = list(connection.execute(
        "SELECT [基金代码], [持仓日期], [报告类型] FROM [基金股票持仓明细] "
        "GROUP BY [基金代码], [持仓日期], [报告类型] HAVING COUNT(*) >= 10 "
        "ORDER BY [持仓日期], [基金代码] LIMIT 30"
    ))
    scale_anchors = list(connection.execute(
        "SELECT [基金代码], [定期报告所属年度], [截止日期] FROM [基金规模变动表] "
        "WHERE [报告期期末基金总份额] IS NOT NULL GROUP BY [基金代码], [定期报告所属年度], [截止日期] "
        "ORDER BY [定期报告所属年度], [截止日期], [基金代码] LIMIT 20"
    ))
    if len(basic_codes) < 30 or len(daily_codes) < 25 or len(holding_anchors) < 30 or len(scale_anchors) < 20:
        raise RuntimeError("database lacks anchors required for the 100-case benchmark")

    cases: list[dict[str, Any]] = []
    n = 1
    # 1--25: fund profile (projection, filters, aggregation and ordering).
    field_sets = [
        ("基金简称和基金类型", "[基金简称], [基金类型]"),
        ("基金全称、管理人和托管人", "[基金全称], [管理人], [托管人]"),
        ("成立日期、到期日期和管理费率", "[成立日期], [到期日期], [管理费率]"),
        ("基金简称、管理费率和托管费率", "[基金简称], [管理费率], [托管费率]"),
        ("基金类型、管理人和成立日期", "[基金类型], [管理人], [成立日期]"),
    ]
    for index in range(20):
        code = basic_codes[index]
        label, columns = field_sets[index % len(field_sets)]
        cases.append(_case(n, "daily" if index < 10 else "hard",
            f"查询基金代码{code}的{label}。",
            f"SELECT {columns} FROM [基金基本信息] WHERE [基金代码] = '{code}';"))
        n += 1
    cases.extend([
        _case(n, "daily", "按基金类型统计基金数量，按数量降序、基金类型升序返回前5类。",
              "SELECT [基金类型], COUNT(*) AS [基金数量] FROM [基金基本信息] GROUP BY [基金类型] ORDER BY [基金数量] DESC, [基金类型] ASC LIMIT 5;"),
        _case(n + 1, "hard", "统计每个管理人管理的不同基金类型数量，仅返回不少于3类的管理人，按类型数降序、管理人升序返回前10位。",
              "SELECT [管理人], COUNT(DISTINCT [基金类型]) AS [基金类型数] FROM [基金基本信息] GROUP BY [管理人] HAVING COUNT(DISTINCT [基金类型]) >= 3 ORDER BY [基金类型数] DESC, [管理人] ASC LIMIT 10;"),
        _case(n + 2, "hard", "按基金类型统计管理费率为空或空字符串的基金数，按数量降序、类型升序返回全部结果。",
              "SELECT [基金类型], COUNT(*) AS [管理费率缺失数] FROM [基金基本信息] WHERE [管理费率] IS NULL OR TRIM([管理费率]) = '' GROUP BY [基金类型] ORDER BY [管理费率缺失数] DESC, [基金类型] ASC;"),
        _case(n + 3, "hard", "找出成立日期最早的10只基金，返回基金代码、简称、成立日期，按成立日期升序、基金代码升序。",
              "SELECT [基金代码], [基金简称], [成立日期] FROM [基金基本信息] WHERE [成立日期] IS NOT NULL AND TRIM([成立日期]) <> '' ORDER BY [成立日期] ASC, [基金代码] ASC LIMIT 10;"),
        _empty(n + 4, "查询基金代码999998的基金简称和管理人；不存在时明确返回无记录。",
               "SELECT [基金简称], [管理人] FROM [基金基本信息] WHERE [基金代码] = '999998';"),
    ])
    n += 5

    # 26--50: daily market facts, extrema, ranges, and grouped aggregates.
    for index in range(10):
        code = daily_codes[index]
        cases.append(_case(n, "daily", f"查询基金{code}在2021年首个有单位净值记录的交易日期、单位净值和累计单位净值。",
            f"SELECT [交易日期], [单位净值], [累计单位净值] FROM [基金日行情表] WHERE [基金代码] = '{code}' AND [交易日期] LIKE '2021%' AND [单位净值] IS NOT NULL ORDER BY [交易日期] ASC LIMIT 1;"))
        n += 1
    for index in range(5):
        code = daily_codes[10 + index]
        cases.append(_case(n, "hard", f"找出基金{code}在2021年第二季度单位净值最低的交易日及单位净值；并列时取日期最早。",
            f"SELECT [交易日期], [单位净值] FROM [基金日行情表] WHERE [基金代码] = '{code}' AND [交易日期] BETWEEN '20210401' AND '20210630' AND [单位净值] IS NOT NULL ORDER BY [单位净值] ASC, [交易日期] ASC LIMIT 1;"))
        n += 1
    for index in range(5):
        code = daily_codes[15 + index]
        cases.append(_case(n, "hard", f"统计基金{code}在2021年7月有单位净值的交易日数量、平均单位净值、最高单位净值和最低单位净值。",
            f"SELECT COUNT(*) AS [交易日数], ROUND(AVG([单位净值]), 6) AS [平均单位净值], MAX([单位净值]) AS [最高单位净值], MIN([单位净值]) AS [最低单位净值] FROM [基金日行情表] WHERE [基金代码] = '{code}' AND [交易日期] LIKE '202107%' AND [单位净值] IS NOT NULL;", tolerance=1e-6))
        n += 1
    cases.extend([
        _case(n, "hard", "统计2021年12月31日各基金类型有资产净值记录的基金数，按基金数降序、类型升序返回前5类。",
              "SELECT b.[基金类型], COUNT(*) AS [基金数] FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码] WHERE d.[交易日期] = '20211231' AND d.[资产净值] IS NOT NULL GROUP BY b.[基金类型] ORDER BY [基金数] DESC, b.[基金类型] ASC LIMIT 5;"),
        _case(n + 1, "hard", "找出2021年12月31日资产净值最高的前10只基金，返回代码、简称和资产净值，按资产净值降序、代码升序。",
              "SELECT d.[基金代码], b.[基金简称], d.[资产净值] FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码] WHERE d.[交易日期] = '20211231' AND d.[资产净值] IS NOT NULL ORDER BY d.[资产净值] DESC, d.[基金代码] ASC LIMIT 10;"),
        _case(n + 2, "hard", "统计2021年每个月末有行情记录的基金数量，返回月份和基金数，按月份升序。",
              "SELECT SUBSTR([交易日期], 1, 6) AS [月份], COUNT(DISTINCT [基金代码]) AS [基金数] FROM [基金日行情表] WHERE [交易日期] IN ('20210129','20210226','20210331','20210430','20210531','20210630','20210730','20210831','20210930','20211029','20211130','20211231') GROUP BY SUBSTR([交易日期], 1, 6) ORDER BY [月份] ASC;"),
        _case(n + 3, "hard", "查询基金日行情表中2021年有单位净值记录的不同基金数量和总记录数。",
              "SELECT COUNT(DISTINCT [基金代码]) AS [基金数], COUNT(*) AS [记录数] FROM [基金日行情表] WHERE [交易日期] LIKE '2021%' AND [单位净值] IS NOT NULL;"),
        _empty(n + 4, "查询基金000001在2099年1月1日的交易日期和单位净值；没有记录时返回无记录。",
               "SELECT [交易日期], [单位净值] FROM [基金日行情表] WHERE [基金代码] = '000001' AND [交易日期] = '20990101';"),
    ])
    n += 5

    # 51--75: stock holdings. Anchors guarantee each requested period exists.
    for index in range(10):
        code, day, report = holding_anchors[index]
        cases.append(_case(n, "daily", f"查询基金{code}在{day}、报告类型为“{report}”时排名前3的重仓股，返回排名、股票代码、股票名称、市值和净值占比，按排名升序、股票代码升序。",
            f"SELECT [第N大重仓股], [股票代码], [股票名称], [市值], [市值占基金资产净值比] FROM [基金股票持仓明细] WHERE [基金代码] = '{code}' AND [持仓日期] = '{day}' AND [报告类型] = '{report}' AND [第N大重仓股] <= 3 ORDER BY [第N大重仓股] ASC, [股票代码] ASC;"))
        n += 1
    for index in range(5):
        code, day, report = holding_anchors[10 + index]
        cases.append(_case(n, "hard", f"统计基金{code}在{day}、报告类型为“{report}”的前十大重仓股记录数和市值合计。",
            f"SELECT COUNT(*) AS [记录数], ROUND(SUM([市值]), 2) AS [前十大市值合计] FROM [基金股票持仓明细] WHERE [基金代码] = '{code}' AND [持仓日期] = '{day}' AND [报告类型] = '{report}' AND [第N大重仓股] BETWEEN 1 AND 10;", tolerance=0.01))
        n += 1
    for index in range(5):
        code, day, report = holding_anchors[15 + index]
        cases.append(_case(n, "hard", f"找出基金{code}在{day}、报告类型为“{report}”持仓数量最多的股票，返回代码、名称、数量和市值；并列时股票代码升序。",
            f"SELECT [股票代码], [股票名称], [数量], [市值] FROM [基金股票持仓明细] WHERE [基金代码] = '{code}' AND [持仓日期] = '{day}' AND [报告类型] = '{report}' ORDER BY [数量] DESC, [股票代码] ASC LIMIT 1;"))
        n += 1
    # Global holdings queries retain a bounded, repeatable result set.
    cases.extend([
        _case(n, "hard", "统计各报告类型的持仓记录数和不同基金数，按记录数降序、报告类型升序。",
              "SELECT [报告类型], COUNT(*) AS [持仓记录数], COUNT(DISTINCT [基金代码]) AS [基金数] FROM [基金股票持仓明细] GROUP BY [报告类型] ORDER BY [持仓记录数] DESC, [报告类型] ASC;"),
        _case(n + 1, "hard", "找出年报(含半年报)持仓中市值最高的前10条记录，返回基金代码、基金简称、持仓日期、股票代码、股票名称和市值，按市值降序、基金代码升序。",
              "SELECT [基金代码], [基金简称], [持仓日期], [股票代码], [股票名称], [市值] FROM [基金股票持仓明细] WHERE [报告类型] = '年报(含半年报)' AND [市值] IS NOT NULL ORDER BY [市值] DESC, [基金代码] ASC LIMIT 10;"),
        _case(n + 2, "hard", "按持仓日期统计年报(含半年报)中不同股票代码数量，按持仓日期升序返回前10个日期。",
              "SELECT [持仓日期], COUNT(DISTINCT [股票代码]) AS [股票数] FROM [基金股票持仓明细] WHERE [报告类型] = '年报(含半年报)' GROUP BY [持仓日期] ORDER BY [持仓日期] ASC LIMIT 10;"),
        _case(n + 3, "hard", "统计季报持仓记录中净值占比大于等于0.05的记录数和不同基金数。",
              "SELECT COUNT(*) AS [记录数], COUNT(DISTINCT [基金代码]) AS [基金数] FROM [基金股票持仓明细] WHERE [报告类型] = '季报' AND [市值占基金资产净值比] >= 0.05;"),
        _empty(n + 4, "查询股票代码999999在2020年12月31日年报(含半年报)中的持仓基金；没有记录时返回无记录。",
               "SELECT [基金代码], [基金简称], [市值] FROM [基金股票持仓明细] WHERE [股票代码] = '999999' AND [持仓日期] = '20201231' AND [报告类型] = '年报(含半年报)';"),
    ])
    n += 5

    # 76--90: scale reports, explicitly at report-period grain.
    for index in range(10):
        code, year, cutoff = scale_anchors[index]
        cases.append(_case(n, "hard", f"查询基金{code}在{year}年度、截止日期为{cutoff}的期初总份额、申购份额、赎回份额和期末总份额。",
            f"SELECT [报告期期初基金总份额], [报告期基金总申购份额], [报告期基金总赎回份额], [报告期期末基金总份额] FROM [基金规模变动表] WHERE [基金代码] = '{code}' AND [定期报告所属年度] = {year} AND [截止日期] = '{cutoff}';"))
        n += 1
    cases.extend([
        _case(n, "hard", "统计2020年度基金规模变动表中期末份额小于期初份额的记录数，以及这些记录的份额净减少量合计。",
              "SELECT COUNT(*) AS [记录数], ROUND(SUM([报告期期初基金总份额] - [报告期期末基金总份额]), 2) AS [份额净减少量合计] FROM [基金规模变动表] WHERE [定期报告所属年度] = 2020 AND [报告期期初基金总份额] IS NOT NULL AND [报告期期末基金总份额] IS NOT NULL AND [报告期期末基金总份额] < [报告期期初基金总份额];", tolerance=0.01),
        _case(n + 1, "hard", "找出2020年度赎回份额最大的前5条基金规模变动记录，返回基金代码、简称、截止日期、赎回份额和期末总份额，按赎回份额降序、基金代码升序。",
              "SELECT [基金代码], [基金简称], [截止日期], [报告期基金总赎回份额], [报告期期末基金总份额] FROM [基金规模变动表] WHERE [定期报告所属年度] = 2020 AND [报告期基金总赎回份额] IS NOT NULL ORDER BY [报告期基金总赎回份额] DESC, [基金代码] ASC LIMIT 5;"),
        _case(n + 2, "hard", "按2020年度报告截止日期统计不同基金数和记录数，按截止日期升序。",
              "SELECT [截止日期], COUNT(DISTINCT [基金代码]) AS [基金数], COUNT(*) AS [记录数] FROM [基金规模变动表] WHERE [定期报告所属年度] = 2020 GROUP BY [截止日期] ORDER BY [截止日期] ASC;"),
        _case(n + 3, "hard", "查询基金规模变动表中2020年度期末总份额不为空的不同基金数。",
              "SELECT COUNT(DISTINCT [基金代码]) AS [基金数] FROM [基金规模变动表] WHERE [定期报告所属年度] = 2020 AND [报告期期末基金总份额] IS NOT NULL;"),
        _empty(n + 4, "查询基金000001在2099年度的基金规模变动记录；没有记录时返回无记录。",
               "SELECT [截止日期], [报告期期末基金总份额] FROM [基金规模变动表] WHERE [基金代码] = '000001' AND [定期报告所属年度] = 2099;"),
    ])
    n += 5

    # 91--100: profile-to-fact joins and comparison-friendly result sets.
    cases.extend([
        _case(n, "hard", "找出2021年12月31日资产净值最高的前5只混合型基金，返回代码、简称、管理人和资产净值，按资产净值降序、代码升序。",
              "SELECT d.[基金代码], b.[基金简称], b.[管理人], d.[资产净值] FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码] WHERE d.[交易日期] = '20211231' AND b.[基金类型] = '混合型' AND d.[资产净值] IS NOT NULL ORDER BY d.[资产净值] DESC, d.[基金代码] ASC LIMIT 5;"),
        _case(n + 1, "hard", "按基金类型统计2021年12月31日的平均单位净值，仅保留至少100只基金的类型，按平均单位净值降序、类型升序。",
              "SELECT b.[基金类型], COUNT(*) AS [基金数], ROUND(AVG(d.[单位净值]), 6) AS [平均单位净值] FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码] WHERE d.[交易日期] = '20211231' AND d.[单位净值] IS NOT NULL GROUP BY b.[基金类型] HAVING COUNT(*) >= 100 ORDER BY [平均单位净值] DESC, b.[基金类型] ASC;", tolerance=1e-6),
        _case(n + 2, "hard", "找出2020年12月31日年报(含半年报)中重仓股排名第一且市值最高的前10条记录，返回基金代码、基金简称、管理人、股票代码、股票名称和市值。",
              "SELECT h.[基金代码], h.[基金简称], b.[管理人], h.[股票代码], h.[股票名称], h.[市值] FROM [基金股票持仓明细] h JOIN [基金基本信息] b ON b.[基金代码] = h.[基金代码] WHERE h.[持仓日期] = '20201231' AND h.[报告类型] = '年报(含半年报)' AND h.[第N大重仓股] = 1 ORDER BY h.[市值] DESC, h.[基金代码] ASC LIMIT 10;"),
        _case(n + 3, "hard", "统计2020年12月31日年报(含半年报)中每种基金类型的持仓记录数，按记录数降序、基金类型升序返回前5类。",
              "SELECT b.[基金类型], COUNT(*) AS [持仓记录数] FROM [基金股票持仓明细] h JOIN [基金基本信息] b ON b.[基金代码] = h.[基金代码] WHERE h.[持仓日期] = '20201231' AND h.[报告类型] = '年报(含半年报)' GROUP BY b.[基金类型] ORDER BY [持仓记录数] DESC, b.[基金类型] ASC LIMIT 5;"),
        _case(n + 4, "hard", "找出2020年度期末总份额最高的前5条规模变动记录，返回基金代码、基金简称、基金类型、截止日期和期末总份额，按期末总份额降序、基金代码升序。",
              "SELECT s.[基金代码], s.[基金简称], b.[基金类型], s.[截止日期], s.[报告期期末基金总份额] FROM [基金规模变动表] s JOIN [基金基本信息] b ON b.[基金代码] = s.[基金代码] WHERE s.[定期报告所属年度] = 2020 AND s.[报告期期末基金总份额] IS NOT NULL ORDER BY s.[报告期期末基金总份额] DESC, s.[基金代码] ASC LIMIT 5;"),
        _case(n + 5, "hard", "统计2020年度每种基金类型的规模变动记录数和不同基金数，按记录数降序、基金类型升序返回前5类。",
              "SELECT b.[基金类型], COUNT(*) AS [记录数], COUNT(DISTINCT s.[基金代码]) AS [基金数] FROM [基金规模变动表] s JOIN [基金基本信息] b ON b.[基金代码] = s.[基金代码] WHERE s.[定期报告所属年度] = 2020 GROUP BY b.[基金类型] ORDER BY [记录数] DESC, b.[基金类型] ASC LIMIT 5;"),
        _case(n + 6, "hard", "查询2021年12月31日同时有资产净值且基金类型为股票型的基金数量和资产净值合计。",
              "SELECT COUNT(*) AS [基金数], ROUND(SUM(d.[资产净值]), 2) AS [资产净值合计] FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码] WHERE d.[交易日期] = '20211231' AND b.[基金类型] = '股票型' AND d.[资产净值] IS NOT NULL;", tolerance=0.01),
        _case(n + 7, "hard", "找出2021年12月31日单位净值最高的前5只基金，返回代码、简称、基金类型、单位净值和累计单位净值，按单位净值降序、代码升序。",
              "SELECT d.[基金代码], b.[基金简称], b.[基金类型], d.[单位净值], d.[累计单位净值] FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码] WHERE d.[交易日期] = '20211231' AND d.[单位净值] IS NOT NULL ORDER BY d.[单位净值] DESC, d.[基金代码] ASC LIMIT 5;"),
        _case(n + 8, "hard", "统计2020年12月31日年报(含半年报)中每个管理人的不同基金数，按基金数降序、管理人升序返回前10位。",
              "SELECT b.[管理人], COUNT(DISTINCT h.[基金代码]) AS [基金数] FROM [基金股票持仓明细] h JOIN [基金基本信息] b ON b.[基金代码] = h.[基金代码] WHERE h.[持仓日期] = '20201231' AND h.[报告类型] = '年报(含半年报)' GROUP BY b.[管理人] ORDER BY [基金数] DESC, b.[管理人] ASC LIMIT 10;"),
        _case(n + 9, "hard", "统计2020年12月31日有资产净值记录的基金中，各管理人的基金数，按基金数降序、管理人升序返回前10位。",
              "SELECT b.[管理人], COUNT(*) AS [基金数] FROM [基金日行情表] d JOIN [基金基本信息] b ON b.[基金代码] = d.[基金代码] WHERE d.[交易日期] = '20201231' AND d.[资产净值] IS NOT NULL GROUP BY b.[管理人] ORDER BY [基金数] DESC, b.[管理人] ASC LIMIT 10;"),
    ])

    # Replace repeated four-table cases with 30 cases spanning every table that
    # was absent from the original Text-to-SQL slice (five cases per table).
    def anchors(table: str, date_column: str, limit: int = 4) -> list[tuple[str, str]]:
        return [(str(row[0]), str(row[1])) for row in connection.execute(
            f"SELECT [股票代码], [{date_column}] FROM [{table}] "
            f"WHERE [股票代码] IS NOT NULL AND [{date_column}] IS NOT NULL "
            f"GROUP BY [股票代码], [{date_column}] ORDER BY [{date_column}], [股票代码] LIMIT {limit}"
        )]

    industry = anchors("A股公司行业划分表", "交易日期")
    a_daily = anchors("A股票日行情表", "交易日")
    hk_daily = anchors("港股票日行情表", "交易日")
    holder = [(str(a), str(b)) for a, b in connection.execute(
        "SELECT [基金代码], [截止日期] FROM [基金份额持有人结构] "
        "WHERE [基金代码] IS NOT NULL AND [截止日期] IS NOT NULL "
        "GROUP BY [基金代码], [截止日期] ORDER BY [截止日期], [基金代码] LIMIT 4"
    )]
    bond = [(str(a), str(b), str(c)) for a, b, c in connection.execute(
        "SELECT [基金代码], [持仓日期], [报告类型] FROM [基金债券持仓明细] "
        "GROUP BY [基金代码], [持仓日期], [报告类型] HAVING COUNT(*) >= 3 "
        "ORDER BY [持仓日期], [基金代码] LIMIT 4"
    )]
    convertible = [(str(a), str(b), str(c)) for a, b, c in connection.execute(
        "SELECT [基金代码], [持仓日期], [报告类型] FROM [基金可转债持仓明细] "
        "GROUP BY [基金代码], [持仓日期], [报告类型] HAVING COUNT(*) >= 1 "
        "ORDER BY [持仓日期], [基金代码] LIMIT 4"
    )]
    if min(map(len, (industry, a_daily, hk_daily, holder, bond, convertible))) < 4:
        raise RuntimeError("newly supported tables lack benchmark anchors")

    replacements = [
        _case(11, "daily", f"查询A股股票{industry[0][0]}在{industry[0][1]}的行业划分标准、一级行业和二级行业。",
              f"SELECT [行业划分标准], [一级行业名称], [二级行业名称] FROM [A股公司行业划分表] WHERE [股票代码]='{industry[0][0]}' AND [交易日期]='{industry[0][1]}';"),
        _case(12, "daily", f"查询A股股票{industry[1][0]}在{industry[1][1]}的一级行业和二级行业。",
              f"SELECT [一级行业名称], [二级行业名称] FROM [A股公司行业划分表] WHERE [股票代码]='{industry[1][0]}' AND [交易日期]='{industry[1][1]}';"),
        _case(13, "hard", f"统计{industry[2][1]}各一级行业包含的不同股票数，按股票数降序、一级行业升序返回前10类。",
              f"SELECT [一级行业名称], COUNT(DISTINCT [股票代码]) AS [股票数] FROM [A股公司行业划分表] WHERE [交易日期]='{industry[2][1]}' GROUP BY [一级行业名称] ORDER BY [股票数] DESC, [一级行业名称] ASC LIMIT 10;"),
        _case(14, "hard", f"统计{industry[3][1]}使用各行业划分标准的股票记录数，按记录数降序、标准升序。",
              f"SELECT [行业划分标准], COUNT(*) AS [记录数] FROM [A股公司行业划分表] WHERE [交易日期]='{industry[3][1]}' GROUP BY [行业划分标准] ORDER BY [记录数] DESC, [行业划分标准] ASC;"),
        _empty(15, "查询A股股票999999在2099年1月1日的行业分类；没有记录时返回无记录。",
               "SELECT [一级行业名称], [二级行业名称] FROM [A股公司行业划分表] WHERE [股票代码]='999999' AND [交易日期]='20990101';"),

        _case(16, "daily", f"查询A股股票{a_daily[0][0]}在{a_daily[0][1]}的开盘价、最高价、最低价和收盘价。",
              f"SELECT [今开盘(元)], [最高价(元)], [最低价(元)], [收盘价(元)] FROM [A股票日行情表] WHERE [股票代码]='{a_daily[0][0]}' AND [交易日]='{a_daily[0][1]}';"),
        _case(17, "daily", f"查询A股股票{a_daily[1][0]}在{a_daily[1][1]}的成交量和成交金额。",
              f"SELECT [成交量(股)], [成交金额(元)] FROM [A股票日行情表] WHERE [股票代码]='{a_daily[1][0]}' AND [交易日]='{a_daily[1][1]}';"),
        _case(18, "hard", f"找出A股股票{a_daily[2][0]}在{a_daily[2][1][:6]}月收盘价最高的交易日和收盘价；并列时日期最早。",
              f"SELECT [交易日], [收盘价(元)] FROM [A股票日行情表] WHERE [股票代码]='{a_daily[2][0]}' AND [交易日] LIKE '{a_daily[2][1][:6]}%' ORDER BY [收盘价(元)] DESC, [交易日] ASC LIMIT 1;"),
        _case(19, "hard", f"统计A股股票{a_daily[3][0]}在{a_daily[3][1][:4]}年的交易日数和平均收盘价。",
              f"SELECT COUNT(*) AS [交易日数], ROUND(AVG([收盘价(元)]), 4) AS [平均收盘价] FROM [A股票日行情表] WHERE [股票代码]='{a_daily[3][0]}' AND [交易日] LIKE '{a_daily[3][1][:4]}%';", tolerance=0.0001),
        _case(20, "hard", f"找出{a_daily[0][1]}成交金额最高的前5只A股，按成交金额降序、股票代码升序。",
              f"SELECT [股票代码], [成交金额(元)] FROM [A股票日行情表] WHERE [交易日]='{a_daily[0][1]}' ORDER BY [成交金额(元)] DESC, [股票代码] ASC LIMIT 5;"),

        _case(66, "daily", f"查询港股股票{hk_daily[0][0]}在{hk_daily[0][1]}的开盘价、最高价、最低价和收盘价。",
              f"SELECT [今开盘(元)], [最高价(元)], [最低价(元)], [收盘价(元)] FROM [港股票日行情表] WHERE [股票代码]='{hk_daily[0][0]}' AND [交易日]='{hk_daily[0][1]}';"),
        _case(67, "daily", f"查询港股股票{hk_daily[1][0]}在{hk_daily[1][1]}的成交量和成交金额。",
              f"SELECT [成交量(股)], [成交金额(元)] FROM [港股票日行情表] WHERE [股票代码]='{hk_daily[1][0]}' AND [交易日]='{hk_daily[1][1]}';"),
        _case(68, "hard", f"找出港股股票{hk_daily[2][0]}在{hk_daily[2][1][:6]}月收盘价最低的交易日和收盘价；并列时日期最早。",
              f"SELECT [交易日], [收盘价(元)] FROM [港股票日行情表] WHERE [股票代码]='{hk_daily[2][0]}' AND [交易日] LIKE '{hk_daily[2][1][:6]}%' ORDER BY [收盘价(元)] ASC, [交易日] ASC LIMIT 1;"),
        _case(69, "hard", f"统计港股股票{hk_daily[3][0]}在{hk_daily[3][1][:4]}年的交易日数和平均成交量。",
              f"SELECT COUNT(*) AS [交易日数], ROUND(AVG([成交量(股)]), 2) AS [平均成交量] FROM [港股票日行情表] WHERE [股票代码]='{hk_daily[3][0]}' AND [交易日] LIKE '{hk_daily[3][1][:4]}%';", tolerance=0.01),
        _case(70, "hard", f"找出{hk_daily[0][1]}收盘价最高的前5只港股，按收盘价降序、股票代码升序。",
              f"SELECT [股票代码], [收盘价(元)] FROM [港股票日行情表] WHERE [交易日]='{hk_daily[0][1]}' ORDER BY [收盘价(元)] DESC, [股票代码] ASC LIMIT 5;"),

        _case(71, "daily", f"查询基金{holder[0][0]}在截止日期{holder[0][1]}的机构和个人投资者持有份额及占比。",
              f"SELECT [机构投资者持有的基金份额], [机构投资者持有的基金份额占总份额比例], [个人投资者持有的基金份额], [个人投资者持有的基金份额占总份额比例] FROM [基金份额持有人结构] WHERE [基金代码]='{holder[0][0]}' AND [截止日期]='{holder[0][1]}';"),
        _case(72, "daily", f"查询基金{holder[1][0]}在截止日期{holder[1][1]}的公告日期、报告类型和机构投资者占比。",
              f"SELECT [公告日期], [报告类型], [机构投资者持有的基金份额占总份额比例] FROM [基金份额持有人结构] WHERE [基金代码]='{holder[1][0]}' AND [截止日期]='{holder[1][1]}';"),
        _case(73, "hard", f"找出截止日期{holder[2][1]}机构投资者持有比例最高的前5只基金，按比例降序、基金代码升序。",
              f"SELECT [基金代码], [基金简称], [机构投资者持有的基金份额占总份额比例] FROM [基金份额持有人结构] WHERE [截止日期]='{holder[2][1]}' ORDER BY [机构投资者持有的基金份额占总份额比例] DESC, [基金代码] ASC LIMIT 5;"),
        _case(74, "hard", f"统计{holder[3][1][:4]}年度有持有人结构记录的不同基金数。",
              f"SELECT COUNT(DISTINCT [基金代码]) AS [基金数] FROM [基金份额持有人结构] WHERE [定期报告所属年度]={int(holder[3][1][:4])};"),
        _empty(75, "查询基金999998在2099年度的持有人结构；没有记录时返回无记录。",
               "SELECT [截止日期], [机构投资者持有的基金份额占总份额比例] FROM [基金份额持有人结构] WHERE [基金代码]='999998' AND [定期报告所属年度]=2099;"),

        _case(81, "daily", f"查询基金{bond[0][0]}在{bond[0][1]}、报告类型为“{bond[0][2]}”的前3大债券持仓，返回排名、债券类型、名称、数量和市值。",
              f"SELECT [第N大重仓股], [债券类型], [债券名称], [持债数量], [持债市值] FROM [基金债券持仓明细] WHERE [基金代码]='{bond[0][0]}' AND [持仓日期]='{bond[0][1]}' AND [报告类型]='{bond[0][2]}' AND [第N大重仓股]<=3 ORDER BY [第N大重仓股] ASC, [债券名称] ASC;"),
        _case(82, "daily", f"统计基金{bond[1][0]}在{bond[1][1]}、报告类型为“{bond[1][2]}”的债券持仓记录数和持债市值合计。",
              f"SELECT COUNT(*) AS [记录数], ROUND(SUM([持债市值]), 2) AS [持债市值合计] FROM [基金债券持仓明细] WHERE [基金代码]='{bond[1][0]}' AND [持仓日期]='{bond[1][1]}' AND [报告类型]='{bond[1][2]}';", tolerance=0.01),
        _case(83, "hard", f"找出基金{bond[2][0]}在{bond[2][1]}、报告类型为“{bond[2][2]}”持债市值最高的债券。",
              f"SELECT [债券类型], [债券名称], [持债市值], [持债市值占基金资产净值比] FROM [基金债券持仓明细] WHERE [基金代码]='{bond[2][0]}' AND [持仓日期]='{bond[2][1]}' AND [报告类型]='{bond[2][2]}' ORDER BY [持债市值] DESC, [债券名称] ASC LIMIT 1;"),
        _case(84, "hard", f"按债券类型统计{bond[3][1]}的持债市值合计，按合计降序、债券类型升序返回前5类。",
              f"SELECT [债券类型], ROUND(SUM([持债市值]), 2) AS [持债市值合计] FROM [基金债券持仓明细] WHERE [持仓日期]='{bond[3][1]}' GROUP BY [债券类型] ORDER BY [持债市值合计] DESC, [债券类型] ASC LIMIT 5;", tolerance=0.01),
        _empty(85, "查询基金999998在2099年1月1日的债券持仓；没有记录时返回无记录。",
               "SELECT [债券名称], [持债市值] FROM [基金债券持仓明细] WHERE [基金代码]='999998' AND [持仓日期]='20990101';"),

        _case(86, "daily", f"查询基金{convertible[0][0]}在{convertible[0][1]}、报告类型为“{convertible[0][2]}”的可转债持仓，返回对应股票代码、债券名称、数量、市值和排名。",
              f"SELECT [对应股票代码], [债券名称], [数量], [市值], [第N大重仓股] FROM [基金可转债持仓明细] WHERE [基金代码]='{convertible[0][0]}' AND [持仓日期]='{convertible[0][1]}' AND [报告类型]='{convertible[0][2]}' ORDER BY [第N大重仓股] ASC, [债券名称] ASC LIMIT 10;"),
        _case(87, "daily", f"统计基金{convertible[1][0]}在{convertible[1][1]}、报告类型为“{convertible[1][2]}”的可转债记录数和市值合计。",
              f"SELECT COUNT(*) AS [记录数], ROUND(SUM([市值]), 2) AS [可转债市值合计] FROM [基金可转债持仓明细] WHERE [基金代码]='{convertible[1][0]}' AND [持仓日期]='{convertible[1][1]}' AND [报告类型]='{convertible[1][2]}';", tolerance=0.01),
        _case(88, "hard", f"找出基金{convertible[2][0]}在{convertible[2][1]}、报告类型为“{convertible[2][2]}”市值最高的可转债。",
              f"SELECT [对应股票代码], [债券名称], [市值], [市值占基金资产净值比] FROM [基金可转债持仓明细] WHERE [基金代码]='{convertible[2][0]}' AND [持仓日期]='{convertible[2][1]}' AND [报告类型]='{convertible[2][2]}' ORDER BY [市值] DESC, [债券名称] ASC LIMIT 1;"),
        _case(89, "hard", f"统计{convertible[3][1]}各报告类型的可转债记录数和市值合计，按记录数降序、报告类型升序。",
              f"SELECT [报告类型], COUNT(*) AS [记录数], ROUND(SUM([市值]), 2) AS [市值合计] FROM [基金可转债持仓明细] WHERE [持仓日期]='{convertible[3][1]}' GROUP BY [报告类型] ORDER BY [记录数] DESC, [报告类型] ASC;", tolerance=0.01),
        _empty(90, "查询基金999998在2099年1月1日的可转债持仓；没有记录时返回无记录。",
               "SELECT [债券名称], [市值] FROM [基金可转债持仓明细] WHERE [基金代码]='999998' AND [持仓日期]='20990101';"),
    ]
    replacement_by_id = {row["case_id"]: row for row in replacements}
    cases = [replacement_by_id.get(row["case_id"], row) for row in cases]
    if len(cases) != 100 or n + 9 != 100:
        raise AssertionError(f"expected 100 cases, got {len(cases)} / last {n + 9}")
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("case ids must be unique")
    return cases


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{DB.resolve().as_posix()}?mode=ro", uri=True) as connection:
        cases = build_cases(connection)
    seed_path = OUT / "sql_gold_v0.2_seed.json"
    gold_path = OUT / "sql_gold_v0.2.json"
    seed_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    gold = materialize(seed_path, DB)
    gold_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    difficulty_counts = dict(sorted(Counter(row["difficulty"] for row in gold).items()))
    manifest = {
        "dataset": "finagent-eval-v2-text2sql",
        "version": "0.2",
        "split": "testrun003-frozen-evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(gold),
        "difficulty_counts": difficulty_counts,
        "case_ids": [row["case_id"] for row in gold],
        "source_files": [
            {"path": str(seed_path.relative_to(ROOT)).replace('\\', '/'), "sha256": _hash(seed_path), "case_count": len(cases)},
            {"path": str(gold_path.relative_to(ROOT)).replace('\\', '/'), "sha256": _hash(gold_path), "case_count": len(gold)},
        ],
        "database_sha256": _hash(DB),
        "policy": {
            "agent_outputs_used_to_author_gold": False,
            "may_tune_on_split": False,
            "sql_accuracy_definition": "Executed SQL result set matches frozen Gold rows, order and numeric tolerance.",
        },
    }
    (OUT / "manifest_v0.2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# testrun003 — 100 题 Text-to-SQL 冻结测评集\n\n"
        "本集仅包含 Text-to-SQL：100 条自然语言题、只读参考 SQL 与从固定 SQLite 数据库物化的 Gold 结果。"
        "题目覆盖 finance.db 的全部 10 张业务表。SQL 准确率按执行结果集与 Gold 的匹配率计算，"
        "不按 SQL 文本字符串匹配。\n\n"
        "- `sql_gold_v0.2_seed.json`：题目、参考 SQL 与评分语义。\n"
        "- `sql_gold_v0.2.json`：已冻结的 Gold 答案。\n"
        "- `manifest_v0.2.json`：文件哈希、数据库哈希与冻结策略。\n\n"
        "## 运行 testrun003\n\n"
        "使用 `run_e2e_calibration.py` 对 `sql_gold_v0.2.json` 运行 100 题，"
        "将运行记录写入 `runs/e2e_text2sql.jsonl`、Trace 写入 `traces/`；"
        "再用 `score_sql_execution_correctness.py` 生成结果集准确率。"
        "冻结后不得修改本目录下的 Gold 文件。\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(gold), "difficulty_counts": difficulty_counts, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
