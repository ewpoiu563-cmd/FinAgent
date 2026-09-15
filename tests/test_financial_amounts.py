from decimal import Decimal

import pytest
from unittest.mock import Mock

from orchestration import (
    AmountSemanticError,
    ExchangeRate,
    amount_ratio,
    convert_currency,
    extract_money_amounts,
    extract_sql_money_amounts,
    normalize_sql_tool_result,
    normalize_web_tool_result,
    RequirementSynthesizer,
    subtract_money,
    TaskRequirement,
)


def test_extracts_currency_and_scale_into_base_units():
    amounts = extract_money_amounts("营业收入为人民币1.25亿元，海外收入为USD 3.5 million。")

    assert [item.currency for item in amounts] == ["CNY", "USD"]
    assert amounts[0].display_unit == "亿元"
    assert amounts[0].base_value == Decimal("125000000")
    assert amounts[1].display_unit == "million"
    assert amounts[1].base_value == Decimal("3500000")


def test_yuan_units_imply_cny_and_bare_numbers_are_ignored():
    amounts = extract_money_amounts("本期100万元，上期800000元，样本数2026。")

    assert [(item.currency, item.base_value) for item in amounts] == [
        ("CNY", Decimal("1000000")),
        ("CNY", Decimal("800000")),
    ]


def test_sql_column_and_row_metadata_are_normalized():
    amounts = extract_sql_money_amounts((
        {"营业收入（万元）": 12.5, "币种": "人民币"},
        {"净利润": "2.5", "币种": "美元", "金额单位": "百万元"},
    ))

    assert amounts[0].to_dict()["base_value"] == "125000"
    assert amounts[0].currency == "CNY"
    assert amounts[1].base_value == Decimal("2500000")
    assert amounts[1].currency == "USD"


def test_same_currency_operations_use_base_values_across_scales():
    left = extract_money_amounts("1亿元")[0]
    right = extract_money_amounts("2500万元")[0]

    difference = subtract_money(left, right, display_unit="万元")

    assert difference.value == Decimal("7500")
    assert difference.base_value == Decimal("75000000")
    assert amount_ratio(left, right) == Decimal("4")


def test_cross_currency_operations_require_exchange_rate_and_date():
    cny = extract_money_amounts("人民币1亿元")[0]
    usd = extract_money_amounts("100万美元")[0]

    with pytest.raises(AmountSemanticError, match="exchange rate and rate date"):
        subtract_money(cny, usd)
    with pytest.raises(AmountSemanticError, match="exchange rate and rate date"):
        amount_ratio(cny, usd)


def test_dated_exchange_rate_enables_explicit_currency_conversion():
    usd = extract_money_amounts("100万美元")[0]
    rate = ExchangeRate("USD", "CNY", Decimal("7.2"), "2026-09-13")

    cny = convert_currency(usd, "人民币", rate, display_unit="万元")

    assert cny.currency == "CNY"
    assert cny.value == Decimal("720")
    assert cny.base_value == Decimal("7200000")
    assert "2026-09-13" in cny.original_text


def test_sql_evidence_exposes_structured_monetary_values():
    bundle = normalize_sql_tool_result({
        "success": True,
        "generated_sql": "SELECT revenue FROM report",
        "columns": ["营业收入（亿元）"],
        "rows": [{"营业收入（亿元）": 1.2}],
        "row_count": 1,
        "truncated": False,
    })

    value = bundle.to_dict()["items"][0]["monetary_values"][0]
    assert value == {
        "value": "1.2",
        "currency": "CNY",
        "display_unit": "亿元",
        "multiplier": "100000000",
        "base_value": "120000000",
        "original_text": "营业收入（亿元）=1.2 亿元",
    }


def test_web_evidence_exposes_explicit_currency_and_scale():
    bundle = normalize_web_tool_result([
        {"title": "公告", "url": "https://example.com", "snippet": "交易对价为2.3亿美元"}
    ])

    value = bundle.to_dict()["items"][0]["monetary_values"][0]
    assert value["currency"] == "USD"
    assert value["display_unit"] == "亿"
    assert value["base_value"] == "230000000"


def test_requirement_synthesis_receives_normalized_amount_contract():
    evidence = normalize_sql_tool_result({
        "success": True,
        "generated_sql": "SELECT revenue FROM report",
        "columns": ["营业收入（万元）"],
        "rows": [{"营业收入（万元）": 12}],
        "row_count": 1,
        "truncated": False,
    }).to_dict()["items"][0]
    evidence["evidence_id"] = "T1:item-1"
    task = TaskRequirement("T1", "营业收入是多少？", "sql", expected_output="营业收入")
    task.start()
    task.output_evidence = [evidence]
    task.complete(answer="12万元", evidence_ids=["T1:item-1"])
    llm = Mock(return_value=(
        '{"claims":[{"text":"营业收入为12万元人民币",'
        '"task_ids":["T1"],"evidence_ids":["T1:item-1"]}],"missing_information":[]}'
    ))

    result = RequirementSynthesizer(llm).synthesize("营业收入是多少？", [task])

    prompt = llm.call_args.args[0]
    assert '"currency": "CNY"' in prompt
    assert '"base_value": "120000"' in prompt
    assert "不同 currency 不得直接比较" in prompt
    assert result.complete is True
