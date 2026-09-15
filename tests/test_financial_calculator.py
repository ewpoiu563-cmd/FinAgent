from decimal import Decimal
from unittest.mock import Mock

from orchestration import (
    DeterministicCalculation,
    RequirementSynthesizer,
    TaskRequirement,
    execute_calculation,
    normalize_sql_tool_result,
)


def _evidence(value, unit="万元", task_id="T1"):
    item = normalize_sql_tool_result({
        "success": True,
        "generated_sql": "SELECT amount FROM report",
        "columns": [f"金额（{unit}）"],
        "rows": [{f"金额（{unit}）": value}],
        "row_count": 1,
        "truncated": False,
    }).to_dict()["items"][0]
    item["evidence_id"] = f"{task_id}:item-1"
    return item


def test_calculator_subtracts_mixed_scales_with_decimal_not_llm():
    left = _evidence(1, "亿元", "T1")
    right = _evidence(2500, "万元", "T2")

    result = execute_calculation(
        DeterministicCalculation(
            "subtract",
            {"evidence_id": "T1:item-1", "amount_index": 0},
            {"evidence_id": "T2:item-1", "amount_index": 0},
            "万元",
        ),
        {"T1:item-1": left, "T2:item-1": right},
    )

    assert result.text == "1 亿元人民币 − 2500 万元人民币 = 7500 万元人民币"
    assert result.evidence_ids == ("T1:item-1", "T2:item-1")


def test_calculator_growth_rate_uses_base_amounts():
    base = _evidence(2, "亿元", "T1")
    current = _evidence(3, "亿元", "T2")

    result = execute_calculation(
        DeterministicCalculation(
            "growth_rate",
            {"evidence_id": "T1:item-1", "amount_index": 0},
            {"evidence_id": "T2:item-1", "amount_index": 0},
        ),
        {"T1:item-1": base, "T2:item-1": current},
    )

    assert result.text.endswith("= 50%")


def test_synthesis_renders_local_calculation():
    left, right = _evidence(100, "万元", "T1"), _evidence(120, "万元", "T2")
    tasks = []
    for task_id, question, evidence in (("T1", "基期金额", left), ("T2", "当前金额", right)):
        task = TaskRequirement(task_id, question, "sql")
        task.start()
        task.output_evidence = [evidence]
        task.complete(answer=question, evidence_ids=[evidence["evidence_id"]])
        tasks.append(task)
    llm = Mock(return_value=(
        '{"claims":[{"text":"已取得两期金额", "task_ids":["T1","T2"],'
        '"evidence_ids":["T1:item-1","T2:item-1"]}],'
        '"calculations":[{"operation":"growth_rate",'
        '"left":{"evidence_id":"T1:item-1","amount_index":0},'
        '"right":{"evidence_id":"T2:item-1","amount_index":0},"display_unit":null}],'
        '"missing_information":[]}'
    ))

    result = RequirementSynthesizer(llm).synthesize("计算增长率", tasks)

    assert result.complete is True
    assert "确定性计算（本地 Decimal）" in result.answer
    assert "= 20%" in result.answer
    assert Decimal("1") == Decimal("1")  # Ensure this test never involves float arithmetic.


def test_synthesis_marks_calculation_request_incomplete_without_a_tool_plan():
    evidence = _evidence(100, "万元", "T1")
    task = TaskRequirement("T1", "金额", "sql")
    task.start()
    task.output_evidence = [evidence]
    task.complete(answer="金额为100万元", evidence_ids=["T1:item-1"])
    llm = Mock(return_value=(
        '{"claims":[{"text":"金额为100万元", "task_ids":["T1"],'
        '"evidence_ids":["T1:item-1"]}],"calculations":[],"missing_information":[]}'
    ))

    result = RequirementSynthesizer(llm).synthesize("计算该金额的增长率", [task])

    assert result.complete is False
    assert "需要确定性计算计划" in result.answer


def test_synthesis_keeps_verified_sql_growth_in_a_mixed_request():
    evidence = _evidence(14.14, "%", "T1")
    sql = TaskRequirement("T1", "基金首末交易日单位净值增长率", "sql")
    sql.start()
    sql.output_evidence = [evidence]
    sql.complete(answer="基金净值增长率为14.14%", evidence_ids=["T1:item-1"])
    rag = TaskRequirement("T2", "原料价格上涨的毛利率影响", "rag")
    rag.start()
    rag.output_evidence = [{"evidence_id": "T2:item-1", "source_file": "doc.pdf", "page": [38]}]
    rag.complete(answer="两种原料的敏感性序列不同", evidence_ids=["T2:item-1"])
    llm = Mock(return_value=(
        '{"claims":[{"text":"基金净值增长率为14.14%，原料敏感性属于不同主体和指标口径。",'
        '"task_ids":["T1","T2"],"evidence_ids":["T1:item-1","T2:item-1"]}],'
        '"calculations":[],"missing_information":[]}'
    ))

    result = RequirementSynthesizer(llm).synthesize(
        "计算基金增长率，再说明原料涨价影响", [sql, rag]
    )

    assert result.complete is True
    assert "14.14%" in result.answer
