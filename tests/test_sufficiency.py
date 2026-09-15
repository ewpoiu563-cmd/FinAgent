"""Strict offline tests for Phase C1 evidence sufficiency and synthesis."""

import json
from unittest.mock import Mock

import pytest

from orchestration import (
    EvidenceSufficiencyEvaluator,
    SourceScopeResolver,
    load_default_catalog,
    normalize_rag_tool_result,
    validate_rag_evidence_scope,
)


JIANFAN = "6c93724343cef6b9da70d51daa739e8a1c607a22-5d11f946465d"
JIANFAN_FILE = "6c93724343cef6b9da70d51daa739e8a1c607a22.pdf"


def _bundle():
    catalog = load_default_catalog()
    raw = {
        "success": True,
        "evidence": [
            {
                "doc_id": JIANFAN,
                "source_file": JIANFAN_FILE,
                "page": [57],
                "headings": ["风险因素"],
                "retrieval_unit_id": "jianfan-risk-57",
                "rerank_score": 0.91,
                "text": "公司面临市场竞争加剧和产品价格下降风险。",
            }
        ],
    }
    return validate_rag_evidence_scope(
        normalize_rag_tool_result(raw, catalog),
        SourceScopeResolver(catalog).resolve(["doc_jianfan"]),
    )


def test_sufficient_structured_result_uses_only_bundle_fields():
    llm = Mock(return_value=json.dumps({
        "sufficient": True,
        "answer": "公司面临市场竞争加剧和产品价格下降风险。",
        "missing_information": [],
        "supported_evidence_ids": ["jianfan-risk-57"],
    }, ensure_ascii=False))

    result = EvidenceSufficiencyEvaluator(llm, timeout=12).evaluate("核心经营风险？", _bundle())

    assert result.sufficient is True
    assert result.generation_failure is False
    assert result.supported_evidence_ids == ("jianfan-risk-57",)
    prompt = llm.call_args.args[0]
    assert "original_question" in prompt and "jianfan-risk-57" in prompt
    assert "逐一覆盖问题中的时间、指标、对象、比较、枚举或其他限定" in prompt
    assert "raw_tool_result" not in prompt
    assert llm.call_args.kwargs == {"temperature": 0.0, "timeout": 12}


def test_numeric_conflict_protocol_receives_full_selection_context():
    llm = Mock(return_value=json.dumps({
        "sufficient": False,
        "answer": None,
        "missing_information": ["相近数值存在口径冲突，当前证据无法消歧"],
        "supported_evidence_ids": [],
    }, ensure_ascii=False))
    context = "原问题要求比较2013年至2015年营业收入，并计算年均复合增长率。"

    EvidenceSufficiencyEvaluator(llm).evaluate(
        "2013年营业收入是多少？",
        _bundle(),
        evidence_selection_context=context,
    )

    prompt = llm.call_args.args[0]
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["evidence_selection_context"] == context
    assert "指标、年份或期间" in prompt
    assert "不得仅按排列位置或 rerank_score 任意选择" in prompt
    assert "若仍无法消歧，必须标记不充分" in prompt


def test_insufficient_has_no_answer_and_no_retry():
    llm = Mock(return_value=json.dumps({
        "sufficient": False,
        "answer": None,
        "missing_information": ["风险分类的完整列表"],
        "supported_evidence_ids": [],
    }, ensure_ascii=False))

    result = EvidenceSufficiencyEvaluator(llm).evaluate("核心经营风险？", _bundle())

    assert result.sufficient is False
    assert result.answer is None
    assert result.missing_information == ("风险分类的完整列表",)
    llm.assert_called_once()


@pytest.mark.parametrize(
    "output",
    [
        "Thought: 我猜答案是市场风险",
        "not json",
        "{}",
        '{"sufficient":true,"answer":"猜测","missing_information":[],"supported_evidence_ids":["unknown"]}',
        TimeoutError("timeout"),
    ],
)
def test_malformed_timeout_or_unknown_evidence_is_generation_failure(output):
    llm = Mock(side_effect=output if isinstance(output, Exception) else None)
    if not isinstance(output, Exception):
        llm.return_value = output

    result = EvidenceSufficiencyEvaluator(llm).evaluate("核心经营风险？", _bundle())

    assert result.generation_failure is True
    assert result.sufficient is False
    assert result.answer is None
    assert "猜答案" not in (result.answer or "")
    llm.assert_called_once()


def test_insufficient_result_with_guessed_answer_is_generation_failure():
    llm = Mock(return_value=json.dumps({
        "sufficient": False,
        "answer": "模型猜测的答案",
        "missing_information": ["资料不足"],
        "supported_evidence_ids": [],
    }, ensure_ascii=False))

    result = EvidenceSufficiencyEvaluator(llm).evaluate("核心经营风险？", _bundle())

    assert result.generation_failure is True
    assert result.answer is None
