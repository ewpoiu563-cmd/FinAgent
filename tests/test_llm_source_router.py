"""Strict lightweight router contract tests; no network calls."""

from unittest.mock import Mock

import pytest

from orchestration import EntityResolver, LLMSourceRouter, SourceType, build_default_tool_registry, load_default_catalog


def test_router_uses_one_small_source_only_call():
    llm = Mock(return_value=(
        '{"primary_source":"rag","required_sources":["rag"],'
        '"web_fallback_allowed":false,"reason":"local disclosure"}'
    ))
    entity = EntityResolver(load_default_catalog()).resolve("分析健帆生物的竞争力")

    decision = LLMSourceRouter(llm, timeout=12).route(
        "分析健帆生物的竞争力", entity, build_default_tool_registry()
    )

    assert decision.primary_source is SourceType.RAG
    assert decision.required_sources == (SourceType.RAG,)
    assert decision.web_fallback_allowed is False
    llm.assert_called_once()
    prompt = llm.call_args.args[0]
    assert "只判断完成问题需要哪些来源" in prompt
    assert "doc_jianfan" in prompt
    assert "6c93724343cef6b9da70d51daa739e8a1c607a22-5d11f946465d" not in prompt
    assert llm.call_args.kwargs == {"temperature": 0.0, "timeout": 12}


@pytest.mark.parametrize(
    "raw,message",
    [
        ("not json", "valid JSON"),
        ('{"primary_source":"rag","required_sources":[]}', "non-empty"),
        ('{"primary_source":"sql","required_sources":["rag"]}', "included"),
        ('{"primary_source":"other","required_sources":["other"]}', "sql, rag, or web"),
    ],
)
def test_router_rejects_invalid_decisions(raw, message):
    with pytest.raises(ValueError, match=message):
        LLMSourceRouter.parse_decision(raw)


def test_router_accepts_json_code_fence():
    decision = LLMSourceRouter.parse_decision(
        '```json\n{"primary_source":"web","required_sources":["web"],'
        '"web_fallback_allowed":true,"reason":"recent information"}\n```'
    )
    assert decision.required_sources == (SourceType.WEB,)
