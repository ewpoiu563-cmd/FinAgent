"""Unified RAG evidence and final pre-synthesis scope validation tests."""

import copy
import json
import logging
import time
from unittest.mock import Mock

import react_agent
from orchestration import (
    SourceScopeResolver,
    load_default_catalog,
    normalize_rag_tool_result,
    validate_rag_evidence_scope,
)


JIANFAN = "6c93724343cef6b9da70d51daa739e8a1c607a22-5d11f946465d"
JIANFAN_FILE = "6c93724343cef6b9da70d51daa739e8a1c607a22.pdf"
SINOCERA_FILE = "b58b0a129b2c62ca8b295d40c91134df5d5e603d.PDF"


def _raw_result():
    return {
        "success": True,
        "query": "健帆经营风险",
        "evidence": [
            {"rank": 1, "doc_id": JIANFAN, "source_file": JIANFAN_FILE,
             "page": [57], "headings": ["风险因素"], "text": "健帆市场竞争风险",
             "retrieval_unit_id": "jianfan-57", "rerank_score": 0.9},
            {"rank": 2, "doc_id": "doc_sinocera", "source_file": SINOCERA_FILE,
             "page": [12], "headings": ["风险因素"], "text": "国瓷原材料风险",
             "retrieval_unit_id": "sinocera-12", "rerank_score": 0.8},
        ],
        "allowed_doc_ids": [JIANFAN],
    }


def test_rag_evidence_schema_preserves_business_and_physical_identity_and_raw_result():
    original = _raw_result()
    snapshot = copy.deepcopy(original)

    bundle = normalize_rag_tool_result(original, load_default_catalog())
    item = bundle.items[0]

    assert item.source_type.value == "rag"
    assert item.tool_name == "retrieve_document"
    assert item.catalog_doc_id == "doc_jianfan"
    assert item.index_doc_id == JIANFAN
    assert item.company_name == "珠海健帆生物科技股份有限公司"
    assert item.source_file == JIANFAN_FILE
    assert item.page == (57,)
    assert item.headings == ("风险因素",)
    assert item.retrieval_unit_id == "jianfan-57"
    assert item.rerank_score == 0.9
    assert item.text == "健帆市场竞争风险"
    assert bundle.raw_tool_result == snapshot
    assert original == snapshot


def test_source_validation_drops_constructed_cross_company_evidence(caplog):
    caplog.set_level(logging.DEBUG)
    bundle = normalize_rag_tool_result(_raw_result(), load_default_catalog())

    validated = validate_rag_evidence_scope(bundle, [JIANFAN])

    assert [item.index_doc_id for item in validated.items] == [JIANFAN]
    assert validated.scope_violation_count == 1
    assert validated.source_match is False
    assert "SOURCE_SCOPE_VALIDATION:" in caplog.text
    assert "SOURCE_SCOPE_VIOLATION:" in caplog.text
    assert "国瓷原材料风险" in caplog.text


def test_scoped_synthesis_receives_only_validated_unified_evidence(monkeypatch):
    result = _raw_result()
    llm = Mock(return_value=json.dumps({
        "sufficient": False, "answer": None, "missing_information": ["更多风险证据"]
    }))
    monkeypatch.setattr(react_agent, "call_llm", llm)
    state = react_agent.ReActState(question="健帆有哪些经营风险？", start_time=time.time())

    react_agent._synthesize_document_answer(state, result)

    prompt = llm.call_args.args[0]
    payload = json.loads(prompt.split("\n", 1)[1])
    assert len(payload["evidence"]) == 1
    evidence = payload["evidence"][0]
    assert evidence["catalog_doc_id"] == "doc_jianfan"
    assert evidence["index_doc_id"] == JIANFAN
    assert evidence["source_type"] == "rag"
    assert evidence["tool_name"] == "retrieve_document"
    assert "doc_sinocera" not in prompt
    assert "国瓷原材料风险" not in prompt


def test_scope_resolution_and_validation_form_business_to_synthesis_boundary():
    allowed = SourceScopeResolver(load_default_catalog()).resolve(["doc_jianfan"])
    validated = validate_rag_evidence_scope(
        normalize_rag_tool_result(_raw_result(), load_default_catalog()),
        allowed,
    )
    assert allowed == (JIANFAN,)
    assert {item.catalog_doc_id for item in validated.items} == {"doc_jianfan"}
