"""Deterministic exact-alias entity resolution tests."""

import pytest

from orchestration import EntityResolver, load_default_catalog


@pytest.fixture(scope="module")
def resolver():
    return EntityResolver(load_default_catalog())


@pytest.mark.parametrize(
    "question,expected_doc_id,expected_alias",
    [
        ("健帆", "doc_jianfan", "健帆"),
        ("健帆生物", "doc_jianfan", "健帆生物"),
        ("健帆公司", "doc_jianfan", "健帆公司"),
        ("信立泰", "doc_xinlitai", "信立泰"),
        ("亚厦", "doc_yaxia", "亚厦"),
        ("国瓷材料", "doc_sinocera", "国瓷材料"),
    ],
)
def test_required_aliases_resolve_to_business_catalog_ids(
    resolver, question, expected_doc_id, expected_alias
):
    result = resolver.resolve(question)
    assert result.catalog_doc_id == expected_doc_id
    assert result.matched_alias == expected_alias
    assert result.document_type == "prospectus"
    assert "index_doc" not in result.to_dict()


def test_longest_normalized_alias_wins(resolver):
    result = resolver.resolve("请查本地《健帆 生物》招股说明书")
    assert result.catalog_doc_id == "doc_jianfan"
    assert result.matched_alias == "健帆生物"
    assert result.company_name == "珠海健帆生物科技股份有限公司"


def test_document_type_can_resolve_without_company(resolver):
    result = resolver.resolve("这份年度报告披露了哪些风险？")
    assert result.catalog_doc_id is None
    assert result.document_type == "annual_report"


def test_unmatched_query_returns_empty_business_identity(resolver):
    result = resolver.resolve("解释一下宏观经济周期")
    assert result.matched is False
    assert result.company_name is None and result.matched_alias is None
