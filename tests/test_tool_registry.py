"""Central tool capability registry tests."""

from orchestration import SourceType, build_default_tool_registry


def test_registry_contains_all_current_tools_and_sources():
    registry = build_default_tool_registry()
    assert {spec.name for spec in registry.all()} == {
        "query_financial_db", "retrieve_document", "search", "fetch"
    }
    assert registry.get("query_financial_db").source_type is SourceType.SQL
    assert registry.get("retrieve_document").source_type is SourceType.RAG
    assert {spec.name for spec in registry.by_source(SourceType.WEB)} == {"search", "fetch"}


def test_every_tool_declares_capabilities_exclusions_and_contracts():
    for spec in build_default_tool_registry().all():
        assert spec.description
        assert spec.capabilities and spec.exclusions
        assert spec.input_contract and spec.output_contract


def test_registry_is_the_renderable_capability_source():
    rendered = build_default_tool_registry().render_capabilities()
    assert "基金股票持仓" in rendered
    assert "招股说明书" in rendered
    assert "最新新闻" in rendered
    assert "数据库 schema 外的信息" in rendered
