"""Rule-first SQL/RAG/Web and explicit multi-source planning tests."""

import pytest
from unittest.mock import Mock

from orchestration import (
    EntityResolver,
    PlanMode,
    SourcePlanner,
    SourceType,
    RouterDecision,
    build_default_tool_registry,
    load_default_catalog,
)


@pytest.fixture(scope="module")
def planner():
    return SourcePlanner(
        EntityResolver(load_default_catalog()),
        build_default_tool_registry(),
    )


@pytest.mark.parametrize(
    "question,source",
    [
        ("本地健帆公司招股说明书中的经营风险", SourceType.RAG),
        ("2021年持有贵州茅台最多的基金", SourceType.SQL),
        ("健帆生物最近有哪些新闻", SourceType.WEB),
    ],
)
def test_clear_single_source_rules(planner, question, source):
    plan = planner.plan(question)
    assert plan.mode is PlanMode.SINGLE_SOURCE
    assert plan.primary_source is source
    assert plan.required_sources == (source,)
    assert plan.routing_method == "rules"


def test_document_plan_uses_business_catalog_id_only(planner):
    plan = planner.plan("本地健帆公司招股说明书中的经营风险")
    assert plan.company_name == "珠海健帆生物科技股份有限公司"
    assert plan.catalog_doc_ids == ("doc_jianfan",)
    assert plan.document_type == "prospectus"
    assert "6c93724343cef6b9da70d51daa739e8a1c607a22-5d11f946465d" not in str(plan.to_dict())


def test_explicit_rag_and_recent_news_is_multi_source(planner):
    plan = planner.plan("对比健帆招股说明书风险和最近新闻")
    assert plan.mode is PlanMode.MULTI_SOURCE
    assert plan.primary_source is SourceType.RAG
    assert set(plan.required_sources) == {SourceType.RAG, SourceType.WEB}
    assert plan.catalog_doc_ids == ("doc_jianfan",)


def test_recent_fund_news_is_web_not_implicitly_sql(planner):
    plan = planner.plan("最近有哪些基金新闻？")
    assert plan.required_sources == (SourceType.WEB,)


def test_ambiguous_query_remains_auto_for_future_lightweight_router(planner):
    plan = planner.plan("分析健帆生物的竞争力")
    assert plan.mode is PlanMode.AUTO
    assert plan.primary_source is None and plan.required_sources == ()
    assert plan.routing_method == "auto_pending"
    assert plan.catalog_doc_ids == ("doc_jianfan",)


def test_ambiguous_query_uses_injected_lightweight_router():
    router = Mock()
    router.route.return_value = RouterDecision(
        primary_source=SourceType.RAG,
        required_sources=(SourceType.RAG,),
        web_fallback_allowed=False,
        reason="company analysis needs local disclosure",
    )
    planner = SourcePlanner(
        EntityResolver(load_default_catalog()),
        build_default_tool_registry(),
        ambiguous_router=router,
    )

    plan = planner.plan("分析健帆生物的竞争力")

    assert plan.primary_source is SourceType.RAG
    assert plan.routing_method == "llm"
    assert plan.web_fallback_allowed is False
    router.route.assert_called_once()


def test_invalid_llm_source_is_rejected_and_routed_again_with_constraints():
    router = Mock()
    router.route.side_effect = [
        RouterDecision(
            primary_source=SourceType.SQL,
            required_sources=(SourceType.SQL,),
            reason="incorrect database choice",
        ),
        RouterDecision(
            primary_source=SourceType.RAG,
            required_sources=(SourceType.RAG,),
            web_fallback_allowed=False,
            reason="catalog document is the valid source",
        ),
    ]
    planner = SourcePlanner(
        EntityResolver(load_default_catalog()),
        build_default_tool_registry(),
        ambiguous_router=router,
    )

    plan = planner.plan("分析健帆生物的竞争力")

    assert plan.primary_source is SourceType.RAG
    assert plan.catalog_doc_ids == ("doc_jianfan",)
    assert router.route.call_count == 2
    recovery_kwargs = router.route.call_args_list[1].kwargs
    assert SourceType.SQL in recovery_kwargs["rejected_sources"]
    assert SourceType.SQL not in recovery_kwargs["allowed_sources"]


def test_clear_rule_never_calls_lightweight_router():
    router = Mock(side_effect=AssertionError("clear rules must not call the router"))
    planner = SourcePlanner(
        EntityResolver(load_default_catalog()),
        build_default_tool_registry(),
        ambiguous_router=router,
    )

    assert planner.plan("健帆生物最近有哪些新闻").primary_source is SourceType.WEB
    router.route.assert_not_called()
