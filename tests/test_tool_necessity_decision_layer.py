"""Architecture tests for requirement-level Tool Necessity decisions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from orchestration import (
    DecomposedRequirement,
    DirectNoToolExecutor,
    EntityResolver,
    EvidenceBundle,
    EvidenceItem,
    LightweightTaskDecomposer,
    OrchestrationEntrypoint,
    RouterDecision,
    SourcePlanner,
    SourceType,
    TaskPlanExecutor,
    TaskPlanner,
    ToolDecisionMode,
    ToolNecessityPolicy,
    build_default_tool_registry,
    load_default_catalog,
)


def _registry():
    return build_default_tool_registry()


def _source_planner(router=None):
    return SourcePlanner(
        EntityResolver(load_default_catalog()),
        _registry(),
        ambiguous_router=router,
    )


def _decision(task_id: str, mode: str, **changes) -> dict:
    payload = {
        "task_id": task_id,
        "mode": mode,
        "reason": "semantic necessity decision",
        "information_gap": None if mode == "no_tool" else "external information gap",
        "external_evidence_required": mode == "tool_required",
        "freshness_required": False,
        "required_capabilities": [] if mode == "no_tool" else ["public_fact_lookup"],
        "optional_capabilities": [],
    }
    payload.update(changes)
    return payload


def _necessity_llm(*decisions):
    return Mock(return_value=json.dumps({"decisions": list(decisions)}, ensure_ascii=False))


def _atomic_decomposition(prompt, **_kwargs):
    question = prompt.rsplit("用户问题：", 1)[-1].strip()
    return json.dumps({
        "requirements": [{
            "id": "T1",
            "question": question,
            "required": True,
            "depends_on": [],
            "expected_output": "直接回答该原子需求",
        }]
    }, ensure_ascii=False)


def _task_planner(source_planner, necessity_llm, *, decomposer=None):
    return TaskPlanner(
        source_planner,
        task_decomposer=decomposer or LightweightTaskDecomposer(Mock(side_effect=_atomic_decomposition)),
        necessity_policy=ToolNecessityPolicy(_registry(), necessity_llm),
    )


def test_stable_knowledge_uses_semantic_no_tool_policy_not_source_allowlist():
    llm = _necessity_llm(_decision("T1", "no_tool"))
    planner = _task_planner(_source_planner(), llm)

    plan = planner.plan("解释机会成本为何会影响个人选择。")

    assert [task.source for task in plan.execution_tasks] == ["direct"]
    assert plan.execution_tasks[0].tool_decision.mode is ToolDecisionMode.NO_TOOL
    assert plan.source_plan.required_sources == (SourceType.DIRECT,)
    llm.assert_called_once()


def test_hard_external_constraints_skip_semantic_necessity_llm():
    cases = [
        ("今天市场有哪些重要变化？", SourceType.WEB),
        ("本地健帆招股说明书中披露了什么？", SourceType.RAG),
        ("查询本地数据库中的基金规模记录。", SourceType.SQL),
    ]
    for question, source in cases:
        necessity_llm = Mock(side_effect=AssertionError("hard constraints need no semantic call"))
        planner = _task_planner(_source_planner(), necessity_llm)
        plan = planner.plan(question)
        task = plan.execution_tasks[0]
        assert task.source == source.value
        assert task.tool_decision.needs_tool is True
        assert task.tool_decision.decision_method == "deterministic_hard_constraint"
        necessity_llm.assert_not_called()


def test_explicit_verification_requires_evidence_then_source_selection():
    router = Mock()
    router.route.return_value = RouterDecision(
        primary_source=SourceType.WEB,
        required_sources=(SourceType.WEB,),
        reason="public evidence can close the verification gap",
    )
    planner = _task_planner(_source_planner(router), Mock())

    task_plan = planner.plan("请核实这项公开事实并提供证据。")

    task = task_plan.execution_tasks[0]
    assert task.tool_decision.external_evidence_required is True
    assert task.source == "web"
    router.route.assert_called_once()


def test_ambiguous_semantics_can_choose_no_tool_without_defaulting_to_web():
    router = Mock(side_effect=AssertionError("no-tool requirement must not select a source"))
    necessity_llm = _necessity_llm(_decision("T1", "no_tool"))
    planner = _task_planner(_source_planner(router), necessity_llm)

    plan = planner.plan("分析这个论证中的循环论证问题。")

    assert plan.source_plan.primary_source is SourceType.DIRECT
    assert plan.execution_tasks[0].tool_decision.needs_tool is False
    router.route.assert_not_called()


def test_domain_nouns_alone_do_not_become_hard_tool_rules():
    router = Mock(side_effect=AssertionError("semantic no-tool task must not select a source"))
    necessity_llm = _necessity_llm(_decision("T1", "no_tool"))
    planner = _task_planner(_source_planner(router), necessity_llm)

    plan = planner.plan("解释年报与新闻传播这两个概念的基本含义。")

    assert [task.source for task in plan.execution_tasks] == ["direct"]
    assert plan.execution_tasks[0].tool_decision.decision_method == "llm_semantic_policy"
    necessity_llm.assert_called_once()
    router.route.assert_not_called()


def test_catalog_company_risk_repairs_missing_capability_and_routes_to_rag():
    router = Mock(side_effect=AssertionError("catalog-bound risk must not need LLM source routing"))
    necessity_llm = Mock(side_effect=AssertionError("catalog-bound risk is a deterministic constraint"))
    planner = _task_planner(_source_planner(router), necessity_llm)

    plan = planner.plan("健帆生物的核心经营风险是什么？")

    task = plan.execution_tasks[0]
    assert task.source == "rag"
    assert task.tool_decision.required_capabilities == ("catalog_document_evidence",)
    assert task.tool_decision.decision_method == "deterministic_catalog_constraint"
    assert plan.source_plan.catalog_doc_ids == ("doc_jianfan",)
    assert plan.source_plan.web_fallback_allowed is False
    necessity_llm.assert_not_called()
    router.route.assert_not_called()


def test_catalog_company_report_fact_skips_duplicate_semantic_and_source_routing():
    router = Mock(side_effect=AssertionError("catalog-bound report fact needs no LLM source routing"))
    necessity_llm = Mock(side_effect=AssertionError("catalog-bound report fact needs no necessity call"))
    planner = _task_planner(_source_planner(router), necessity_llm)

    plan = planner.plan("健帆生物2015年营业收入是多少？")

    task = plan.execution_tasks[0]
    assert task.source == "rag"
    assert task.tool_decision.decision_method == "deterministic_catalog_constraint"
    assert plan.source_plan.catalog_doc_ids == ("doc_jianfan",)
    assert plan.source_plan.web_fallback_allowed is False
    necessity_llm.assert_not_called()
    router.route.assert_not_called()


def test_tool_required_missing_capability_is_repaired_without_legacy_failure():
    raw = json.dumps({
        "decisions": [_decision("T1", "tool_required", required_capabilities=[])]
    }, ensure_ascii=False)

    decision = ToolNecessityPolicy.parse(raw, {"T1"})[0]

    assert decision.required_capabilities == ("external_information_lookup",)
    assert decision.decision_method == "llm_semantic_policy_repaired"


def test_deterministic_sql_constraint_prevents_false_no_tool_decision():
    necessity_llm = _necessity_llm(_decision("T1", "no_tool"))
    planner = _task_planner(_source_planner(), necessity_llm)

    plan = planner.plan("2021年持有贵州茅台最多的基金是什么？")

    task = plan.execution_tasks[0]
    assert task.source == "sql"
    assert task.tool_decision.decision_method == "deterministic_route_guard_correction"
    necessity_llm.assert_called_once()


def test_necessity_contract_failure_recovers_through_bounded_source_planner():
    source_planner = _source_planner(
        Mock(side_effect=AssertionError("catalog rule should resolve the source"))
    )
    task_planner = _task_planner(source_planner, Mock(return_value="not-json"))
    result = Mock()
    result.public_answer.return_value = "RAG answer"
    task_executor = Mock()
    task_executor.can_execute.return_value = True
    task_executor.execute = AsyncMock(return_value=result)
    legacy = AsyncMock(side_effect=AssertionError("contract recovery must not use legacy"))
    entrypoint = OrchestrationEntrypoint(
        source_planner,
        Mock(),
        legacy,
        task_planner=task_planner,
        task_executor=task_executor,
    )

    answer = asyncio.run(entrypoint.run("健帆生物的核心经营风险是什么？"))

    assert answer == "RAG answer"
    recovered_plan = task_executor.execute.await_args.args[0]
    assert recovered_plan.source_plan.primary_source is SourceType.RAG
    legacy.assert_not_awaited()


def test_named_document_facts_use_catalog_bound_rag_or_web_not_direct():
    decomposition = Mock(return_value=json.dumps({
        "requirements": [
            {"id": "T1", "question": "健帆生物在其招股说明书中披露的2015年营业收入是多少？",
             "required": True, "depends_on": [], "expected_output": "2015年营业收入"},
            {"id": "T2", "question": "健帆生物2025年年度报告披露的营业收入是多少？",
             "required": True, "depends_on": [], "expected_output": "2025年营业收入"},
        ]
    }, ensure_ascii=False))
    necessity_llm = Mock(side_effect=AssertionError("named document facts are hard constraints"))
    planner = _task_planner(
        _source_planner(),
        necessity_llm,
        decomposer=LightweightTaskDecomposer(decomposition),
    )

    plan = planner.plan("对比健帆生物招股说明书和2025年年度报告披露的营业收入。")

    assert [task.source for task in plan.execution_tasks] == ["rag", "web"]
    assert all(task.tool_decision.needs_tool for task in plan.execution_tasks)
    necessity_llm.assert_not_called()


def test_decomposer_prompt_requires_exact_execution_constraints():
    prompt = LightweightTaskDecomposer._prompt("查询2021年末资产净值，仅限本地数据库。")

    assert "年份、日期、年末/截至等时间点、指标口径" in prompt
    assert "不得改写成其他时间聚合、其他指标或其他来源" in prompt
    assert "retrieval_query" in prompt


def test_decomposer_accepts_bounded_upstream_rag_query():
    raw = json.dumps({
        "requirements": [{
            "id": "T1",
            "question": "健帆生物2015年营业收入是多少？",
            "required": True,
            "depends_on": [],
            "expected_output": "2015年营业收入",
            "retrieval_query": "2015年 营业收入 合并利润表 经营成果分析",
        }]
    }, ensure_ascii=False)

    requirement = LightweightTaskDecomposer.parse(raw)[0]

    assert requirement.retrieval_query == "2015年 营业收入 合并利润表 经营成果分析"


def test_required_rag_does_not_implicitly_enable_web_fallback():
    planner = _task_planner(
        _source_planner(),
        Mock(side_effect=AssertionError("explicit source is a hard constraint")),
    )

    plan = planner.plan("请根据本地健帆招股说明书回答。")

    assert plan.source_plan.required_sources == (SourceType.RAG,)
    assert plan.source_plan.web_fallback_allowed is False


def test_compound_query_gets_independent_no_tool_and_web_decisions():
    decomposition = Mock(return_value=json.dumps({
        "requirements": [
            {"id": "T1", "question": "解释机会成本。", "required": True,
             "depends_on": [], "expected_output": "概念解释"},
            {"id": "T2", "question": "今天市场有哪些变化？", "required": True,
             "depends_on": [], "expected_output": "当前市场变化"},
        ]
    }, ensure_ascii=False))
    necessity_llm = _necessity_llm(_decision("T1", "no_tool"))
    planner = _task_planner(
        _source_planner(),
        necessity_llm,
        decomposer=LightweightTaskDecomposer(decomposition),
    )

    plan = planner.plan("解释机会成本，同时说明今天市场有哪些变化。")

    assert [task.source for task in plan.execution_tasks] == ["direct", "web"]
    assert plan.source_plan.required_sources == (SourceType.DIRECT, SourceType.WEB)
    assert len(plan.synthesis_tasks) == 1
    necessity_llm.assert_called_once()
    prompt = necessity_llm.call_args.args[0]
    assert '"id": "T1"' in prompt and '"id": "T2"' not in prompt


def test_compound_external_tasks_keep_distinct_sql_rag_and_web_assignments():
    decomposition = Mock(return_value=json.dumps({
        "requirements": [
            {"id": "T1", "question": "查询本地数据库中的基金规模。", "required": True,
             "depends_on": [], "expected_output": "结构化记录"},
            {"id": "T2", "question": "根据本地健帆招股说明书概括风险。", "required": True,
             "depends_on": [], "expected_output": "文档证据"},
            {"id": "T3", "question": "说明今天市场的重要变化。", "required": True,
             "depends_on": [], "expected_output": "当前信息"},
        ]
    }, ensure_ascii=False))
    necessity_llm = Mock(side_effect=AssertionError("all requirements have hard constraints"))
    planner = _task_planner(
        _source_planner(),
        necessity_llm,
        decomposer=LightweightTaskDecomposer(decomposition),
    )

    plan = planner.plan("查询数据库；阅读本地文档；再说明今天的市场变化。")

    assert [task.source for task in plan.execution_tasks] == ["sql", "rag", "web"]
    assert TaskPlanExecutor(
        sql_executor=Mock(),
        rag_executor=Mock(),
        web_executor=Mock(),
    ).can_execute(plan)
    necessity_llm.assert_not_called()


def _web_result():
    evidence = EvidenceBundle(
        source_type=SourceType.WEB,
        tool_name="search",
        items=(EvidenceItem(
            source_type=SourceType.WEB,
            tool_name="search",
            retrieval_unit_id="web-1",
            text="today changed",
            title="news",
            url="https://example.com/news",
            snippet="today changed",
        ),),
        raw_tool_result={},
        execution_success=True,
        source_match=True,
    )
    return SimpleNamespace(
        status="success",
        answer="今天市场出现变化。",
        evidence=evidence,
        web_evidence=evidence,
        missing_information=(),
        error_type=None,
    )


def test_mixed_no_tool_and_web_executes_only_the_required_external_task():
    decomposition = Mock(return_value=json.dumps({
        "requirements": [
            {"id": "T1", "question": "解释机会成本。", "required": True,
             "depends_on": [], "expected_output": "概念解释"},
            {"id": "T2", "question": "今天市场有哪些变化？", "required": True,
             "depends_on": [], "expected_output": "当前市场变化"},
        ]
    }, ensure_ascii=False))
    source_planner = _source_planner()
    task_planner = _task_planner(
        source_planner,
        _necessity_llm(_decision("T1", "no_tool")),
        decomposer=LightweightTaskDecomposer(decomposition),
    )
    direct_llm = Mock(return_value="机会成本是放弃的最佳替代方案价值。")
    web = Mock()
    web.execute_for_source = AsyncMock(return_value=_web_result())
    rag = Mock(side_effect=AssertionError("RAG must not run"))
    sql = Mock(side_effect=AssertionError("SQL must not run"))
    legacy = AsyncMock(side_effect=AssertionError("planned mixed task must not use legacy"))
    entrypoint = OrchestrationEntrypoint(
        source_planner,
        rag,
        legacy,
        direct_sql_executor=sql,
        direct_web_executor=web,
        direct_executor=DirectNoToolExecutor(direct_llm),
        task_planner=task_planner,
    )

    answer = asyncio.run(entrypoint.run("解释机会成本，同时说明今天市场有哪些变化。"))

    assert "机会成本" in answer and "今天市场出现变化" in answer
    direct_llm.assert_called_once()
    web.execute_for_source.assert_awaited_once()
    rag.assert_not_called()
    sql.assert_not_called()
    legacy.assert_not_awaited()


def test_no_tool_entrypoint_calls_no_external_executor_or_legacy():
    source_planner = _source_planner()
    planner = _task_planner(
        source_planner,
        _necessity_llm(_decision("T1", "no_tool")),
    )
    direct_llm = Mock(return_value="这是由稳定知识和推理得到的答案。")
    rag = Mock(side_effect=AssertionError("RAG must not run"))
    sql = Mock(side_effect=AssertionError("SQL must not run"))
    web = Mock()
    web.execute_for_source = AsyncMock(side_effect=AssertionError("Web must not run"))
    legacy = AsyncMock(side_effect=AssertionError("legacy must not run"))
    entrypoint = OrchestrationEntrypoint(
        source_planner,
        rag,
        legacy,
        direct_sql_executor=sql,
        direct_web_executor=web,
        direct_executor=DirectNoToolExecutor(direct_llm),
        task_planner=planner,
    )

    answer = asyncio.run(entrypoint.run("说明供给与需求曲线的基本关系。"))

    assert answer == "这是由稳定知识和推理得到的答案。"
    direct_llm.assert_called_once()
    rag.assert_not_called()
    sql.assert_not_called()
    web.execute_for_source.assert_not_awaited()
    legacy.assert_not_awaited()


def test_legacy_react_is_reserved_for_explicit_exploratory_decision():
    source_planner = _source_planner()
    planner = _task_planner(
        source_planner,
        _necessity_llm(_decision(
            "T1",
            "exploratory",
            required_capabilities=["dynamic_open_research"],
        )),
    )
    legacy = AsyncMock(return_value="legacy exploratory answer")
    entrypoint = OrchestrationEntrypoint(source_planner, Mock(), legacy, task_planner=planner)

    answer = asyncio.run(entrypoint.run("开展一个无法预先确定路径的开放研究。"))

    assert answer == "legacy exploratory answer"
    legacy.assert_awaited_once()
