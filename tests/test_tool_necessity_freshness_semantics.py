"""Temporal semantics for Tool Necessity freshness decisions."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import Mock

from orchestration import (
    DecomposedRequirement,
    TemporalResolver,
    ToolDecisionMode,
    ToolNecessityPolicy,
    build_default_tool_registry,
)


REFERENCE_DATE = date(2026, 9, 9)


def _requirement(identifier: str, question: str) -> DecomposedRequirement:
    return DecomposedRequirement(identifier, question, expected_output="基金规模排名")


def _semantic_decision(identifier: str, *, freshness: bool = True) -> dict:
    return {
        "task_id": identifier,
        "mode": "tool_required",
        "reason": "requires external financial data",
        "information_gap": "current financial ranking",
        "external_evidence_required": True,
        "freshness_required": freshness,
        "required_capabilities": ["实时或最新金融数据查询"],
        "optional_capabilities": [],
    }


def _policy(*records: dict) -> tuple[ToolNecessityPolicy, Mock]:
    llm = Mock(return_value=json.dumps({"decisions": list(records)}, ensure_ascii=False))
    return ToolNecessityPolicy(build_default_tool_registry(), llm), llm


def test_absolute_historical_year_is_tool_required_but_not_fresh_even_if_llm_claims_otherwise():
    policy, llm = _policy(_semantic_decision("T1"))

    decision = policy.decide(
        (_requirement("T1", "2025年基金规模最大的前十个是谁？"),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert decision.mode is ToolDecisionMode.TOOL_REQUIRED
    assert decision.freshness_required is False
    assert decision.required_capabilities == ("historical_external_data_lookup",)
    assert "绝对历史时间" in llm.call_args.args[0]


def test_compound_absolute_history_normalizes_every_atomic_requirement():
    policy, _ = _policy(_semantic_decision("T1"), _semantic_decision("T2"))

    decisions = policy.decide(
        (
            _requirement("T1", "2021年基金规模最大的前十个是谁？"),
            _requirement("T2", "2025年基金规模最大的前十个是谁？"),
        ),
        reference_date=REFERENCE_DATE,
    )

    assert [decision.freshness_required for decision in decisions] == [False, False]
    assert all(decision.mode is ToolDecisionMode.TOOL_REQUIRED for decision in decisions)


def test_temporally_resolved_last_year_is_historical_not_fresh():
    resolved = TemporalResolver(REFERENCE_DATE).resolve("去年基金规模最大的前十个是谁？")
    assert resolved.resolved_text == "2025年基金规模最大的前十个是谁？"
    policy, _ = _policy(_semantic_decision("T1"))

    decision = policy.decide(
        (_requirement("T1", resolved.resolved_text),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert decision.mode is ToolDecisionMode.TOOL_REQUIRED
    assert decision.freshness_required is False


def test_current_query_is_deterministically_fresh():
    policy = ToolNecessityPolicy(
        build_default_tool_registry(),
        Mock(side_effect=AssertionError("current request is a hard constraint")),
    )

    decision = policy.decide(
        (_requirement("T1", "当前基金规模最大的前十个是谁？"),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert decision.mode is ToolDecisionMode.TOOL_REQUIRED
    assert decision.freshness_required is True


def test_latest_query_is_deterministically_fresh():
    policy = ToolNecessityPolicy(
        build_default_tool_registry(),
        Mock(side_effect=AssertionError("latest request is a hard constraint")),
    )

    decision = policy.decide(
        (_requirement("T1", "最新基金规模排名是什么？"),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert decision.mode is ToolDecisionMode.TOOL_REQUIRED
    assert decision.freshness_required is True


def test_current_year_is_not_normalized_as_finished_history():
    policy = ToolNecessityPolicy(
        build_default_tool_registry(),
        Mock(side_effect=AssertionError("current-year request is a hard constraint")),
    )

    decision = policy.decide(
        (_requirement("T1", "2026年基金规模排名"),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert decision.mode is ToolDecisionMode.TOOL_REQUIRED
    assert decision.freshness_required is True


def test_cutoff_current_is_fresh_but_absolute_cutoff_is_historical():
    current_policy = ToolNecessityPolicy(
        build_default_tool_registry(),
        Mock(side_effect=AssertionError("current cutoff is a hard constraint")),
    )
    current = current_policy.decide(
        (_requirement("T1", "截至今天的基金规模排名"),),
        reference_date=REFERENCE_DATE,
    )[0]
    historical_policy, _ = _policy(_semantic_decision("T2"))
    historical = historical_policy.decide(
        (_requirement("T2", "截至2025-12-31的基金规模排名"),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert current.freshness_required is True
    assert historical.freshness_required is False


def test_current_anchor_overrides_a_finished_historical_period():
    policy = ToolNecessityPolicy(
        build_default_tool_registry(),
        Mock(side_effect=AssertionError("current disclosure state is a hard constraint")),
    )

    decision = policy.decide(
        (_requirement("T1", "截至今天，2025年基金规模数据是否已经完整披露？"),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert decision.mode is ToolDecisionMode.TOOL_REQUIRED
    assert decision.freshness_required is True


def test_latest_is_scoped_to_finished_history_when_an_absolute_period_exists():
    policy, _ = _policy(_semantic_decision("T1"))

    decision = policy.decide(
        (_requirement("T1", "2025年最新基金规模排名"),),
        reference_date=REFERENCE_DATE,
    )[0]

    assert decision.mode is ToolDecisionMode.TOOL_REQUIRED
    assert decision.freshness_required is False
