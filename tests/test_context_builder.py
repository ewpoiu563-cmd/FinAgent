import asyncio
import copy
import json
from unittest.mock import AsyncMock, Mock

from orchestration.context_builder import (
    CONTEXT_MAX_CHARS,
    CONTEXT_RECENT_MESSAGE_LIMIT,
    ContextBuildResult,
    ContextBuilder,
)
from orchestration.entrypoint import OrchestrationEntrypoint
from orchestration.session_state import SessionState
from orchestration.trace import TraceRecorder


class _CollectingSink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


def _response(**overrides):
    payload = {
        "depends_on_context": False,
        "used_message_indices": [],
        "standalone_question": "圆周率是多少？",
        "inherited_entity": None,
        "inherited_doc_scope": [],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_state_none_returns_original_without_llm_call():
    llm = Mock(side_effect=AssertionError("LLM must not run"))

    result = ContextBuilder(llm).build("圆周率是多少？", None)

    assert result == ContextBuildResult.unchanged("圆周率是多少？")
    llm.assert_not_called()


def test_missing_session_id_does_not_load_or_build_context():
    planner = Mock()
    planner.plan.side_effect = RuntimeError("force legacy boundary")
    legacy = AsyncMock(return_value="answer")
    store = Mock()
    context_builder = Mock()
    entrypoint = OrchestrationEntrypoint(
        planner,
        Mock(),
        legacy,
        session_store=store,
        context_builder=context_builder,
    )

    assert asyncio.run(entrypoint.run("original")) == "answer"
    legacy.assert_awaited_once_with("original")
    store.get.assert_not_called()
    store.set.assert_not_called()
    context_builder.build.assert_not_called()


def test_independent_question_is_preserved_with_existing_session():
    llm = Mock(return_value=_response())
    state = SessionState(
        "session-1",
        current_entity="健帆生物",
        conversation_summary='{"active_entities":[{"name":"健帆生物","doc_scope":[]}]}',
        recent_messages=[{"role": "user", "content": "健帆生物的风险是什么？"}],
    )

    result = ContextBuilder(llm).build("圆周率是多少？", state)

    assert result == ContextBuildResult.unchanged("圆周率是多少？")


def test_definite_company_reference_repairs_false_negative_from_llm():
    llm = Mock(return_value=_response(
        standalone_question="2015 年公司的营业收入是多少？"
    ))
    state = SessionState(
        "session-1",
        current_entity="珠海健帆生物科技股份有限公司",
        current_doc_scope=["doc_jianfan"],
        recent_messages=[
            {"role": "user", "content": "健帆生物的核心经营风险是什么？"},
            {"role": "assistant", "content": "历史答案"},
        ],
    )

    result = ContextBuilder(llm).build("2015 年公司的营业收入是多少？", state)

    assert result.depends_on_context is True
    assert result.standalone_question == "2015 年珠海健帆生物科技股份有限公司的营业收入是多少？"
    assert result.inherited_doc_scope == ["doc_jianfan"]
    assert result.resolution_method == "deterministic_reference_guard"


def test_context_input_obeys_character_budget_without_mutating_history():
    llm = Mock(return_value=_response())
    state = SessionState(
        "session-1",
        current_entity="健帆生物",
        current_doc_scope=["doc-1", "doc-2"],
        conversation_summary="摘要" * CONTEXT_MAX_CHARS,
        recent_messages=[
            {"role": "user", "content": "历史" * CONTEXT_MAX_CHARS},
            {"role": "assistant", "content": "回答" * CONTEXT_MAX_CHARS},
        ],
    )
    before = copy.deepcopy(state.recent_messages)

    ContextBuilder(llm).build("圆周率是多少？", state)

    context = json.loads(llm.call_args.args[0].split("输入 JSON：", 1)[1])
    assert len(llm.call_args.args[0]) <= CONTEXT_MAX_CHARS
    assert context["current_question"] == "圆周率是多少？"
    assert context["current_entity"] == "健帆生物"
    assert state.recent_messages == before


def test_dependent_question_uses_only_bounded_indexed_history():
    state = SessionState(
        "session-1",
        current_entity="健帆生物",
        current_doc_scope=["jianfan-2023"],
        recent_messages=[
            {"role": "user", "content": f"历史消息 {index}"}
            for index in range(CONTEXT_RECENT_MESSAGE_LIMIT + 2)
        ],
    )
    first_allowed_index = len(state.recent_messages) - CONTEXT_RECENT_MESSAGE_LIMIT
    llm = Mock(
        return_value=_response(
            depends_on_context=True,
            used_message_indices=[first_allowed_index],
            standalone_question="健帆生物2023年的核心经营风险是什么？",
            inherited_entity="健帆生物",
            inherited_doc_scope=["jianfan-2023"],
        )
    )
    before = copy.deepcopy(state.recent_messages)

    result = ContextBuilder(llm).build("那2023年呢？", state)

    assert result.depends_on_context is True
    assert result.standalone_question == "健帆生物2023年的核心经营风险是什么？"
    assert result.used_message_indices == [first_allowed_index]
    assert state.recent_messages == before
    prompt_context = json.loads(llm.call_args.args[0].split("输入 JSON：", 1)[1])
    assert len(prompt_context["recent_messages"]) == CONTEXT_RECENT_MESSAGE_LIMIT
    assert prompt_context["recent_messages"][0]["index"] == first_allowed_index


def test_llm_or_schema_failure_is_fail_open():
    state = SessionState(
        "session-1",
        recent_messages=[{"role": "user", "content": "健帆生物的风险是什么？"}],
    )

    assert ContextBuilder(Mock(side_effect=TimeoutError())).build(
        "那2023年呢？", state
    ) == ContextBuildResult.unchanged("那2023年呢？")
    assert ContextBuilder(Mock(return_value="not-json")).build(
        "那2023年呢？", state
    ) == ContextBuildResult.unchanged("那2023年呢？")


def test_entrypoint_plans_rewrite_but_persists_original_question():
    planner = Mock()
    planner.plan.side_effect = RuntimeError("force legacy boundary")
    legacy = AsyncMock(return_value="答案")
    old_messages = [
        {"role": "user", "content": "健帆生物的核心经营风险是什么？"},
        {"role": "assistant", "content": "历史答案"},
    ]
    state = SessionState("session-1", recent_messages=copy.deepcopy(old_messages))
    store = Mock()
    store.get.return_value = state
    context_builder = Mock()
    context_builder.build.return_value = ContextBuildResult(
        original_question="那2023年呢？",
        standalone_question="健帆生物2023年的核心经营风险是什么？",
        depends_on_context=True,
        used_message_indices=[0],
    )
    entrypoint = OrchestrationEntrypoint(
        planner,
        Mock(),
        legacy,
        session_store=store,
        context_builder=context_builder,
    )

    assert asyncio.run(entrypoint.run("那2023年呢？", session_id="session-1")) == "答案"

    legacy.assert_awaited_once_with("健帆生物2023年的核心经营风险是什么？")
    saved = store.set.call_args.args[0]
    assert saved.recent_messages == old_messages + [
        {"role": "user", "content": "那2023年呢？"},
        {"role": "assistant", "content": "答案"},
    ]


def test_context_trace_contains_only_bounded_metadata():
    planner = Mock()
    planner.plan.side_effect = RuntimeError("force legacy boundary")
    state = SessionState(
        "session-1",
        recent_messages=[{"role": "user", "content": "SECRET HISTORY"}],
    )
    store = Mock()
    store.get.return_value = state
    context_builder = Mock()
    context_builder.build.return_value = ContextBuildResult(
        original_question="那2023年呢？",
        standalone_question="健帆生物2023年的核心经营风险是什么？",
        depends_on_context=True,
        used_message_indices=[0],
    )
    sink = _CollectingSink()
    entrypoint = OrchestrationEntrypoint(
        planner,
        Mock(),
        AsyncMock(return_value="答案"),
        session_store=store,
        context_builder=context_builder,
        trace_recorder=TraceRecorder(sink),
    )

    asyncio.run(entrypoint.run("那2023年呢？", session_id="session-1"))

    event = next(
        item for item in sink.events if item.event_type == "context.build.completed"
    )
    assert event.attributes == {
        "depends_on_context": True,
        "original_question_length": len("那2023年呢？"),
        "standalone_question_length": len("健帆生物2023年的核心经营风险是什么？"),
        "used_message_count": 1,
        "resolution_method": "unchanged",
    }
    assert "SECRET HISTORY" not in json.dumps(event.to_dict(), ensure_ascii=False)


def test_redis_read_failure_runs_original_question_and_skips_context_builder():
    planner = Mock()
    planner.plan.side_effect = RuntimeError("force legacy boundary")
    legacy = AsyncMock(return_value="answer")
    store = Mock()
    store.get.side_effect = ConnectionError("redis unavailable")
    context_builder = Mock()
    entrypoint = OrchestrationEntrypoint(
        planner,
        Mock(),
        legacy,
        session_store=store,
        context_builder=context_builder,
    )

    assert asyncio.run(entrypoint.run("original", session_id="session-1")) == "answer"
    legacy.assert_awaited_once_with("original")
    context_builder.build.assert_not_called()
