import asyncio
import json
from unittest.mock import AsyncMock, Mock

from orchestration.entrypoint import OrchestrationEntrypoint
from orchestration.models import PlanMode, SourcePlan, SourceType
from orchestration.session_state import (
    MAX_RECENT_MESSAGES,
    RedisSessionStore,
    SessionState,
)


def test_session_state_json_round_trip_does_not_trim_before_compression():
    state = SessionState(
        session_id="session-1",
        current_entity="贵州茅台",
        current_doc_scope=["doc-1", "doc-2"],
        recent_messages=[
            {"role": "user", "content": str(index)}
            for index in range(MAX_RECENT_MESSAGES + 2)
        ],
        conversation_summary="reserved",
        last_run_id="run-1",
    )

    restored = SessionState.from_json(state.to_json())

    assert restored == state
    assert len(restored.recent_messages) == MAX_RECENT_MESSAGES + 2
    assert restored.recent_messages[0]["content"] == "0"


def test_redis_session_store_set_get_and_ttl():
    client = Mock()
    state = SessionState("session-1", current_entity="贵州茅台")
    store = RedisSessionStore("redis://unused", 86400, client=client)

    store.set(state)

    client.set.assert_called_once()
    key, payload = client.set.call_args.args
    assert key == "finagent:session:session-1"
    assert json.loads(payload)["current_entity"] == "贵州茅台"
    assert client.set.call_args.kwargs == {"ex": 86400}

    client.get.return_value = payload.encode("utf-8")
    assert store.get("session-1") == state


def test_redis_session_store_delete():
    client = Mock()
    store = RedisSessionStore("redis://unused", 86400, client=client)

    store.delete("session-1")

    client.delete.assert_called_once_with("finagent:session:session-1")


def test_redis_session_store_get_missing_returns_none():
    client = Mock()
    client.get.return_value = None
    store = RedisSessionStore("redis://unused", 86400, client=client)

    assert store.get("missing") is None


def test_session_update_preserves_unresolved_scope_and_appends_messages():
    store = Mock()
    store.get.return_value = SessionState(
        "session-1",
        current_entity="已有实体",
        current_doc_scope=["existing-doc"],
    )
    entrypoint = object.__new__(OrchestrationEntrypoint)
    entrypoint.session_store = store
    plan = SourcePlan(
        mode=PlanMode.SINGLE_SOURCE,
        primary_source=SourceType.DIRECT,
        required_sources=(SourceType.DIRECT,),
    )

    entrypoint._persist_session(
        session_id="session-1",
        run_id="run-1",
        question="问题",
        answer="答案",
        plan=plan,
    )

    saved = store.set.call_args.args[0]
    assert saved.current_entity == "已有实体"
    assert saved.current_doc_scope == ["existing-doc"]
    assert saved.recent_messages == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答案"},
    ]
    assert saved.last_run_id == "run-1"


def test_redis_failure_and_missing_session_id_are_fail_open():
    failing_store = Mock()
    failing_store.get.side_effect = ConnectionError("redis unavailable")
    entrypoint = object.__new__(OrchestrationEntrypoint)
    entrypoint.session_store = failing_store
    entrypoint._persist_session(
        session_id="session-1",
        run_id="run-1",
        question="问题",
        answer="答案",
        plan=None,
    )

    planner = Mock()
    planner.plan.side_effect = RuntimeError("use legacy")
    legacy = AsyncMock(return_value="stateless answer")
    store = Mock()
    stateless_entrypoint = OrchestrationEntrypoint(
        planner,
        Mock(),
        legacy,
        session_store=store,
    )

    assert asyncio.run(stateless_entrypoint.run("question")) == "stateless answer"
    store.get.assert_not_called()
    store.set.assert_not_called()
