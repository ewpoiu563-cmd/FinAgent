import copy
import json
from unittest.mock import Mock

import pytest

from orchestration.context_compressor import (
    COMPRESSION_MAX_CHARS,
    CONVERSATION_SUMMARY_MAX_CHARS,
    SUMMARY_KEEP_RECENT_MESSAGES,
    SUMMARY_TRIGGER_MESSAGES,
    ContextCompressor,
)
from orchestration.entrypoint import OrchestrationEntrypoint
from orchestration.session_state import SessionState
from orchestration.trace import TraceRecorder


def _messages(count):
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index}",
        }
        for index in range(count)
    ]


def _summary(**overrides):
    payload = {
        "active_entities": [
            {"name": "健帆生物", "doc_scope": ["jianfan-2023"]}
        ],
        "user_goals": ["分析核心经营风险"],
        "constraints": [],
        "discussed_topics": ["经营风险"],
        "open_questions": [],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class _CollectingSink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


def _entrypoint(compressor):
    instance = object.__new__(OrchestrationEntrypoint)
    instance.context_compressor = compressor
    return instance


def test_below_threshold_does_not_call_compressor_or_change_messages():
    compressor = Mock()
    state = SessionState("session-1", recent_messages=_messages(SUMMARY_TRIGGER_MESSAGES))
    before = copy.deepcopy(state.recent_messages)

    _entrypoint(compressor)._maybe_compress_session(state)

    compressor.compress.assert_not_called()
    assert state.recent_messages == before
    assert state.conversation_summary == ""


def test_success_summarizes_old_prefix_before_trimming_and_keeps_complete_turns():
    total = SUMMARY_TRIGGER_MESSAGES + 2
    state = SessionState("session-1", recent_messages=_messages(total))
    original_messages = copy.deepcopy(state.recent_messages)
    compressor = Mock()

    def compress(existing_summary, old_messages):
        assert existing_summary == ""
        assert state.recent_messages == original_messages
        assert old_messages == original_messages[:-SUMMARY_KEEP_RECENT_MESSAGES]
        return ContextCompressor.parse_summary(_summary())

    compressor.compress.side_effect = compress

    _entrypoint(compressor)._maybe_compress_session(state)

    assert state.recent_messages == original_messages[-SUMMARY_KEEP_RECENT_MESSAGES:]
    assert state.recent_messages[0]["role"] == "user"
    assert original_messages[-SUMMARY_KEEP_RECENT_MESSAGES - 1]["role"] == "assistant"
    assert json.loads(state.conversation_summary)["user_goals"] == [
        "分析核心经营风险"
    ]


def test_compressor_failure_preserves_all_old_messages_and_existing_summary():
    original_summary = _summary(user_goals=["已有目标"])
    state = SessionState(
        "session-1",
        recent_messages=_messages(SUMMARY_TRIGGER_MESSAGES + 2),
        conversation_summary=original_summary,
    )
    before = copy.deepcopy(state.recent_messages)
    compressor = Mock()
    compressor.compress.side_effect = TimeoutError("LLM timeout")

    _entrypoint(compressor)._maybe_compress_session(state)

    assert state.recent_messages == before
    assert state.conversation_summary == original_summary


def test_compressor_receives_existing_summary_and_only_new_old_prefix():
    existing = _summary(user_goals=["比较年度风险"])
    old_messages = _messages(4)
    llm = Mock(return_value=_summary(user_goals=["比较年度风险", "补充2023年"]))

    encoded = ContextCompressor(llm).compress(existing, old_messages)

    prompt_input = json.loads(llm.call_args.args[0].split("输入 JSON：", 1)[1])
    assert prompt_input == {
        "existing_summary": existing,
        "newly_evicted_old_messages": old_messages,
    }
    assert json.loads(encoded)["user_goals"] == ["比较年度风险", "补充2023年"]


def test_overlong_summary_is_rejected_before_any_session_trim():
    raw = _summary(user_goals=["x" * CONVERSATION_SUMMARY_MAX_CHARS])
    with pytest.raises(ValueError, match="exceeds"):
        ContextCompressor.parse_summary(raw)

    state = SessionState(
        "session-1",
        recent_messages=_messages(SUMMARY_TRIGGER_MESSAGES + 2),
    )
    before = copy.deepcopy(state.recent_messages)
    compressor = Mock()
    compressor.compress.side_effect = ValueError("conversation summary exceeds")
    _entrypoint(compressor)._maybe_compress_session(state)
    assert state.recent_messages == before


def test_compression_trace_has_counts_but_no_message_text():
    state = SessionState(
        "session-1",
        recent_messages=_messages(SUMMARY_TRIGGER_MESSAGES + 2),
    )
    compressor = Mock(return_value=None)
    compressor.compress.return_value = ContextCompressor.parse_summary(_summary())
    sink = _CollectingSink()

    with TraceRecorder(sink).run(session_id="session-1"):
        _entrypoint(compressor)._maybe_compress_session(state)

    event = next(
        item for item in sink.events if item.event_type == "context.compression.completed"
    )
    assert event.attributes == {
        "compression_triggered": True,
        "messages_before": SUMMARY_TRIGGER_MESSAGES + 2,
        "messages_after": SUMMARY_KEEP_RECENT_MESSAGES,
        "compressed_message_count": SUMMARY_TRIGGER_MESSAGES + 2 - SUMMARY_KEEP_RECENT_MESSAGES,
        "remaining_message_count": SUMMARY_KEEP_RECENT_MESSAGES,
        "compression_input_chars": ContextCompressor.prompt_length(
            "", _messages(SUMMARY_TRIGGER_MESSAGES + 2 - SUMMARY_KEEP_RECENT_MESSAGES)
        ),
        "summary_length": len(state.conversation_summary),
    }
    assert "message-" not in json.dumps(event.to_dict(), ensure_ascii=False)


def test_persist_appends_original_turn_then_compresses_before_redis_set():
    existing = _messages(SUMMARY_TRIGGER_MESSAGES)
    state = SessionState("session-1", recent_messages=copy.deepcopy(existing))
    store = Mock()
    store.get.return_value = state
    compressor = Mock()
    compressor.compress.return_value = ContextCompressor.parse_summary(_summary())
    entrypoint = _entrypoint(compressor)
    entrypoint.session_store = store

    entrypoint._persist_session(
        session_id="session-1",
        run_id="run-2",
        question="那2023年呢？",
        answer="本轮答案",
        plan=None,
    )

    saved = store.set.call_args.args[0]
    compressor.compress.assert_called_once()
    compressed_prefix = compressor.compress.call_args.args[1]
    assert compressed_prefix == existing[: -SUMMARY_KEEP_RECENT_MESSAGES + 2]
    assert saved.recent_messages[-2:] == [
        {"role": "user", "content": "那2023年呢？"},
        {"role": "assistant", "content": "本轮答案"},
    ]
    assert saved.last_run_id == "run-2"


def test_compression_selects_largest_complete_turn_prefix_within_prompt_budget():
    messages = _messages(40)
    for message in messages:
        message["content"] = message["content"] + ("x" * 1500)
    state = SessionState("session-1", recent_messages=copy.deepcopy(messages))
    compressor = Mock()
    compressor.compress.return_value = ContextCompressor.parse_summary(_summary())

    _entrypoint(compressor)._maybe_compress_session(state)

    batch = compressor.compress.call_args.args[1]
    pending_count = len(messages) - SUMMARY_KEEP_RECENT_MESSAGES
    assert 0 < len(batch) < pending_count
    assert len(batch) % 2 == 0
    assert batch[0]["role"] == "user" and batch[-1]["role"] == "assistant"
    assert ContextCompressor.prompt_length("", batch) <= COMPRESSION_MAX_CHARS
    next_turn = messages[len(batch) : len(batch) + 2]
    assert ContextCompressor.prompt_length("", batch + next_turn) > COMPRESSION_MAX_CHARS
    assert state.recent_messages == messages[len(batch) :]


def test_oversized_oldest_turn_fails_open_without_calling_llm_or_trimming():
    messages = _messages(SUMMARY_TRIGGER_MESSAGES + 2)
    messages[0]["content"] = "u" * COMPRESSION_MAX_CHARS
    messages[1]["content"] = "a" * COMPRESSION_MAX_CHARS
    state = SessionState("session-1", recent_messages=copy.deepcopy(messages))
    compressor = Mock()
    sink = _CollectingSink()

    with TraceRecorder(sink).run(session_id="session-1"):
        _entrypoint(compressor)._maybe_compress_session(state)

    compressor.compress.assert_not_called()
    assert state.recent_messages == messages
    event = next(
        item for item in sink.events if item.event_type == "context.compression.failed"
    )
    assert event.error_type == "CompressionBudgetExceeded"
    assert event.attributes["compressed_message_count"] == 0
    assert event.attributes["compression_input_chars"] > COMPRESSION_MAX_CHARS


def test_context_compressor_rejects_direct_over_budget_input_before_llm_call():
    llm = Mock()
    oversized = [
        {"role": "user", "content": "u" * COMPRESSION_MAX_CHARS},
        {"role": "assistant", "content": "a" * COMPRESSION_MAX_CHARS},
    ]

    with pytest.raises(ValueError, match="compression input exceeds"):
        ContextCompressor(llm).compress("", oversized)

    llm.assert_not_called()
