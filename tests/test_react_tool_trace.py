"""Offline Tool Trace and generation failure regressions."""

import asyncio
import copy
import json
import logging
from unittest.mock import Mock

import pytest

import react_agent as agent
from orchestration.trace import TraceRecorder


TOOLS = ["query_financial_db", "retrieve_document"]


class CollectingSink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


def result_for(tool, **changes):
    result = {
        "success": True, "error_type": None, "error": None,
        "generated_sql": "SELECT n FROM funds", "columns": ["n"],
        "rows": [{"n": i} for i in range(8)], "row_count": 8,
        "retrieval_metadata": {"candidate_count": 20, "returned_count": 6},
        "degraded": True, "fallback": "rrf",
        "evidence": [{"rank": i, "doc_id": f"doc{i}", "source_file": "报告.pdf",
                      "page": [i], "headings": ["风险"], "rerank_score": 0.9,
                      "text": "风险\n" * 5000} for i in range(1, 7)],
    }
    result.update(changes)
    return result


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("success", [True, False])
def test_trace_schema_bounded_and_original_preserved(caplog, tool, success):
    result = result_for(tool, success=success, error_type="timeout", error="timed out")
    original = copy.deepcopy(result)
    caplog.set_level(logging.DEBUG, logger=agent.__name__)
    assert agent._call_traced_tool(tool, "查询", Mock(return_value=result)) is result
    assert result == original
    records = [r for r in caplog.records if r.message.startswith("TOOL_")]
    assert len(records) == 2
    assert all(r.levelno == logging.DEBUG for r in records)
    assert json.loads(records[0].message.split(": ", 1)[1]) == {"tool": tool, "input": "查询"}
    payload = json.loads(records[1].message.split(": ", 1)[1])
    assert payload["success"] is success
    assert payload["latency_ms"] >= 0
    assert payload["error_type"] == "timeout" and payload["error"] == "timed out"
    if tool == "query_financial_db":
        assert payload["generated_sql"] == original["generated_sql"]
        assert payload["columns"] == ["n"] and payload["row_count"] == 8
        assert payload["rows"] == original["rows"][:5]
    else:
        assert payload["candidate_count"] == 20 and payload["returned_count"] == 6
        assert payload["degraded"] is True and payload["fallback"] == "rrf"
        assert len(payload["evidence"]) == 5
        for logged, item in zip(payload["evidence"], original["evidence"]):
            assert "text" not in logged
            assert logged.pop("text_preview") == item["text"][:300]
            assert logged == {k: item[k] for k in logged}


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_exception_logged_and_propagated(caplog, tool):
    caplog.set_level(logging.DEBUG, logger=agent.__name__)
    error = TimeoutError("tool timeout")
    with pytest.raises(TimeoutError) as caught:
        agent._call_traced_tool(tool, "query", Mock(side_effect=error))
    assert caught.value is error
    payload = json.loads(caplog.records[-1].message.split(": ", 1)[1])
    assert payload["success"] is False and payload["error_type"] == "TimeoutError"


def test_trace_disabled_at_info(caplog):
    caplog.set_level(logging.INFO, logger=agent.__name__)
    agent._call_traced_tool(TOOLS[0], "query", Mock(return_value=result_for(TOOLS[0])))
    assert not caplog.records


def test_legacy_tool_call_persists_bounded_input_and_output(monkeypatch):
    monkeypatch.setattr(agent, "TRACE_INCLUDE_CONTENT", True)
    sink = CollectingSink()
    result = result_for("retrieve_document")

    with TraceRecorder(sink).run(run_id="legacy-tool-io"):
        agent._call_traced_tool(
            "retrieve_document",
            "健帆生物 经营风险",
            Mock(return_value=result),
            iteration=2,
        )

    tool_events = [event for event in sink.events if event.event_type.startswith("tool.call.")]
    assert [event.event_type for event in tool_events] == [
        "tool.call.started",
        "tool.call.completed",
    ]
    completed = tool_events[-1]
    assert completed.attributes["input"] == {"query": "健帆生物 经营风险"}
    assert completed.attributes["output"]["evidence"][0]["doc_id"] == "doc1"
    assert len(completed.attributes["output"]["evidence"][0]["text"]) == 1200
    assert completed.attributes["iteration"] == 2


def test_legacy_rag_forwards_catalog_resolved_physical_scope(monkeypatch):
    monkeypatch.setattr(agent, "TRACE_INCLUDE_CONTENT", True)
    sink = CollectingSink()
    result = result_for("retrieve_document")
    retrieve = Mock(return_value=result)
    allowed = ("physical-jianfan",)

    with TraceRecorder(sink).run(run_id="legacy-rag-scope"):
        agent._call_traced_tool(
            "retrieve_document",
            "健帆生物 经营风险",
            retrieve,
            iteration=1,
            allowed_doc_ids=allowed,
        )

    retrieve.assert_called_once_with(
        "健帆生物 经营风险",
        allowed_doc_ids=allowed,
    )
    completed = [
        event for event in sink.events if event.event_type == "tool.call.completed"
    ][0]
    assert completed.attributes["input"]["allowed_doc_ids"] == ["physical-jianfan"]


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("ending", ["timeout", "error", "no_finish", "finish", "later_tool_failure"])
def test_tool_success_requires_generation(monkeypatch, tool, ending):
    monkeypatch.setattr(agent, "_synthesize_document_answer", lambda *args: {
        "sufficient": False, "answer": None, "missing_information": ["更多证据"],
    })
    first = f"Thought: 答案是猜测值。\nAction: {tool}\nAction Input: query"
    outputs = [first]
    if ending == "finish":
        outputs += ["Action: finish\nAction Input: 已验证答案"]
    elif ending == "later_tool_failure":
        outputs += [f"Action: {tool}\nAction Input: other query", "Thought: 答案是猜测值。"]
    elif ending in ("timeout", "error"):
        error = TimeoutError if ending == "timeout" else RuntimeError
        outputs += [error("generation failed")] * 3
    else:
        outputs += ["Thought: 答案是猜测值。\nFinding: answer = 猜测值"]
    llm = Mock(side_effect=outputs)
    monkeypatch.setattr(agent, "MAX_ITERATIONS", len(outputs))
    monkeypatch.setattr(agent, "call_llm", llm)
    monkeypatch.setattr(agent, tool, Mock(side_effect=[result_for(tool), result_for(tool, success=False)]))
    regex = Mock(side_effect=AssertionError("Thought extraction must not run"))
    fallback = Mock(side_effect=AssertionError("No fallback LLM call"))
    monkeypatch.setattr(agent, "_regex_thought_extraction", regex)
    monkeypatch.setattr(agent, "_llm_fallback_extraction", fallback)
    answer = asyncio.run(agent.run_react_agent("查询本地数据"))
    assert answer == ("已验证答案" if ending == "finish" else agent._GENERATION_FAILURE.rstrip("。"))
    assert llm.call_count == len(outputs)
    regex.assert_not_called()
    fallback.assert_not_called()


def test_web_only_regex_fallback_unchanged(monkeypatch):
    state = agent.ReActState(question="test", trace=[{"thought": "答案是原有答案。"}])
    monkeypatch.setattr(agent, "_llm_fallback_extraction", lambda state: "")
    assert agent._extract_best_answer_from_trace(state) == "原有答案"


def test_successful_empty_result_does_not_mark_retrieval_succeeded(monkeypatch):
    outputs = ["Action: retrieve_document\nAction Input: query", "Thought: 答案是猜测值。"]
    monkeypatch.setattr(agent, "MAX_ITERATIONS", 2)
    monkeypatch.setattr(agent, "call_llm", Mock(side_effect=outputs))
    monkeypatch.setattr(agent, "retrieve_document", Mock(return_value={"success": True, "evidence": []}))
    fallback = Mock(return_value="")
    monkeypatch.setattr(agent, "_llm_fallback_extraction", fallback)
    asyncio.run(agent.run_react_agent("查询本地文档"))
    assert not fallback.call_args.args[0].document_retrieval_succeeded
