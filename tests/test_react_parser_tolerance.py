import asyncio
import json
from unittest.mock import Mock

import pytest
import react_agent as agent


@pytest.mark.parametrize("action,key", [("search", "query"), ("query_financial_db", "question"),
                                      ("retrieve_document", "question"), ("fetch", "url"), ("finish", "answer")])
@pytest.mark.parametrize("style", ["plain", "call", "json", "multiline_json"])
def test_normalize(action, key, style):
    value = '中文 question (2025), "quoted"'
    if style == "call":
        raw = f"Action: {action}({key}={json.dumps(value)})"
    else:
        payload = value if style == "plain" else json.dumps({key: value}, indent=2 if style == "multiline_json" else None)
        raw = f"Action: {action}\nAction Input: {payload}"
    parsed = agent._parse_react_output(raw.replace("retrieve_document", r"retrieve\_document"))
    assert parsed["action"] == action
    assert parsed["action_input"] == value


@pytest.mark.parametrize("raw", ["Thought: no action", "Action: imaginary\nAction Input: x",
                                  'Action: search(query=42)', 'Action: search\nAction Input: {"query": []}',
                                  "Thought: " + "x" * 40000], ids=["missing", "unknown", "numeric", "array", "truncated"])
def test_format_errors_stop_at_three(monkeypatch, raw):
    llm = Mock(return_value=raw)
    monkeypatch.setattr(agent, "call_llm", llm)
    monkeypatch.setattr(agent, "MAX_ITERATIONS", 20)
    assert "generation failure" in asyncio.run(agent.run_react_agent("测试"))
    assert llm.call_count == 3
