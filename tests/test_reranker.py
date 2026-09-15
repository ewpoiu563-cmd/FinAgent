"""Mock/unit coverage for the fixed-pool DashScope reranker."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from rag.reranker import (
    DASHSCOPE_RERANK_URL,
    DashScopeRerankerClient,
    RerankerAPIError,
    RerankerResponseError,
    RerankerRetriever,
    candidate_text,
)


def _response(results, status=200, payload=None):
    response = Mock(status_code=status, headers={"x-request-id": "req-test"})
    response.json.return_value = payload if payload is not None else {"output": {"results": results}}
    return response


def _candidate(index: int, *, score: float = 0.02):
    return {"rank": index, "rrf_score": score, "retrieval_unit_id": f"unit-{index}", "dense_rank": index,
            "bm25_rank": None, "chunk_id": f"chunk-{index}", "doc_id": "doc-a", "file_name": "a.pdf",
            "pages": [index], "chunk_type": "text", "headings": ["Section", str(index)], "text": f"body {index}",
            "parent_chunk_id": f"parent-{index}"}


def test_client_sends_one_batch_and_restores_input_order():
    post = Mock(return_value=_response([{"index": 1, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.9}]))
    client = DashScopeRerankerClient(api_key="key", request_post=post, sleep=Mock())
    assert client.rerank("question", ["first", "second"]) == [0.9, 0.2]
    assert post.call_args.args[0] == DASHSCOPE_RERANK_URL
    assert post.call_args.kwargs["json"] == {"model": "qwen3.7-text-rerank", "input": {"query": "question", "documents": ["first", "second"]}, "parameters": {"return_documents": False}}
    assert post.call_args.kwargs["timeout"] == (5.0, 30.0)


def test_client_retries_timeout_then_maps_exhausted_transport_error():
    post = Mock(side_effect=[requests.Timeout(), _response([{"index": 0, "relevance_score": 0.5}])])
    client = DashScopeRerankerClient(api_key="key", request_post=post, sleep=Mock(), max_retries=1)
    assert client.rerank("q", ["d"]) == [0.5]
    assert post.call_count == 2
    failed = DashScopeRerankerClient(api_key="key", request_post=Mock(return_value=_response([], status=401, payload={"code": "InvalidApiKey", "message": "bad"})), sleep=Mock())
    with pytest.raises(RerankerAPIError, match="status_code=401.*InvalidApiKey"):
        failed.rerank("q", ["d"])


def test_client_rejects_invalid_response_coverage():
    client = DashScopeRerankerClient(api_key="key", request_post=Mock(return_value=_response([{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}])), sleep=Mock())
    with pytest.raises(RerankerResponseError, match="invalid score/index"):
        client.rerank("q", ["a", "b"])


def test_candidate_text_requires_headings_and_text():
    assert candidate_text(_candidate(1)) == "Headings: Section > 1\n\nText: body 1"
    with pytest.raises(RerankerResponseError, match="headings"):
        candidate_text({"headings": "wrong", "text": "body"})


def test_reranker_only_changes_order_and_preserves_candidate_metadata():
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    client = Mock()
    client.rerank.return_value = [0.2, 0.9, 0.5]
    retriever = RerankerRetriever(Mock(), reranker_client=client)
    results = retriever.rerank_candidates("question", candidates)
    assert [item["retrieval_unit_id"] for item in results] == ["unit-2", "unit-3", "unit-1"]
    assert [item["rank"] for item in results] == [1, 2, 3]
    assert results[0]["doc_id"] == "doc-a"
    assert results[0]["parent_chunk_id"] == "parent-2"
    documents = client.rerank.call_args.args[1]
    assert len(documents) == 3 and all("Headings:" in text and "Text:" in text for text in documents)


def test_fixed_configuration_and_candidate_pool_limit_are_enforced():
    retriever = RerankerRetriever(Mock(), reranker_client=Mock())
    with pytest.raises(ValueError, match="candidate_k=20"):
        retriever.retrieve("q", candidate_k=19)
    with pytest.raises(ValueError, match="final_k=5"):
        retriever.rerank_candidates("q", [_candidate(1)], final_k=4)
    with pytest.raises(ValueError, match="1 through 20"):
        retriever.rerank_candidates("q", [_candidate(index) for index in range(1, 22)])
