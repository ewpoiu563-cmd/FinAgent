"""Tests for the high-level Financial RAG evidence tool."""

from __future__ import annotations

from time import sleep
from unittest.mock import Mock

import pytest
import requests

from rag.embedding import EmbeddingAPIError
from rag.reranker import RerankerAPIError
from tools.document_retrieval_tool import (
    CANDIDATE_K,
    FINAL_K,
    DocumentRetrievalTool,
    _call_with_timeout,
)


def _candidate(index: int) -> dict:
    return {
        "rank": index,
        "rrf_score": 1 / (60 + index),
        "retrieval_unit_id": f"unit-{index}",
        "dense_rank": index,
        "bm25_rank": None,
        "chunk_id": f"chunk-{index}",
        "doc_id": "doc-a",
        "file_name": "report.pdf",
        "pages": [index],
        "chunk_type": "text",
        "headings": ["Section", str(index)],
        "text": f"evidence {index}",
    }


def _tool(*, candidate_error=None, rerank_error=None, candidates=None):
    pool = candidates if candidates is not None else [_candidate(i) for i in range(1, 21)]
    hybrid = Mock()
    hybrid.retrieve.side_effect = candidate_error
    if candidate_error is None:
        hybrid.retrieve.return_value = pool
    reranker = Mock()
    reranker.rerank_candidates.side_effect = rerank_error
    if rerank_error is None:
        reranked = []
        for candidate in reversed(pool):
            item = dict(candidate)
            item["rerank_score"] = float(candidate["rank"]) / 20
            reranked.append(item)
        reranker.rerank_candidates.return_value = reranked[:5]
    return DocumentRetrievalTool(
        hybrid_retriever=hybrid,
        reranker_retriever=reranker,
        timeout_runner=lambda call, timeout: call(),
    ), hybrid, reranker


def test_normal_retrieval_reuses_frozen_pipeline_configuration():
    tool, hybrid, reranker = _tool()

    result = tool.retrieve("公司的核心经营风险是什么？")

    assert result["success"] is True and result["degraded"] is False
    hybrid.retrieve.assert_called_once_with(
        result["query"], top_k=CANDIDATE_K, candidate_k=CANDIDATE_K
    )
    reranker.rerank_candidates.assert_called_once()
    assert reranker.rerank_candidates.call_args.kwargs == {"final_k": FINAL_K}
    assert result["retrieval_metadata"] == {
        "candidate_count": 20,
        "returned_count": 5,
        "reranker_used": True,
    }


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_empty_or_invalid_query_is_rejected(query):
    tool, hybrid, reranker = _tool()
    result = tool.retrieve(query)
    assert result["success"] is False
    assert result["error_type"] == "invalid_query"
    hybrid.retrieve.assert_not_called()
    reranker.rerank_candidates.assert_not_called()


@pytest.mark.parametrize(
    "error,error_type",
    [
        (RuntimeError("index unavailable"), "retrieval_error"),
        (EmbeddingAPIError("embedding unavailable"), "embedding_error"),
        (TimeoutError("candidate timeout"), "timeout"),
    ],
)
def test_candidate_retrieval_failure_is_fatal_and_mapped(error, error_type):
    tool, _, reranker = _tool(candidate_error=error)
    result = tool.retrieve("query")
    assert result["success"] is False
    assert result["error_type"] == error_type
    assert result["evidence"] == []
    reranker.rerank_candidates.assert_not_called()


def test_reranker_success_controls_final_order_and_scores():
    tool, _, _ = _tool()
    result = tool.retrieve("query")
    assert [item["retrieval_unit_id"] for item in result["evidence"]] == [
        "unit-20", "unit-19", "unit-18", "unit-17", "unit-16"
    ]
    assert all(item["rerank_score"] is not None for item in result["evidence"])


@pytest.mark.parametrize(
    "error,error_type",
    [
        (requests.Timeout("API timeout"), "timeout"),
        (RerankerAPIError("status_code=500"), "rerank_error"),
    ],
)
def test_reranker_failure_fails_closed_without_rrf_evidence(error, error_type):
    tool, _, _ = _tool(rerank_error=error)
    result = tool.retrieve("query")
    assert result["success"] is False
    assert result["degraded"] is False and result["fallback"] is None
    assert result["error_type"] == error_type
    assert result["retrieval_metadata"]["reranker_used"] is False
    assert result["evidence"] == []


def test_evidence_schema_and_source_file_mapping():
    tool, _, _ = _tool()
    evidence = tool.retrieve("query")["evidence"][0]
    assert set(evidence) == {
        "rank", "doc_id", "source_file", "page", "headings", "text",
        "retrieval_unit_id", "rerank_score",
    }
    assert evidence["source_file"] == "report.pdf"
    assert isinstance(evidence["page"], list)


def test_final_result_never_exceeds_top_five():
    tool, _, _ = _tool(candidates=[_candidate(i) for i in range(1, 8)])
    result = tool.retrieve("query")
    assert len(result["evidence"]) == FINAL_K
    assert result["retrieval_metadata"]["returned_count"] == FINAL_K


def test_timeout_runner_enforces_deadline():
    with pytest.raises(TimeoutError, match="exceeded"):
        _call_with_timeout(lambda: sleep(0.1), 0.01)
