"""Tests for retrieval-unit-normalized reciprocal-rank fusion."""

from __future__ import annotations

import pytest

from rag.hybrid_retriever import HybridRetriever
from eval.evaluate_hybrid_retrieval import score_case as score_hybrid_case


def _result(rank: int, chunk_id: str, parent: str | None = None) -> dict:
    result = {
        "rank": rank,
        "score": 100.0 - rank,
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [rank],
        "chunk_type": "text",
        "headings": ["heading"],
        "text": chunk_id,
    }
    if parent is not None:
        result.update(
            parent_chunk_id=parent,
            fallback_split=True,
            part_index=rank,
            part_count=3,
        )
    return result


class _StubRetriever:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        if self.error:
            raise self.error
        return self.results[:top_k]


def _hybrid(dense, bm25):
    return HybridRetriever(
        dense_retriever=_StubRetriever(dense),
        bm25_retriever=_StubRetriever(bm25),
    )


def test_same_unit_from_both_retrievers_sums_rrf_contributions():
    result = _hybrid([_result(1, "shared")], [_result(2, "shared")]).retrieve("q")

    assert len(result) == 1
    assert result[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert result[0]["dense_rank"] == 1
    assert result[0]["bm25_rank"] == 2
    assert result[0]["contributing_retrievers"] == ["dense", "bm25"]


def test_dense_only_and_bm25_only_units_are_retained():
    results = _hybrid(
        [_result(1, "dense-only")], [_result(1, "bm25-only")]
    ).retrieve("q")

    by_id = {result["retrieval_unit_id"]: result for result in results}
    assert by_id["dense-only"]["bm25_rank"] is None
    assert by_id["dense-only"]["contributing_retrievers"] == ["dense"]
    assert by_id["bm25-only"]["dense_rank"] is None
    assert by_id["bm25-only"]["contributing_retrievers"] == ["bm25"]


def test_sibling_fallback_children_contribute_only_best_rank_per_retriever():
    dense = [
        _result(1, "parent__part_001", "parent"),
        _result(2, "parent__part_002", "parent"),
    ]
    bm25 = [_result(3, "parent__part_003", "parent")]

    result = _hybrid(dense, bm25).retrieve("q")[0]

    assert result["rrf_score"] == pytest.approx(1 / 61 + 1 / 63)
    assert result["chunk_id"] == "parent__part_001"
    assert result["parent_chunk_id"] == "parent"


def test_rrf_formula_orders_two_medium_ranks_above_one_first_rank():
    dense = [_result(1, "dense-only"), _result(2, "shared")]
    bm25 = [_result(1, "bm25-only"), _result(2, "shared")]

    results = _hybrid(dense, bm25).retrieve("q", top_k=3)

    assert [result["retrieval_unit_id"] for result in results] == [
        "shared",
        "bm25-only",
        "dense-only",
    ]
    assert results[0]["rrf_score"] == pytest.approx(2 / 62)


def test_deterministic_tie_break_and_representative_selection():
    dense = [_result(1, "z-unit"), _result(2, "shared-z", "shared")]
    bm25 = [_result(1, "a-unit"), _result(2, "shared-a", "shared")]

    first = _hybrid(dense, bm25).retrieve("q", top_k=3)
    second = _hybrid(dense, bm25).retrieve("q", top_k=3)

    assert first == second
    assert [item["retrieval_unit_id"] for item in first] == ["shared", "a-unit", "z-unit"]
    assert first[0]["chunk_id"] == "shared-z"  # equal rank: Dense precedes BM25


@pytest.mark.parametrize(
    "field,value",
    [("top_k", 0), ("top_k", True), ("candidate_k", -1), ("candidate_k", 1.5)],
)
def test_invalid_top_k_and_candidate_k_are_rejected(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        _hybrid([], []).retrieve("q", **kwargs)


def test_top_k_cannot_exceed_candidate_k():
    with pytest.raises(ValueError, match="must not exceed"):
        _hybrid([], []).retrieve("q", top_k=6, candidate_k=5)


def test_candidate_k_is_forwarded_to_both_retrievers():
    dense = _StubRetriever([_result(1, "dense")])
    bm25 = _StubRetriever([_result(1, "bm25")])
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    retriever.retrieve("query", candidate_k=20)

    assert dense.calls == [("query", 20)]
    assert bm25.calls == [("query", 20)]


@pytest.mark.parametrize("failing", ["dense", "bm25"])
def test_retriever_exceptions_are_not_silently_swallowed(failing):
    error = RuntimeError(f"{failing} failed")
    dense = _StubRetriever(error=error if failing == "dense" else None)
    bm25 = _StubRetriever(error=error if failing == "bm25" else None)
    retriever = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    with pytest.raises(RuntimeError, match=f"{failing} failed"):
        retriever.retrieve("query")


def test_hybrid_eval_equivalent_units_in_one_group_use_or_semantics():
    case = {
        "query_id": "RAG-020",
        "question": "question",
        "gold_evidence": [
            {"retrieval_unit_id": "equivalent-a", "evidence_group": "answer"},
            {"retrieval_unit_id": "equivalent-b", "evidence_group": "answer"},
        ],
    }

    record = score_hybrid_case(case, [_result(1, "equivalent-a")])

    assert record["recall_at_5"] == pytest.approx(1.0)
