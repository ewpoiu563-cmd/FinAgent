"""Tests for dense retrieval; query embeddings are always mocked."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from rag.retriever import (
    DenseRetriever,
    DenseRetrieverError,
    IndexLoadError,
    QueryEmbeddingError,
)
from rag.vector_store import FaissVectorStore


DIMENSION = 1024


def _vector(*values: float) -> list[float]:
    return [*values, *([0.0] * (DIMENSION - len(values)))]


def _chunk(index: int) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [index + 1, index + 2],
        "chunk_type": "table" if index == 1 else "text",
        "headings": ["风险因素", f"Section {index}"],
        "text": f"final text {index}",
        "embedding_text": f"context only {index}",
        "source": {"path": "data/report.pdf"},
    }


def _save_store(path, *, empty: bool = False) -> None:
    store = FaissVectorStore()
    if not empty:
        store.add(
            [_vector(1, 0), _vector(0.8, 0.6), _vector(-1, 0)],
            [_chunk(0), _chunk(1), _chunk(2)],
        )
    store.save(path)


def _client(vector=None) -> Mock:
    client = Mock()
    client.embed_texts.return_value = [vector or _vector(1, 0)]
    return client


def test_query_vector_search_normal_path_and_query_text_type(tmp_path):
    _save_store(tmp_path)
    client = _client()

    results = DenseRetriever(tmp_path, embedding_client=client).retrieve(
        "公司的应收账款风险是什么？", top_k=2
    )

    client.embed_texts.assert_called_once_with(
        ["公司的应收账款风险是什么？"], text_type="query"
    )
    assert [result["chunk_id"] for result in results] == ["chunk-0", "chunk-1"]


def test_top_k_results_have_one_based_rank_in_similarity_order(tmp_path):
    _save_store(tmp_path)

    results = DenseRetriever(tmp_path, embedding_client=_client()).retrieve("risk", top_k=3)

    assert [result["rank"] for result in results] == [1, 2, 3]
    assert [result["score"] for result in results] == sorted(
        [result["score"] for result in results], reverse=True
    )


def test_metadata_is_complete_and_final_text_is_not_embedding_text(tmp_path):
    _save_store(tmp_path)

    result = DenseRetriever(tmp_path, embedding_client=_client()).retrieve("risk", top_k=1)[0]

    assert result["chunk_id"] == "chunk-0"
    assert result["doc_id"] == "doc-1"
    assert result["file_name"] == "report.pdf"
    assert result["pages"] == [1, 2]
    assert result["chunk_type"] == "text"
    assert result["headings"] == ["风险因素", "Section 0"]
    assert result["text"] == "final text 0"
    assert result["source"] == {"path": "data/report.pdf"}


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected_without_embedding(tmp_path, query):
    _save_store(tmp_path)
    client = _client()
    retriever = DenseRetriever(tmp_path, embedding_client=client)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve(query)

    client.embed_texts.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2", None])
def test_invalid_top_k_is_rejected_without_embedding(tmp_path, top_k):
    _save_store(tmp_path)
    client = _client()
    retriever = DenseRetriever(tmp_path, embedding_client=client)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("risk", top_k=top_k)

    client.embed_texts.assert_not_called()


def test_embedding_failure_is_explicit_and_preserves_cause(tmp_path):
    _save_store(tmp_path)
    client = _client()
    failure = RuntimeError("mock API failure")
    client.embed_texts.side_effect = failure

    with pytest.raises(QueryEmbeddingError, match="mock API failure") as caught:
        DenseRetriever(tmp_path, embedding_client=client).retrieve("risk")

    assert caught.value.__cause__ is failure


def test_query_embedding_dimension_mismatch_is_rejected_before_search(tmp_path):
    _save_store(tmp_path)

    with pytest.raises(QueryEmbeddingError, match="dimension mismatch"):
        DenseRetriever(tmp_path, embedding_client=_client([1.0, 0.0])).retrieve("risk")


def test_empty_index_is_rejected_without_embedding(tmp_path):
    _save_store(tmp_path, empty=True)
    client = _client()

    with pytest.raises(DenseRetrieverError, match="empty index"):
        DenseRetriever(tmp_path, embedding_client=client).retrieve("risk")

    client.embed_texts.assert_not_called()


def test_index_load_failure_is_explicit_and_preserves_cause(tmp_path, monkeypatch):
    failure = OSError("mock corrupt index")

    def fail_load(_path):
        raise failure

    monkeypatch.setattr(FaissVectorStore, "load", fail_load)

    with pytest.raises(IndexLoadError, match="mock corrupt index") as caught:
        DenseRetriever(tmp_path, embedding_client=_client())

    assert caught.value.__cause__ is failure


def test_missing_index_directory_is_rejected(tmp_path):
    with pytest.raises(IndexLoadError, match="does not exist"):
        DenseRetriever(tmp_path / "missing", embedding_client=_client())
