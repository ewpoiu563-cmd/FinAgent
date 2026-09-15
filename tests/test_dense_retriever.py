"""Unit tests for the Phase 2 dense retriever."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from rag.dense_retriever import DenseRetriever
from rag.embedding import DEFAULT_MODEL
from rag.vector_store import FaissVectorStore


DIMENSION = 1024


def _vector(*values: float) -> list[float]:
    return [*values, *([0.0] * (DIMENSION - len(values)))]


def _chunk(index: int, *, fallback: bool = False) -> dict:
    chunk = {
        "chunk_id": f"chunk-{index}",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [index + 1],
        "chunk_type": "table" if index == 1 else "text",
        "headings": ["风险因素", f"Section {index}"],
        "text": f"chunk text {index}",
        "embedding_text": f"contextual text {index}",
    }
    if fallback:
        chunk.update(
            parent_chunk_id="parent-1",
            fallback_split=True,
            part_index=1,
            part_count=2,
        )
    return chunk


def _save_store(path) -> None:
    store = FaissVectorStore(dimension=DIMENSION, embedding_model=DEFAULT_MODEL)
    store.add(
        [_vector(1, 0), _vector(0.8, 0.6), _vector(-1, 0)],
        [_chunk(0), _chunk(1, fallback=True), _chunk(2)],
    )
    store.save(path)


def _client(vector: list[float] | None = None) -> Mock:
    client = Mock()
    client.model = DEFAULT_MODEL
    client.dimension = DIMENSION
    client.embed_texts.return_value = [vector if vector is not None else _vector(1, 0)]
    return client


def test_normal_top_k_and_structured_metadata(tmp_path):
    _save_store(tmp_path)

    results = DenseRetriever(tmp_path, embedding_client=_client()).retrieve("风险", top_k=2)

    assert [result["chunk_id"] for result in results] == ["chunk-0", "chunk-1"]
    assert results[0] == {
        "rank": 1,
        "score": pytest.approx(1.0),
        "chunk_id": "chunk-0",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [1],
        "chunk_type": "text",
        "headings": ["风险因素", "Section 0"],
        "text": "chunk text 0",
    }
    assert results[1]["parent_chunk_id"] == "parent-1"
    assert results[1]["fallback_split"] is True
    assert results[1]["part_index"] == 1
    assert results[1]["part_count"] == 2


def test_rank_and_score_follow_faiss_order(tmp_path):
    _save_store(tmp_path)

    results = DenseRetriever(tmp_path, embedding_client=_client()).retrieve("风险", top_k=3)

    assert [result["rank"] for result in results] == [1, 2, 3]
    assert [result["score"] for result in results] == sorted(
        (result["score"] for result in results), reverse=True
    )


def test_query_embedding_uses_query_text_type(tmp_path):
    _save_store(tmp_path)
    client = _client()

    DenseRetriever(tmp_path, embedding_client=client).retrieve("现金流")

    client.embed_texts.assert_called_once_with(["现金流"], text_type="query")


@pytest.mark.parametrize("top_k, expected_count", [(1, 1), (2, 2), (99, 3)])
def test_top_k_controls_result_count(tmp_path, top_k, expected_count):
    _save_store(tmp_path)

    results = DenseRetriever(tmp_path, embedding_client=_client()).retrieve(
        "风险", top_k=top_k
    )

    assert len(results) == expected_count


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2", None])
def test_invalid_top_k_is_rejected_before_embedding(tmp_path, top_k):
    _save_store(tmp_path)
    client = _client()
    retriever = DenseRetriever(tmp_path, embedding_client=client)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("风险", top_k=top_k)

    client.embed_texts.assert_not_called()


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected_before_embedding(tmp_path, query):
    _save_store(tmp_path)
    client = _client()
    retriever = DenseRetriever(tmp_path, embedding_client=client)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve(query)

    client.embed_texts.assert_not_called()


def test_default_client_is_pinned_to_loaded_index(monkeypatch):
    store = Mock()
    store.embedding_model = "index-model"
    store.dimension = 768
    monkeypatch.setattr(FaissVectorStore, "load", Mock(return_value=store))
    constructed = _client()
    constructed.model = "index-model"
    constructed.dimension = 768
    client_class = Mock(return_value=constructed)
    monkeypatch.setattr("rag.dense_retriever.EmbeddingClient", client_class)

    DenseRetriever("unused-index-path")

    client_class.assert_called_once_with(model="index-model", dimension=768)


def test_environment_model_cannot_override_loaded_index(tmp_path, monkeypatch):
    _save_store(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "unused-test-key")
    monkeypatch.setenv("FINAGENT_EMBEDDING_MODEL", "wrong-environment-model")

    retriever = DenseRetriever(tmp_path)

    assert retriever.embedding_client.model == DEFAULT_MODEL
    assert retriever.embedding_client.dimension == DIMENSION


@pytest.mark.parametrize(
    "model, dimension, message",
    [
        ("wrong-model", DIMENSION, "embedding model mismatch"),
        (DEFAULT_MODEL, 768, "embedding dimension mismatch"),
    ],
)
def test_injected_client_must_match_index(tmp_path, model, dimension, message):
    _save_store(tmp_path)
    client = _client()
    client.model = model
    client.dimension = dimension

    with pytest.raises(ValueError, match=message):
        DenseRetriever(tmp_path, embedding_client=client)


def test_query_vector_is_passed_unchanged_to_vector_store(monkeypatch):
    query_vector = [3.0, 4.0]
    store = Mock()
    store.embedding_model = "model"
    store.dimension = 2
    store.search.return_value = []
    monkeypatch.setattr(FaissVectorStore, "load", Mock(return_value=store))
    client = Mock(model="model", dimension=2)
    client.model = "model"
    client.dimension = 2
    client.embed_texts.return_value = [query_vector]

    DenseRetriever("unused-index-path", embedding_client=client).retrieve("query", top_k=1)

    store.search.assert_called_once_with(query_vector, top_k=1)


def test_embedding_exception_propagates_unchanged(tmp_path):
    _save_store(tmp_path)
    client = _client()
    failure = RuntimeError("mock API failure")
    client.embed_texts.side_effect = failure

    with pytest.raises(RuntimeError, match="mock API failure") as caught:
        DenseRetriever(tmp_path, embedding_client=client).retrieve("风险")

    assert caught.value is failure


def test_index_load_exception_propagates_unchanged(monkeypatch):
    failure = OSError("mock corrupt index")
    monkeypatch.setattr(FaissVectorStore, "load", Mock(side_effect=failure))

    with pytest.raises(OSError, match="mock corrupt index") as caught:
        DenseRetriever("missing")

    assert caught.value is failure
