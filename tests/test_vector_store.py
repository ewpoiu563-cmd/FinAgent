"""Tests for the local FAISS vector store; all vectors are synthetic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag.vector_store import FaissVectorStore


DIMENSION = 1024


def _vector(*values: float) -> list[float]:
    return [*values, *([0.0] * (DIMENSION - len(values)))]


def _chunk(index: int) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "doc_id": "doc-1",
        "file_name": "report.pdf",
        "pages": [index + 1],
        "chunk_type": "table" if index == 2 else "text",
        "headings": [f"Section {index}"],
        "text": f"raw text {index}",
        "embedding_text": f"contextual text {index}",
    }


def _populated_store() -> FaissVectorStore:
    store = FaissVectorStore()
    store.add(
        [_vector(1, 0), _vector(0, 1), _vector(1, 1)],
        [_chunk(0), _chunk(1), _chunk(2)],
    )
    return store


def test_add_and_search_returns_results():
    store = _populated_store()

    results = store.search(_vector(1, 0), top_k=2)

    assert store.vector_count == 3
    assert len(results) == 2
    assert results[0]["chunk"]["chunk_id"] == "chunk-0"
    assert results[0]["score"] == pytest.approx(1.0)


def test_cosine_ranking_is_correct_for_unnormalized_vectors():
    store = FaissVectorStore()
    store.add(
        [_vector(100, 0), _vector(2, 2), _vector(-1, 0)],
        [_chunk(0), _chunk(1), _chunk(2)],
    )

    results = store.search(_vector(10, 1), top_k=3)

    assert [result["chunk"]["chunk_id"] for result in results] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]


def test_metadata_stays_aligned_with_vector_position():
    store = FaissVectorStore()
    store.add(
        [_vector(1, 0), _vector(0, 1), _vector(-1, 0)],
        [_chunk(10), _chunk(11), _chunk(12)],
    )

    assert store.search(_vector(0, 5), top_k=1)[0]["chunk"] == _chunk(11)
    assert store.search(_vector(-3, 0), top_k=1)[0]["chunk"] == _chunk(12)


def test_save_and_load_files_and_metadata(tmp_path):
    store = _populated_store()
    store.save(tmp_path)
    loaded = FaissVectorStore.load(tmp_path)

    assert (tmp_path / "financial.faiss").is_file()
    assert (tmp_path / "metadata.jsonl").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert loaded.vector_count == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": "1.0",
        "index_type": "IndexFlatIP",
        "dimension": 1024,
        "embedding_model": "qwen3.7-text-embedding-flash",
        "normalization": "l2",
        "vector_count": 3,
        "created_at": store.created_at,
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["position"] for row in rows] == [0, 1, 2]
    assert rows[2]["chunk"] == _chunk(2)


def test_save_load_preserves_search_results(tmp_path):
    store = _populated_store()
    before = store.search(_vector(0.4, 0.8), top_k=3)
    store.save(tmp_path)

    after = FaissVectorStore.load(tmp_path).search(_vector(0.4, 0.8), top_k=3)

    assert after == before


def test_add_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        FaissVectorStore().add([[1.0, 2.0]], [_chunk(0)])


def test_search_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        _populated_store().search([1.0, 2.0])


def test_add_rejects_duplicate_chunk_id_within_batch():
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        FaissVectorStore().add([_vector(1), _vector(2)], [_chunk(0), _chunk(0)])


def test_add_rejects_duplicate_chunk_id_from_earlier_batch():
    store = FaissVectorStore()
    store.add([_vector(1)], [_chunk(0)])

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        store.add([_vector(2)], [_chunk(0)])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_add_rejects_nan_and_infinity(bad_value):
    with pytest.raises(ValueError, match="NaN or infinity"):
        FaissVectorStore().add([_vector(bad_value)], [_chunk(0)])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_search_rejects_nan_and_infinity(bad_value):
    with pytest.raises(ValueError, match="NaN or infinity"):
        _populated_store().search(_vector(bad_value))


def test_add_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        FaissVectorStore().add([], [])


def test_search_rejects_empty_index():
    with pytest.raises(ValueError, match="empty index"):
        FaissVectorStore().search(_vector(1))


def test_top_k_larger_than_vector_count_returns_all_vectors():
    assert len(_populated_store().search(_vector(1), top_k=99)) == 3


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2", None])
def test_search_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        _populated_store().search(_vector(1), top_k=top_k)


def test_add_rejects_count_mismatch_without_mutating_store():
    store = FaissVectorStore()

    with pytest.raises(ValueError, match="vector count must equal chunk count"):
        store.add([_vector(1), _vector(2)], [_chunk(0)])

    assert store.vector_count == 0


def test_rejects_zero_norm_vectors():
    with pytest.raises(ValueError, match="zero-norm"):
        FaissVectorStore().add([_vector(0)], [_chunk(0)])
