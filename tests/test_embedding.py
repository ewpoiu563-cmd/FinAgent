"""Unit tests for the native DashScope embedding adapter; no real API calls."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from rag.embedding import EmbeddingAPIError, EmbeddingClient, EmbeddingResponseError


DIMENSION = 1024


def _vector(value=0.0, dimension=DIMENSION):
    return [value] * dimension


def _response(vectors=(), *, status=200, indices=None):
    if indices is None:
        indices = range(len(vectors))
    embeddings = [
        {"text_index": index, "embedding": vector}
        for index, vector in zip(indices, vectors)
    ]
    response = Mock()
    response.status_code = status
    response.headers = {"x-request-id": "mock-request"}
    response.json.return_value = {
        "output": {"embeddings": embeddings},
        "code": None if status == 200 else "MockError",
        "message": None if status == 200 else "mock failure",
        "request_id": "mock-request",
    }
    return response


def _client(monkeypatch, **kwargs):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    request_post = kwargs.pop("request_post", Mock())
    return EmbeddingClient(request_post=request_post, **kwargs), request_post


def test_document_text_type_is_default(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response([_vector(1.0)])

    assert client.embed_texts(["营业收入增长"]) == [_vector(1.0)]
    request = embedding_call.call_args
    assert request.args[0].endswith("/text-embedding/text-embedding")
    assert request.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert request.kwargs["json"] == {
        "model": "qwen3.7-text-embedding-flash",
        "input": {"texts": ["营业收入增长"]},
        "parameters": {
            "text_type": "document", "dimension": 1024, "output_type": "dense"
        },
    }
    assert request.kwargs["timeout"] == (5.0, 30.0)


def test_query_text_type(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response([_vector()])

    client.embed_texts(["原材料风险"], text_type="query")

    assert embedding_call.call_args.kwargs["json"]["parameters"]["text_type"] == "query"


def test_text_index_restores_provider_order(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response(
        [_vector(2.0), _vector(1.0)],
        indices=[1, 0],
    )

    result = client.embed_texts(["a", "b"])

    assert [vector[0] for vector in result] == [1.0, 2.0]


def test_text_index_zero_is_valid(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response([_vector(7.0)], indices=[0])

    assert client.embed_texts(["a"]) == [_vector(7.0)]


def test_sequential_text_indices_are_valid(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response(
        [_vector(0.0), _vector(1.0), _vector(2.0), _vector(3.0)],
        indices=[0, 1, 2, 3],
    )

    result = client.embed_texts(["a", "b", "c", "d"])

    assert [vector[0] for vector in result] == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ("indices", "expected_message"),
    [
        ([None, 1], "missing or invalid text_index"),
        ([0, 0], "text_index mismatch"),
        ([0, 2], "out of range"),
        ([-1, 1], "out of range"),
    ],
)
def test_invalid_text_index_is_rejected(monkeypatch, indices, expected_message):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response(
        [_vector(), _vector()],
        indices=indices,
    )

    with pytest.raises(EmbeddingResponseError, match=expected_message):
        client.embed_texts(["a", "b"])


def test_batch_splitting_preserves_global_input_order(monkeypatch):
    client, embedding_call = _client(monkeypatch, batch_size=2)
    embedding_call.side_effect = [
        _response([_vector(2.0), _vector(1.0)], indices=[1, 0]),
        _response([_vector(4.0), _vector(3.0)], indices=[1, 0]),
        _response([_vector(5.0)]),
    ]

    result = client.embed_texts(["a", "b", "c", "d", "e"])

    assert [vector[0] for vector in result] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [
        call.kwargs["json"]["input"]["texts"] for call in embedding_call.call_args_list
    ] == [["a", "b"], ["c", "d"], ["e"]]


def test_missing_text_index_is_rejected(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response([_vector()])
    embedding_call.return_value.json.return_value["output"]["embeddings"] = [
        {"embedding": _vector()}
    ]

    with pytest.raises(EmbeddingResponseError, match="missing or invalid text_index"):
        client.embed_texts(["a"])


def test_embedding_count_mismatch(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response([_vector()])

    with pytest.raises(EmbeddingResponseError, match="count mismatch"):
        client.embed_texts(["a", "b"])


def test_embedding_dimension_mismatch(monkeypatch):
    client, embedding_call = _client(monkeypatch)
    embedding_call.return_value = _response([_vector(dimension=8)])

    with pytest.raises(EmbeddingResponseError, match="dimension mismatch"):
        client.embed_texts(["a"])


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_then_success(monkeypatch, status):
    sleeps = []
    client, embedding_call = _client(
        monkeypatch,
        max_retries=2,
        backoff_factor=0.25,
        sleep=sleeps.append,
    )
    embedding_call.side_effect = [
        _response(status=status),
        _response([_vector()]),
    ]

    assert client.embed_texts(["a"]) == [_vector()]
    assert embedding_call.call_count == 2
    assert sleeps == [0.25]


def test_transient_network_error_then_success(monkeypatch):
    client, embedding_call = _client(
        monkeypatch,
        max_retries=1,
        sleep=lambda _: None,
    )
    embedding_call.side_effect = [
        requests.Timeout("temporary"),
        _response([_vector()]),
    ]

    assert client.embed_texts(["a"]) == [_vector()]


def test_exhausted_retries_fail(monkeypatch):
    client, embedding_call = _client(
        monkeypatch,
        max_retries=2,
        sleep=lambda _: None,
    )
    embedding_call.return_value = _response(status=500)

    with pytest.raises(EmbeddingAPIError, match="status_code=500"):
        client.embed_texts(["a"])
    assert embedding_call.call_count == 3


def test_non_retryable_status_fails_immediately(monkeypatch):
    client, embedding_call = _client(monkeypatch, max_retries=3)
    embedding_call.return_value = _response(status=400)

    with pytest.raises(EmbeddingAPIError, match="status_code=400"):
        client.embed_texts(["a"])
    assert embedding_call.call_count == 1


def test_empty_list_returns_without_api_call(monkeypatch):
    client, embedding_call = _client(monkeypatch)

    assert client.embed_texts([]) == []
    embedding_call.assert_not_called()


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        EmbeddingClient()


@pytest.mark.parametrize("texts", [[""], ["   "], [None], "not-a-list"])
def test_invalid_input(monkeypatch, texts):
    client, embedding_call = _client(monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        client.embed_texts(texts)
    embedding_call.assert_not_called()


@pytest.mark.parametrize("text_type", ["", "passage", None])
def test_invalid_text_type(monkeypatch, text_type):
    client, embedding_call = _client(monkeypatch)

    with pytest.raises(ValueError, match="text_type"):
        client.embed_texts(["a"], text_type=text_type)
    embedding_call.assert_not_called()


def test_model_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("FINAGENT_EMBEDDING_MODEL", "custom-model")
    client, _ = _client(monkeypatch)

    assert client.model == "custom-model"


@pytest.mark.parametrize("timeout", [20, 30])
def test_read_timeout_is_passed_to_requests(monkeypatch, timeout):
    client, request_post = _client(monkeypatch, timeout=timeout)
    request_post.return_value = _response([_vector()])

    client.embed_texts(["a"])

    assert request_post.call_args.kwargs["timeout"] == (5.0, float(timeout))
