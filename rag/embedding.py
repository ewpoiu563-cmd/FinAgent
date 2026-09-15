"""Lightweight HTTP adapter for DashScope's native text embedding API."""

from __future__ import annotations

import os
import time
from http import HTTPStatus
from typing import Any, Callable, Mapping, Sequence

import requests
DEFAULT_MODEL = "qwen3.7-text-embedding-flash"
DEFAULT_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5
OUTPUT_TYPE = "dense"
VALID_TEXT_TYPES = frozenset({"document", "query"})
DASHSCOPE_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "text-embedding/text-embedding"
)
CONNECT_TIMEOUT = 5.0


class EmbeddingError(RuntimeError):
    """Base error raised by the embedding adapter."""


class EmbeddingAPIError(EmbeddingError):
    """Raised when DashScope returns a non-success response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmbeddingResponseError(EmbeddingError):
    """Raised when DashScope returns an invalid embedding response."""


class EmbeddingClient:
    """Synchronous client for DashScope's native ``TextEmbedding`` API.

    ``max_retries`` is the number of retries after the initial request. Only
    rate limits, server errors, timeouts, and connection failures are retried.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        request_post: Callable[..., Any] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not resolved_key:
            raise ValueError("DASHSCOPE_API_KEY is required for embeddings")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor < 0:
            raise ValueError("backoff_factor must be non-negative")

        self.api_key = resolved_key
        self.model = model or os.getenv("FINAGENT_EMBEDDING_MODEL") or DEFAULT_MODEL
        self.dimension = dimension
        self.batch_size = batch_size
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.backoff_factor = float(backoff_factor)
        self._request_post = request_post
        self._sleep = sleep

    def embed_texts(
        self,
        texts: list[str],
        text_type: str = "document",
    ) -> list[list[float]]:
        """Embed texts in stable input order using document or query semantics."""

        self._validate_texts(texts)
        if text_type not in VALID_TEXT_TYPES:
            raise ValueError("text_type must be 'document' or 'query'")
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch, text_type=text_type))

        if len(embeddings) != len(texts):
            raise EmbeddingResponseError(
                f"embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        return embeddings

    @staticmethod
    def _validate_texts(texts: object) -> None:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str]")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be str, got {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"texts[{index}] must not be empty or whitespace-only")

    def _embed_batch(
        self,
        texts: Sequence[str],
        *,
        text_type: str,
    ) -> list[list[float]]:
        response = self._call_with_retry(texts, text_type=text_type)
        output = self._field(response, "output")
        embeddings = self._field(output, "embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingResponseError(
                "embedding response output.embeddings must be a list"
            )
        if len(embeddings) != len(texts):
            raise EmbeddingResponseError(
                f"embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        indexed_vectors: list[tuple[int, Any]] = []
        for item_position, item in enumerate(embeddings):
            if isinstance(item, Mapping):
                if "text_index" in item:
                    text_index = item["text_index"]
                elif "index" in item:
                    text_index = item["index"]
                else:
                    raise EmbeddingResponseError(
                        f"embedding item {item_position} has missing or invalid text_index"
                    )
            elif hasattr(item, "text_index"):
                text_index = getattr(item, "text_index")
            elif hasattr(item, "index"):
                text_index = getattr(item, "index")
            else:
                raise EmbeddingResponseError(
                    f"embedding item {item_position} has missing or invalid text_index"
                )
            if not isinstance(text_index, int) or isinstance(text_index, bool):
                raise EmbeddingResponseError(
                    f"embedding item {item_position} has missing or invalid text_index"
                )
            if not 0 <= text_index < len(texts):
                raise EmbeddingResponseError(
                    f"embedding text_index {text_index} is out of range for batch size {len(texts)}"
                )
            indexed_vectors.append((text_index, self._field(item, "embedding")))

        actual_indices = sorted(index for index, _ in indexed_vectors)
        expected_indices = list(range(len(texts)))
        if actual_indices != expected_indices:
            raise EmbeddingResponseError(
                f"embedding text_index mismatch: expected {expected_indices}, "
                f"got {actual_indices}"
            )

        result: list[list[float]] = []
        for text_index, vector in sorted(indexed_vectors, key=lambda pair: pair[0]):
            if not isinstance(vector, list):
                raise EmbeddingResponseError(
                    f"embedding {text_index} vector is not a list"
                )
            if len(vector) != self.dimension:
                raise EmbeddingResponseError(
                    f"embedding {text_index} dimension mismatch: "
                    f"expected {self.dimension}, got {len(vector)}"
                )
            if any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in vector
            ):
                raise EmbeddingResponseError(
                    f"embedding {text_index} contains a non-numeric value"
                )
            result.append([float(value) for value in vector])
        return result

    def _call_with_retry(self, texts: Sequence[str], *, text_type: str) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request_post(
                    DASHSCOPE_EMBEDDING_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": {"texts": list(texts)},
                        "parameters": {
                            "text_type": text_type,
                            "dimension": self.dimension,
                            "output_type": OUTPUT_TYPE,
                        },
                    },
                    timeout=(CONNECT_TIMEOUT, self.timeout),
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt >= self.max_retries:
                    raise
                self._backoff(attempt)
                continue

            status_code = self._field(response, "status_code")
            if status_code == HTTPStatus.OK:
                return self._response_json(response)
            if self._is_retryable_status(status_code) and attempt < self.max_retries:
                self._backoff(attempt)
                continue
            raise EmbeddingAPIError(
                self._api_error_message(response, status_code),
                status_code=status_code if isinstance(status_code, int) else None,
            )

        raise AssertionError("retry loop exited unexpectedly")

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @staticmethod
    def _response_json(response: Any) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except (ValueError, TypeError, AttributeError) as error:
            raise EmbeddingResponseError(
                "DashScope embedding response is not valid JSON"
            ) from error
        if not isinstance(payload, Mapping):
            raise EmbeddingResponseError(
                "DashScope embedding response must be a JSON object"
            )
        return payload

    @staticmethod
    def _is_retryable_status(status_code: Any) -> bool:
        return isinstance(status_code, int) and (
            status_code == HTTPStatus.TOO_MANY_REQUESTS or status_code >= 500
        )

    def _api_error_message(self, response: Any, status_code: Any) -> str:
        try:
            payload = self._response_json(response)
        except EmbeddingResponseError:
            payload = {}
        code = self._field(payload, "code")
        message = self._field(payload, "message")
        request_id = self._field(payload, "request_id")
        if request_id is None:
            request_id = getattr(response, "headers", {}).get("x-request-id")
        return (
            "DashScope embedding request failed: "
            f"status_code={status_code!r}, code={code!r}, "
            f"message={message!r}, request_id={request_id!r}"
        )

    def _backoff(self, attempt: int) -> None:
        self._sleep(self.backoff_factor * (2**attempt))
