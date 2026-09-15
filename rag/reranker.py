"""Fixed-pool DashScope reranking over the frozen RRF candidate generation.

Candidate generation intentionally remains outside this module's control:
Dense Top20 + BM25 Top20, retrieval-unit normalization/deduplication, then
RRF k=60.  This module only sends the resulting units to
``qwen3.7-text-rerank`` and changes their final order.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections.abc import Collection
from http import HTTPStatus
from typing import Any, Callable, Mapping, Sequence, TypedDict

import requests

from rag.hybrid_retriever import HybridRetriever
from rag.retrieval_scope import log_scope_filter, normalize_allowed_doc_ids


DEFAULT_MODEL = "qwen3.7-text-rerank"
DEFAULT_BATCH_SIZE = 20
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5
CONNECT_TIMEOUT = 5.0
DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
    "text-rerank/text-rerank"
)
logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    """Base error for the reranker transport and response contract."""


class RerankerAPIError(RerankerError):
    """A non-success DashScope response, mapped with status/code/request ID."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RerankerResponseError(RerankerError):
    """A syntactically successful response that violates the expected schema."""


class RerankedResult(TypedDict):
    """Unchanged RRF candidate plus rerank provenance and final rank."""

    rank: int
    rerank_score: float
    rrf_score: float
    retrieval_unit_id: str
    dense_rank: int | None
    bm25_rank: int | None
    chunk_id: str
    doc_id: str
    file_name: str
    pages: list[int]
    chunk_type: str
    headings: list[str]
    text: str


def candidate_text(candidate: Mapping[str, Any]) -> str:
    """Build the only text sent for a candidate, retaining headings and text."""

    headings = candidate.get("headings")
    text = candidate.get("text")
    if not isinstance(headings, list) or any(not isinstance(x, str) for x in headings):
        raise RerankerResponseError("candidate headings must be a list[str]")
    if not isinstance(text, str) or not text.strip():
        raise RerankerResponseError("candidate text must be a non-empty string")
    heading_text = " > ".join(heading.strip() for heading in headings if heading.strip())
    return f"Headings: {heading_text}\n\nText: {text}" if heading_text else f"Text: {text}"


class DashScopeRerankerClient:
    """Synchronous native API adapter with stable batching, retry and timeout.

    DashScope scores documents relative to a query.  A request batch therefore
    consists only of documents for one query; callers should use a batch size
    at least as large as their candidate pool to keep all candidates together.
    The frozen first version uses 20 for both values, producing one call/query.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        request_post: Callable[..., Any] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not resolved_key:
            raise ValueError("DASHSCOPE_API_KEY is required for reranking")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor < 0:
            raise ValueError("backoff_factor must be non-negative")
        self.api_key, self.model, self.batch_size = resolved_key, model, batch_size
        self.timeout, self.max_retries = float(timeout), max_retries
        self.backoff_factor, self._request_post, self._sleep = float(backoff_factor), request_post, sleep

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of strings")
        docs = list(documents)
        if not docs:
            return []
        if any(not isinstance(doc, str) or not doc.strip() for doc in docs):
            raise ValueError("documents must contain only non-empty strings")

        scores: list[float] = []
        for start in range(0, len(docs), self.batch_size):
            scores.extend(self._rerank_batch(query, docs[start : start + self.batch_size]))
        if len(scores) != len(docs):
            raise RerankerResponseError("rerank score count does not match document count")
        return scores

    def _rerank_batch(self, query: str, documents: list[str]) -> list[float]:
        payload = self._call_with_retry(query, documents)
        output = self._field(payload, "output")
        results = self._field(output, "results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise RerankerResponseError("rerank response output.results has an invalid count")
        indexed: dict[int, float] = {}
        for position, item in enumerate(results):
            index, score = self._field(item, "index"), self._field(item, "relevance_score")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(documents):
                raise RerankerResponseError(f"rerank result {position} has invalid index")
            if index in indexed or not isinstance(score, (int, float)) or isinstance(score, bool):
                raise RerankerResponseError(f"rerank result {position} has invalid score/index")
            indexed[index] = float(score)
        if sorted(indexed) != list(range(len(documents))):
            raise RerankerResponseError("rerank response indices do not cover the input batch")
        return [indexed[index] for index in range(len(documents))]

    def _call_with_retry(self, query: str, documents: list[str]) -> Mapping[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request_post(
                    DASHSCOPE_RERANK_URL,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "input": {"query": query, "documents": documents}, "parameters": {"return_documents": False}},
                    timeout=(CONNECT_TIMEOUT, self.timeout),
                )
            except (requests.Timeout, requests.ConnectionError) as error:
                if attempt >= self.max_retries:
                    raise RerankerAPIError(f"DashScope rerank transport failure: {type(error).__name__}") from error
                self._backoff(attempt)
                continue
            status = self._field(response, "status_code")
            if status == HTTPStatus.OK:
                return self._response_json(response)
            if self._retryable(status) and attempt < self.max_retries:
                self._backoff(attempt)
                continue
            raise RerankerAPIError(self._api_error_message(response, status), status_code=status if isinstance(status, int) else None)
        raise AssertionError("retry loop exited unexpectedly")

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)

    @classmethod
    def _response_json(cls, response: Any) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except (ValueError, TypeError, AttributeError) as error:
            raise RerankerResponseError("DashScope rerank response is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise RerankerResponseError("DashScope rerank response must be a JSON object")
        return payload

    @staticmethod
    def _retryable(status: Any) -> bool:
        return isinstance(status, int) and (status == HTTPStatus.TOO_MANY_REQUESTS or status >= 500)

    def _api_error_message(self, response: Any, status: Any) -> str:
        try:
            payload = self._response_json(response)
        except RerankerResponseError:
            payload = {}
        request_id = self._field(payload, "request_id") or getattr(response, "headers", {}).get("x-request-id")
        return ("DashScope rerank request failed: "
                f"status_code={status!r}, code={self._field(payload, 'code')!r}, "
                f"message={self._field(payload, 'message')!r}, request_id={request_id!r}")

    def _backoff(self, attempt: int) -> None:
        self._sleep(self.backoff_factor * (2 ** attempt))


class RerankerRetriever:
    """Return Final Top-K after reranking the frozen RRF Top-20 candidate units."""

    def __init__(self, hybrid_retriever: HybridRetriever, *, reranker_client: DashScopeRerankerClient | Any | None = None) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker_client = reranker_client or DashScopeRerankerClient()

    def retrieve(
        self,
        query: str,
        *,
        candidate_k: int = 20,
        final_k: int = 5,
        allowed_doc_ids: Collection[str] | None = None,
    ) -> list[RerankedResult]:
        if candidate_k != DEFAULT_BATCH_SIZE:
            raise ValueError("the frozen first version requires candidate_k=20")
        if final_k != 5:
            raise ValueError("the frozen first version requires final_k=5")
        allowed = normalize_allowed_doc_ids(allowed_doc_ids)
        if allowed is None:
            candidates = self.hybrid_retriever.retrieve(query, top_k=candidate_k, candidate_k=20)
            return self.rerank_candidates(query, candidates, final_k=final_k)
        candidates = self.hybrid_retriever.retrieve(
            query, top_k=candidate_k, candidate_k=20, allowed_doc_ids=allowed
        )
        return self.rerank_candidates(
            query, candidates, final_k=final_k, allowed_doc_ids=allowed
        )

    def rerank_candidates(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        *,
        final_k: int = 5,
        allowed_doc_ids: Collection[str] | None = None,
    ) -> list[RerankedResult]:
        """Rerank a pre-generated frozen pool; used to avoid duplicate retrieval in eval."""

        if final_k != 5:
            raise ValueError("the frozen first version requires final_k=5")
        allowed = normalize_allowed_doc_ids(allowed_doc_ids)
        scoped_candidates = list(candidates)
        if allowed is not None:
            scoped_candidates = [
                candidate for candidate in scoped_candidates
                if candidate.get("doc_id") in allowed
            ]
            log_scope_filter(
                logger,
                stage="reranker_input",
                allowed_doc_ids=allowed,
                candidate_count_before_scope=len(candidates),
                candidate_count_after_scope=len(scoped_candidates),
            )
        if not scoped_candidates or len(scoped_candidates) > DEFAULT_BATCH_SIZE:
            raise ValueError("candidate pool must contain 1 through 20 retrieval units")
        texts = [candidate_text(candidate) for candidate in scoped_candidates]
        scores = self.reranker_client.rerank(query, texts)
        ranked = sorted(
            enumerate(zip(scoped_candidates, scores)),
            key=lambda pair: (-pair[1][1], pair[0]),
        )
        output: list[RerankedResult] = []
        for rank, (_, (candidate, score)) in enumerate(ranked[:final_k], 1):
            result = copy.deepcopy(dict(candidate))
            result["rank"] = rank
            result["rerank_score"] = score
            output.append(result)  # type: ignore[arg-type]
        return output
