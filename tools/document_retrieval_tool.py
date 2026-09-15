"""High-level evidence retrieval tool over the frozen Financial RAG pipeline.

This module is deliberately orchestration-only. Candidate generation remains
owned by :class:`HybridRetriever`, and reranking remains owned by
:class:`RerankerRetriever`. No answer-generating LLM is called here.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Collection
from pathlib import Path
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import requests
from dotenv import load_dotenv

from rag.embedding import EmbeddingError
from rag.hybrid_retriever import HybridRetriever
from rag.retrieval_scope import log_scope_filter, normalize_allowed_doc_ids
from rag.reranker import RerankerRetriever
from rag.retriever import QueryEmbeddingError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs" / "index" / "multi_doc_v2"
CANDIDATE_K = 20
FINAL_K = 5
DEFAULT_CANDIDATE_TIMEOUT = 45.0
DEFAULT_RERANK_TIMEOUT = 45.0


def _call_with_timeout(call: Callable[[], Any], timeout: float) -> Any:
    """Run a bounded synchronous stage without waiting for it after timeout.

    The underlying HTTP clients also have their own request timeouts. The
    daemon thread supplies a hard tool-level response deadline across retries.
    """

    outcome: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def run() -> None:
        try:
            outcome.put((True, call()))
        except Exception as error:  # propagate the original stage error
            outcome.put((False, error))

    worker = Thread(target=run, name="retrieve-document-stage", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"retrieval stage exceeded {timeout:g}s timeout")
    succeeded, value = outcome.get_nowait()
    if succeeded:
        return value
    if not isinstance(value, Exception):
        raise RuntimeError("retrieval stage returned an invalid error")
    raise value


def _exception_chain(error: BaseException):
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_timeout(error: BaseException) -> bool:
    return any(
        isinstance(item, (TimeoutError, requests.Timeout))
        or "timeout" in str(item).casefold()
        for item in _exception_chain(error)
    )


def _is_embedding_error(error: BaseException) -> bool:
    return any(
        isinstance(item, (EmbeddingError, QueryEmbeddingError))
        or "embedding" in str(item).casefold()
        for item in _exception_chain(error)
    )


def _evidence(candidate: Mapping[str, Any], rank: int) -> dict[str, Any]:
    pages = copy.deepcopy(candidate.get("pages", []))
    item: dict[str, Any] = {
        "rank": rank,
        "doc_id": candidate.get("doc_id"),
        "page": pages,
        "headings": copy.deepcopy(candidate.get("headings", [])),
        "text": candidate.get("text"),
        "retrieval_unit_id": candidate.get("retrieval_unit_id"),
        "rerank_score": candidate.get("rerank_score"),
    }
    source_file = candidate.get("source_file", candidate.get("file_name"))
    if source_file is not None:
        item["source_file"] = source_file
    return item


class DocumentRetrievalTool:
    """Retrieve evidence only when the frozen hybrid-plus-reranker chain completes."""

    def __init__(
        self,
        *,
        hybrid_retriever: Any | None = None,
        reranker_retriever: Any | None = None,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        candidate_timeout: float = DEFAULT_CANDIDATE_TIMEOUT,
        rerank_timeout: float = DEFAULT_RERANK_TIMEOUT,
        timeout_runner: Callable[[Callable[[], Any], float], Any] = _call_with_timeout,
    ) -> None:
        self._hybrid = hybrid_retriever
        self._reranker = reranker_retriever
        self.index_dir = Path(index_dir)
        self.candidate_timeout = float(candidate_timeout)
        self.rerank_timeout = float(rerank_timeout)
        self._timeout_runner = timeout_runner

    def _candidate_generation(
        self,
        query: str,
        allowed_doc_ids: Collection[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        if self._hybrid is None:
            self._hybrid = HybridRetriever(self.index_dir)
        if allowed_doc_ids is not None:
            return self._hybrid.retrieve(
                query,
                top_k=CANDIDATE_K,
                candidate_k=CANDIDATE_K,
                allowed_doc_ids=allowed_doc_ids,
            )
        return self._hybrid.retrieve(
            query,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

    def _rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        allowed_doc_ids: Collection[str] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        if self._reranker is None:
            self._reranker = RerankerRetriever(self._hybrid)
        if allowed_doc_ids is not None:
            return self._reranker.rerank_candidates(
                query,
                candidates,
                final_k=FINAL_K,
                allowed_doc_ids=allowed_doc_ids,
            )
        return self._reranker.rerank_candidates(query, candidates, final_k=FINAL_K)

    def retrieve(
        self,
        query: str,
        allowed_doc_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        result: dict[str, Any] = {
            "success": False,
            "query": query,
            "evidence": [],
            "retrieval_metadata": {
                "candidate_count": 0,
                "returned_count": 0,
                "reranker_used": False,
            },
            "degraded": False,
            "fallback": None,
            "error_type": None,
            "error": None,
            "latency_ms": 0.0,
        }

        try:
            if not isinstance(query, str) or not query.strip():
                result.update(
                    error_type="invalid_query",
                    error="query must be a non-empty string",
                )
                return result

            allowed = normalize_allowed_doc_ids(allowed_doc_ids)
            if allowed is not None:
                result["allowed_doc_ids"] = sorted(allowed)
                result["retrieval_metadata"].update(
                    scope_applied=True,
                    candidate_count_before_scope=0,
                    candidate_count_after_scope=0,
                )

            try:
                candidates = list(
                    self._timeout_runner(
                        lambda: self._candidate_generation(query, allowed),
                        self.candidate_timeout,
                    )
                )
            except Exception as error:
                error_type = (
                    "timeout"
                    if _is_timeout(error)
                    else "embedding_error"
                    if _is_embedding_error(error)
                    else "retrieval_error"
                )
                result.update(error_type=error_type, error=str(error))
                return result

            if allowed is not None:
                before_scope = len(candidates)
                candidates = [
                    candidate for candidate in candidates
                    if candidate.get("doc_id") in allowed
                ]
                result["retrieval_metadata"].update(
                    candidate_count_before_scope=before_scope,
                    candidate_count_after_scope=len(candidates),
                )
                log_scope_filter(
                    logger,
                    stage="document_tool_candidates",
                    allowed_doc_ids=allowed,
                    candidate_count_before_scope=before_scope,
                    candidate_count_after_scope=len(candidates),
                )
            result["retrieval_metadata"]["candidate_count"] = len(candidates)
            if not candidates:
                result.update(
                    error_type="retrieval_error",
                    error="candidate retrieval returned no evidence",
                )
                return result

            try:
                selected = list(
                    self._timeout_runner(
                        lambda: self._rerank(query, candidates, allowed),
                        self.rerank_timeout,
                    )
                )[:FINAL_K]
                result["retrieval_metadata"]["reranker_used"] = True
            except Exception as error:
                result.update(
                    success=False,
                    error_type="timeout" if _is_timeout(error) else "rerank_error",
                    error=str(error),
                )
                return result

            if allowed is not None:
                before_output_scope = len(selected)
                selected = [
                    candidate for candidate in selected
                    if candidate.get("doc_id") in allowed
                ]
                log_scope_filter(
                    logger,
                    stage="document_tool_output",
                    allowed_doc_ids=allowed,
                    candidate_count_before_scope=before_output_scope,
                    candidate_count_after_scope=len(selected),
                )

            evidence = [
                _evidence(candidate, rank)
                for rank, candidate in enumerate(selected, 1)
            ]
            result.update(success=True, evidence=evidence)
            result["retrieval_metadata"]["returned_count"] = len(evidence)
            return result
        finally:
            result["latency_ms"] = round((perf_counter() - started) * 1000, 4)


_DEFAULT_TOOL = DocumentRetrievalTool()


def retrieve_document(
    query: str,
    allowed_doc_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Return evidence only for one query; final answer generation is external."""

    if allowed_doc_ids is None:
        return _DEFAULT_TOOL.retrieve(query)
    return _DEFAULT_TOOL.retrieve(query, allowed_doc_ids=allowed_doc_ids)
