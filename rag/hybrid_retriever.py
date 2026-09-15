"""Reciprocal-rank fusion over the existing Dense and BM25 retrievers."""

from __future__ import annotations

import copy
import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any, Mapping, NotRequired, Protocol, Sequence, TypedDict

from rag.bm25_retriever import BM25Retriever
from rag.dense_retriever import DenseRetriever
from rag.retrieval_scope import log_scope_filter, normalize_allowed_doc_ids


RRF_K = 60
logger = logging.getLogger(__name__)


class _Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[Mapping[str, Any]]: ...


class HybridRetrievalResult(TypedDict):
    rank: int
    rrf_score: float
    retrieval_unit_id: str
    dense_rank: int | None
    bm25_rank: int | None
    contributing_retrievers: list[str]
    chunk_id: str
    doc_id: str
    file_name: str
    pages: list[int]
    chunk_type: str
    headings: list[str]
    text: str
    parent_chunk_id: NotRequired[str]
    fallback_split: NotRequired[bool]
    part_index: NotRequired[int]
    part_count: NotRequired[int]


def retrieval_unit_id(result: Mapping[str, Any]) -> str:
    """Normalize a physical chunk to its source retrieval-unit boundary."""

    unit_id = result.get("parent_chunk_id") or result["chunk_id"]
    if not isinstance(unit_id, str) or not unit_id.strip():
        raise ValueError("retrieved result has no valid retrieval_unit_id")
    return unit_id


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class HybridRetriever:
    """Fuse source-normalized Dense and BM25 candidate ranks with RRF.

    The two raw score spaces are intentionally ignored. Within each retriever,
    sibling fallback chunks normalize to one unit and only the best unit rank
    contributes. ``RRF_K`` is fixed at 60.
    """

    def __init__(
        self,
        index_dir: str | Path | None = None,
        *,
        dense_retriever: _Retriever | None = None,
        bm25_retriever: _Retriever | None = None,
    ) -> None:
        if dense_retriever is None or bm25_retriever is None:
            if index_dir is None:
                raise ValueError(
                    "index_dir is required unless both retrievers are provided"
                )
        self.dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else DenseRetriever(index_dir)  # type: ignore[arg-type]
        )
        self.bm25_retriever = (
            bm25_retriever
            if bm25_retriever is not None
            else BM25Retriever(index_dir)  # type: ignore[arg-type]
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        allowed_doc_ids: Collection[str] | None = None,
    ) -> list[HybridRetrievalResult]:
        """Retrieve candidates from both backends and return fused Top-K units."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        _positive_integer(top_k, "top_k")
        _positive_integer(candidate_k, "candidate_k")
        if top_k > candidate_k:
            raise ValueError("top_k must not exceed candidate_k")
        allowed = normalize_allowed_doc_ids(allowed_doc_ids)

        # Deliberately let either backend exception propagate to the caller.
        if allowed is None:
            # Frozen unscoped calls intentionally omit the new keyword.
            dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
            bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)
        else:
            dense_raw = self.dense_retriever.retrieve(
                query, top_k=candidate_k, allowed_doc_ids=allowed
            )
            bm25_raw = self.bm25_retriever.retrieve(
                query, top_k=candidate_k, allowed_doc_ids=allowed
            )
            before_scope = len(dense_raw) + len(bm25_raw)
            dense_results = [item for item in dense_raw if item.get("doc_id") in allowed]
            bm25_results = [item for item in bm25_raw if item.get("doc_id") in allowed]
            log_scope_filter(
                logger,
                stage="hybrid",
                allowed_doc_ids=allowed,
                candidate_count_before_scope=before_scope,
                candidate_count_after_scope=len(dense_results) + len(bm25_results),
            )
        return self.fuse(dense_results, bm25_results, top_k=top_k)

    @staticmethod
    def fuse(
        dense_results: Sequence[Mapping[str, Any]],
        bm25_results: Sequence[Mapping[str, Any]],
        *,
        top_k: int = 5,
    ) -> list[HybridRetrievalResult]:
        """Fuse already-ranked candidate lists, primarily for evaluation/testing."""

        _positive_integer(top_k, "top_k")
        units: dict[str, dict[str, Any]] = {}

        for retriever_name, results in (
            ("dense", dense_results),
            ("bm25", bm25_results),
        ):
            for position, result in enumerate(results, start=1):
                if not isinstance(result, Mapping):
                    raise TypeError(f"{retriever_name} result {position} must be a mapping")
                rank = result.get("rank")
                if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                    raise ValueError(
                        f"{retriever_name} result {position} has invalid rank"
                    )
                unit_id = retrieval_unit_id(result)
                unit = units.setdefault(
                    unit_id,
                    {
                        "dense_rank": None,
                        "bm25_rank": None,
                        "representatives": [],
                    },
                )
                rank_field = f"{retriever_name}_rank"
                previous_rank = unit[rank_field]
                if previous_rank is None or rank < previous_rank:
                    unit[rank_field] = rank
                unit["representatives"].append(
                    (
                        rank,
                        0 if retriever_name == "dense" else 1,
                        str(result.get("chunk_id", "")),
                        result,
                    )
                )

        fused: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for unit_id, unit in units.items():
            dense_rank = unit["dense_rank"]
            bm25_rank = unit["bm25_rank"]
            ranks = [rank for rank in (dense_rank, bm25_rank) if rank is not None]
            score = sum(1.0 / (RRF_K + rank) for rank in ranks)
            representative = min(unit["representatives"], key=lambda item: item[:3])[3]
            result = copy.deepcopy(dict(representative))
            result.pop("rank", None)
            result.pop("score", None)
            result.update(
                {
                    "rrf_score": score,
                    "retrieval_unit_id": unit_id,
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                    "contributing_retrievers": [
                        name
                        for name, rank in (("dense", dense_rank), ("bm25", bm25_rank))
                        if rank is not None
                    ],
                }
            )
            best_rank = min(ranks)
            rank_sum = sum(ranks)
            sort_key = (
                -score,
                best_rank,
                rank_sum,
                unit_id,
            )
            fused.append((sort_key, result))

        fused.sort(key=lambda item: item[0])
        output: list[HybridRetrievalResult] = []
        for rank, (_, result) in enumerate(fused[:top_k], start=1):
            result["rank"] = rank
            output.append(result)  # type: ignore[arg-type]
        return output
