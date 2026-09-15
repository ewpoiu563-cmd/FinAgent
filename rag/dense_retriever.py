"""Dense retrieval over a persisted FinAgent FAISS index."""

from __future__ import annotations

import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from rag.embedding import EmbeddingClient
from rag.retrieval_scope import log_scope_filter, normalize_allowed_doc_ids
from rag.vector_store import FaissVectorStore


logger = logging.getLogger(__name__)


class DenseRetrievalResult(TypedDict):
    """Public result schema returned by :class:`DenseRetriever`."""

    rank: int
    score: float
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


_RESULT_FIELDS = (
    "chunk_id",
    "doc_id",
    "file_name",
    "pages",
    "chunk_type",
    "headings",
    "text",
)
_OPTIONAL_RESULT_FIELDS = (
    "parent_chunk_id",
    "fallback_split",
    "part_index",
    "part_count",
)


class DenseRetriever:
    """Embed queries and return the original FAISS Top-K chunks.

    The persisted index is the source of truth for the embedding model and
    dimension.  Passing an embedding client with different settings is rejected
    instead of allowing environment configuration to create a silent mismatch.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.vector_store = FaissVectorStore.load(index_dir)
        self.embedding_client = embedding_client or EmbeddingClient(
            model=self.vector_store.embedding_model,
            dimension=self.vector_store.dimension,
        )
        self._validate_embedding_configuration()

    def _validate_embedding_configuration(self) -> None:
        client_model = getattr(self.embedding_client, "model", None)
        client_dimension = getattr(self.embedding_client, "dimension", None)
        if client_model != self.vector_store.embedding_model:
            raise ValueError(
                "embedding model mismatch: "
                f"index={self.vector_store.embedding_model!r}, "
                f"query={client_model!r}"
            )
        if client_dimension != self.vector_store.dimension:
            raise ValueError(
                "embedding dimension mismatch: "
                f"index={self.vector_store.dimension}, query={client_dimension!r}"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_doc_ids: Collection[str] | None = None,
    ) -> list[DenseRetrievalResult]:
        """Return ranked FAISS results, optionally scoped by physical doc ID."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        allowed = normalize_allowed_doc_ids(allowed_doc_ids)

        embeddings = self.embedding_client.embed_texts([query], text_type="query")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            count: Any = len(embeddings) if isinstance(embeddings, list) else "invalid"
            raise ValueError(
                f"query embedding count mismatch: expected 1, got {count}"
            )

        query_vector = embeddings[0]
        if not isinstance(query_vector, list):
            raise TypeError("query embedding must be a list")
        if len(query_vector) != self.vector_store.dimension:
            raise ValueError(
                "query embedding dimension mismatch: "
                f"expected {self.vector_store.dimension}, got {len(query_vector)}"
            )

        # FaissVectorStore.search() owns query L2 normalization.  Passing the
        # embedding through unchanged avoids normalizing it twice here.
        if allowed is None:
            # Frozen unscoped path: retain the exact original search call.
            matches = self.vector_store.search(query_vector, top_k=top_k)
        else:
            # IndexFlatIP has no metadata predicate. Embed once, progressively
            # enlarge the exact-FAISS prefix, and filter before returning any
            # candidates to Hybrid/RRF.
            search_k = min(self.vector_store.vector_count, max(top_k, top_k * 4))
            while True:
                prefix = self.vector_store.search(query_vector, top_k=search_k)
                matches = [
                    match for match in prefix
                    if match["chunk"].get("doc_id") in allowed
                ]
                if len(matches) >= top_k or search_k >= self.vector_store.vector_count:
                    break
                search_k = min(self.vector_store.vector_count, search_k * 2)
            log_scope_filter(
                logger,
                stage="dense",
                allowed_doc_ids=allowed,
                candidate_count_before_scope=len(prefix),
                candidate_count_after_scope=len(matches),
            )
            matches = matches[:top_k]
        results: list[DenseRetrievalResult] = []
        for rank, match in enumerate(matches, start=1):
            chunk = match["chunk"]
            result: dict[str, Any] = {
                "rank": rank,
                "score": float(match["score"]),
                **{field: chunk[field] for field in _RESULT_FIELDS},
            }
            result.update(
                {
                    field: chunk[field]
                    for field in _OPTIONAL_RESULT_FIELDS
                    if field in chunk
                }
            )
            results.append(result)  # type: ignore[arg-type]
        return results
