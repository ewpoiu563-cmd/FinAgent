"""Dense retrieval over a persisted FinAgent FAISS index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from rag.embedding import EmbeddingClient
from rag.vector_store import FaissVectorStore


class DenseRetrieverError(RuntimeError):
    """Base error raised by dense retrieval orchestration."""


class IndexLoadError(DenseRetrieverError):
    """Raised when a persisted vector index cannot be loaded."""


class QueryEmbeddingError(DenseRetrieverError):
    """Raised when the query cannot be embedded into one valid vector."""


class DenseRetrievalResult(TypedDict):
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    file_name: str
    pages: list[int]
    chunk_type: str
    headings: list[str]
    text: str


class DenseRetriever:
    """Embed a query and retrieve cosine-ranked unified chunks.

    When no client is injected, the client is configured from the persisted
    index manifest so query and document embeddings use the same model and
    dimension.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        source = Path(index_dir)
        if not source.is_dir():
            raise IndexLoadError(f"index directory does not exist: {source}")
        try:
            self.vector_store = FaissVectorStore.load(source)
        except Exception as error:
            raise IndexLoadError(f"failed to load index from {source}: {error}") from error

        self.embedding_client = embedding_client or EmbeddingClient(
            model=self.vector_store.embedding_model,
            dimension=self.vector_store.dimension,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return ranked chunks using query-mode embeddings and cosine search."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.vector_store.vector_count == 0:
            raise DenseRetrieverError("cannot retrieve from an empty index")

        try:
            embeddings = self.embedding_client.embed_texts(
                [query],
                text_type="query",
            )
        except Exception as error:
            raise QueryEmbeddingError(f"failed to embed query: {error}") from error

        if not isinstance(embeddings, list) or len(embeddings) != 1:
            count = len(embeddings) if isinstance(embeddings, list) else "invalid"
            raise QueryEmbeddingError(
                f"query embedding count mismatch: expected 1, got {count}"
            )
        query_vector = embeddings[0]
        if not isinstance(query_vector, list):
            raise QueryEmbeddingError("query embedding must be a list")
        if len(query_vector) != self.vector_store.dimension:
            raise QueryEmbeddingError(
                "query embedding dimension mismatch: "
                f"expected {self.vector_store.dimension}, got {len(query_vector)}"
            )

        matches = self.vector_store.search(query_vector, top_k=top_k)
        results: list[dict[str, Any]] = []
        for rank, match in enumerate(matches, start=1):
            result = dict(match["chunk"])
            result.update(rank=rank, score=float(match["score"]))
            results.append(result)
        return results
