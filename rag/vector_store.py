"""Local FAISS vector store for FinAgent unified chunks.

Document and query vectors are L2-normalized before use.  Therefore the inner
product returned by ``faiss.IndexFlatIP`` is cosine similarity.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

import faiss
import numpy as np

from rag.embedding import DEFAULT_DIMENSION, DEFAULT_MODEL


INDEX_FILE_NAME = "financial.faiss"
METADATA_FILE_NAME = "metadata.jsonl"
MANIFEST_FILE_NAME = "manifest.json"
SCHEMA_VERSION = "1.0"
INDEX_TYPE = "IndexFlatIP"
NORMALIZATION = "l2"
REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "doc_id",
    "file_name",
    "pages",
    "chunk_type",
    "headings",
    "text",
)


class SearchResult(TypedDict):
    score: float
    chunk: dict[str, Any]


class FaissVectorStore:
    """Small, exact cosine-similarity store backed by ``IndexFlatIP``.

    Position ``i`` in the FAISS index always maps to ``self._chunks[i]`` and to
    the metadata JSONL record whose explicit ``position`` is ``i``.
    """

    def __init__(
        self,
        *,
        dimension: int = DEFAULT_DIMENSION,
        embedding_model: str = DEFAULT_MODEL,
    ) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError("embedding_model must be a non-empty string")

        self.dimension = dimension
        self.embedding_model = embedding_model
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[dict[str, Any]] = []
        self._chunk_ids: set[str] = set()
        self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def vector_count(self) -> int:
        return int(self._index.ntotal)

    def add(
        self,
        vectors: Sequence[Sequence[float]] | np.ndarray,
        chunks: Sequence[Mapping[str, Any]],
    ) -> None:
        """Validate, normalize, and append vectors and aligned chunk metadata."""

        if len(vectors) == 0 or len(chunks) == 0:
            raise ValueError("vectors and chunks must not be empty")
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vector count must equal chunk count: {len(vectors)} != {len(chunks)}"
            )

        matrix = self._as_matrix(vectors, name="vectors")
        normalized_chunks = [self._validate_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        new_ids = [chunk["chunk_id"] for chunk in normalized_chunks]
        duplicate_ids = {chunk_id for chunk_id in new_ids if new_ids.count(chunk_id) > 1}
        duplicate_ids.update(self._chunk_ids.intersection(new_ids))
        if duplicate_ids:
            duplicate = sorted(duplicate_ids)[0]
            raise ValueError(f"duplicate chunk_id: {duplicate}")

        self._normalize(matrix, name="vectors")
        self._index.add(matrix)
        self._chunks.extend(normalized_chunks)
        self._chunk_ids.update(new_ids)

    def search(
        self,
        query_vector: Sequence[float] | np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the nearest chunks ranked by cosine similarity."""

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.vector_count == 0:
            raise ValueError("cannot search an empty index")

        query = self._as_matrix(query_vector, name="query_vector", single_vector=True)
        self._normalize(query, name="query_vector")
        result_count = min(top_k, self.vector_count)
        scores, positions = self._index.search(query, result_count)
        return [
            {
                "score": float(score),
                "chunk": copy.deepcopy(self._chunks[int(position)]),
            }
            for score, position in zip(scores[0], positions[0])
        ]

    def save(
        self,
        directory: str | Path,
        *,
        ingestion_stats: Mapping[str, int] | None = None,
    ) -> None:
        """Persist the index, aligned metadata, and validation manifest."""

        if self.vector_count != len(self._chunks):
            raise RuntimeError("FAISS vector count and metadata count are not aligned")

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(destination / INDEX_FILE_NAME))

        with (destination / METADATA_FILE_NAME).open(
            "w", encoding="utf-8", newline="\n"
        ) as output:
            for position, chunk in enumerate(self._chunks):
                record = {"position": position, "chunk": chunk}
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "index_type": INDEX_TYPE,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "normalization": NORMALIZATION,
            "vector_count": self.vector_count,
            "created_at": self.created_at,
        }
        if ingestion_stats is not None:
            manifest.update(dict(ingestion_stats))
        with (destination / MANIFEST_FILE_NAME).open(
            "w", encoding="utf-8", newline="\n"
        ) as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        """Load a store and reject inconsistent or unsupported persisted data."""

        source = Path(directory)
        with (source / MANIFEST_FILE_NAME).open(encoding="utf-8") as input_file:
            manifest = json.load(input_file)

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {manifest.get('schema_version')!r}")
        if manifest.get("index_type") != INDEX_TYPE:
            raise ValueError(f"unsupported index_type: {manifest.get('index_type')!r}")
        if manifest.get("normalization") != NORMALIZATION:
            raise ValueError(f"unsupported normalization: {manifest.get('normalization')!r}")

        dimension = manifest.get("dimension")
        embedding_model = manifest.get("embedding_model")
        store = cls(dimension=dimension, embedding_model=embedding_model)
        index = faiss.read_index(str(source / INDEX_FILE_NAME))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError(f"persisted FAISS index is not {INDEX_TYPE}")
        if index.d != dimension:
            raise ValueError(
                f"persisted index dimension mismatch: manifest={dimension}, index={index.d}"
            )

        chunks: list[dict[str, Any]] = []
        with (source / METADATA_FILE_NAME).open(encoding="utf-8") as input_file:
            for expected_position, line in enumerate(input_file):
                record = json.loads(line)
                if record.get("position") != expected_position:
                    raise ValueError(
                        "metadata position mismatch: "
                        f"expected {expected_position}, got {record.get('position')!r}"
                    )
                chunks.append(store._validate_chunk(record.get("chunk"), expected_position))

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("metadata contains duplicate chunk_id values")
        expected_count = manifest.get("vector_count")
        if not isinstance(expected_count, int) or isinstance(expected_count, bool):
            raise ValueError("manifest vector_count must be an integer")
        if index.ntotal != expected_count or len(chunks) != expected_count:
            raise ValueError(
                "persisted vector count mismatch: "
                f"manifest={expected_count}, index={index.ntotal}, metadata={len(chunks)}"
            )

        store._index = index
        store._chunks = chunks
        store._chunk_ids = set(chunk_ids)
        store.created_at = manifest.get("created_at")
        return store

    def _as_matrix(
        self,
        values: Sequence[Any] | np.ndarray,
        *,
        name: str,
        single_vector: bool = False,
    ) -> np.ndarray:
        try:
            matrix = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric values") from error
        if single_vector:
            if matrix.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional vector")
            matrix = matrix.reshape(1, -1)
        elif matrix.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"{name} dimension mismatch: expected {self.dimension}, got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinity")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    @staticmethod
    def _normalize(matrix: np.ndarray, *, name: str) -> None:
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError(f"{name} contains a zero-norm vector")
        faiss.normalize_L2(matrix)

    @staticmethod
    def _validate_chunk(chunk: Any, position: int) -> dict[str, Any]:
        if not isinstance(chunk, Mapping):
            raise TypeError(f"chunks[{position}] must be a mapping")
        missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
        if missing:
            raise ValueError(f"chunks[{position}] is missing required fields: {missing}")
        chunk_id = chunk["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"chunks[{position}].chunk_id must be a non-empty string")
        normalized = copy.deepcopy(dict(chunk))
        try:
            json.dumps(normalized, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"chunks[{position}] must be JSON serializable") from error
        return normalized
