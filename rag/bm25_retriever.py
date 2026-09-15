"""Local BM25 retrieval over persisted FinAgent FAISS chunk metadata."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Collection
from pathlib import Path
import re
from typing import Any, NotRequired, TypedDict

import jieba
from rank_bm25 import BM25Okapi

from rag.retrieval_scope import log_scope_filter, normalize_allowed_doc_ids


METADATA_FILE_NAME = "metadata.jsonl"
_REQUIRED_CHUNK_FIELDS = (
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

# Match non-Chinese business/financial entities before sending Chinese spans to
# jieba.  This keeps HA/DNA-like names atomic and preserves values such as
# 2015, 50,890.52 and 94.80% as single tokens.
_LEXICAL_SPAN_RE = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|\d+(?:\.\d+)?%?"
    r"|[A-Za-z]+(?:[._+-][A-Za-z0-9]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
)

jieba.setLogLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class BM25RetrievalResult(TypedDict):
    """Public result schema aligned with ``DenseRetrievalResult``."""

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


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and financial text for local BM25.

    Latin tokens are case-folded for matching while remaining atomic. Chinese
    spans are segmented with jieba; punctuation and whitespace are discarded.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens: list[str] = []
    for match in _LEXICAL_SPAN_RE.finditer(text):
        span = match.group(0)
        if "\u3400" <= span[0] <= "\u9fff":
            tokens.extend(token.strip() for token in jieba.lcut(span) if token.strip())
        else:
            tokens.append(span.casefold())
    return tokens


class BM25Retriever:
    """Build an in-memory BM25Okapi index from persisted chunk metadata."""

    def __init__(self, index_dir: str | Path) -> None:
        try:
            self.index_dir = Path(index_dir)
        except TypeError as error:
            raise TypeError("index_dir must be a path-like value") from error

        self.metadata_path = self.index_dir / METADATA_FILE_NAME
        self._chunks = self._load_metadata(self.metadata_path)
        self._positions_by_doc_id: dict[str, list[int]] = {}
        for position, chunk in enumerate(self._chunks):
            self._positions_by_doc_id.setdefault(chunk["doc_id"], []).append(position)
        corpus = [
            tokenize("\n".join([*chunk["headings"], chunk["text"]]))
            for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(corpus)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @classmethod
    def _load_metadata(cls, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"BM25 metadata file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"BM25 metadata path is not a file: {path}")

        chunks: list[dict[str, Any]] = []
        chunk_ids: set[str] = set()
        try:
            with path.open(encoding="utf-8") as input_file:
                for expected_position, line in enumerate(input_file):
                    line_number = expected_position + 1
                    if not line.strip():
                        raise ValueError(
                            f"{path}:{line_number} must not be an empty metadata record"
                        )
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSON in BM25 metadata at {path}:{line_number}"
                        ) from error
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"BM25 metadata record {line_number} must be an object"
                        )
                    if record.get("position") != expected_position:
                        raise ValueError(
                            "BM25 metadata position mismatch: "
                            f"expected {expected_position}, "
                            f"got {record.get('position')!r}"
                        )
                    chunk = cls._validate_chunk(record.get("chunk"), line_number)
                    if chunk["chunk_id"] in chunk_ids:
                        raise ValueError(
                            f"BM25 metadata contains duplicate chunk_id: "
                            f"{chunk['chunk_id']}"
                        )
                    chunks.append(chunk)
                    chunk_ids.add(chunk["chunk_id"])
        except UnicodeDecodeError as error:
            raise ValueError(f"BM25 metadata is not valid UTF-8: {path}") from error

        if not chunks:
            raise ValueError(f"BM25 metadata contains no chunks: {path}")
        return chunks

    @staticmethod
    def _validate_chunk(chunk: Any, line_number: int) -> dict[str, Any]:
        label = f"BM25 metadata line {line_number} chunk"
        if not isinstance(chunk, dict):
            raise ValueError(f"{label} must be an object")
        missing = [field for field in _REQUIRED_CHUNK_FIELDS if field not in chunk]
        if missing:
            raise ValueError(f"{label} is missing required fields: {missing}")

        for field in ("chunk_id", "doc_id", "file_name", "chunk_type", "text"):
            if not isinstance(chunk[field], str) or not chunk[field].strip():
                raise ValueError(f"{label}.{field} must be a non-empty string")
        if not isinstance(chunk["headings"], list) or any(
            not isinstance(heading, str) for heading in chunk["headings"]
        ):
            raise ValueError(f"{label}.headings must be a list of strings")
        if not isinstance(chunk["pages"], list) or any(
            not isinstance(page, int) or isinstance(page, bool) or page <= 0
            for page in chunk["pages"]
        ):
            raise ValueError(f"{label}.pages must be a list of positive integers")
        return copy.deepcopy(chunk)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_doc_ids: Collection[str] | None = None,
    ) -> list[BM25RetrievalResult]:
        """Return raw BM25 Top-K, optionally scoped before candidate ranking."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        allowed = normalize_allowed_doc_ids(allowed_doc_ids)

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query must contain at least one searchable token")
        scores = self._bm25.get_scores(query_tokens)
        if allowed is None:
            result_count = min(top_k, self.chunk_count)
            # Frozen unscoped path: keep original ordering and tie-breaking.
            positions = sorted(
                range(self.chunk_count), key=lambda position: (-scores[position], position)
            )[:result_count]
        else:
            eligible_positions = [
                position
                for index_doc_id in allowed
                for position in self._positions_by_doc_id.get(index_doc_id, [])
            ]
            positions = sorted(
                eligible_positions, key=lambda position: (-scores[position], position)
            )[: min(top_k, len(eligible_positions))]
            log_scope_filter(
                logger,
                stage="bm25",
                allowed_doc_ids=allowed,
                candidate_count_before_scope=self.chunk_count,
                candidate_count_after_scope=len(eligible_positions),
            )

        results: list[BM25RetrievalResult] = []
        for rank, position in enumerate(positions, start=1):
            chunk = self._chunks[position]
            result: dict[str, Any] = {
                "rank": rank,
                "score": float(scores[position]),
                **{field: copy.deepcopy(chunk[field]) for field in _REQUIRED_CHUNK_FIELDS},
            }
            result.update(
                {
                    field: copy.deepcopy(chunk[field])
                    for field in _OPTIONAL_RESULT_FIELDS
                    if field in chunk
                }
            )
            results.append(result)  # type: ignore[arg-type]
        return results
