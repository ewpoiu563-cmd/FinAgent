"""Physical document-scope helpers shared by RAG retrieval stages."""

from __future__ import annotations

import json
import logging
from collections.abc import Collection


def normalize_allowed_doc_ids(allowed_doc_ids: Collection[str] | None) -> frozenset[str] | None:
    if allowed_doc_ids is None:
        return None
    if isinstance(allowed_doc_ids, (str, bytes)) or not isinstance(allowed_doc_ids, Collection):
        raise TypeError("allowed_doc_ids must be a collection of physical index doc IDs")
    values = tuple(allowed_doc_ids)
    if not values:
        raise ValueError("allowed_doc_ids must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError("allowed_doc_ids must contain non-empty strings")
    return frozenset(values)


def log_scope_filter(
    logger: logging.Logger,
    *,
    stage: str,
    allowed_doc_ids: Collection[str],
    candidate_count_before_scope: int,
    candidate_count_after_scope: int,
) -> None:
    logger.debug(
        "SOURCE_SCOPE_FILTER: %s",
        json.dumps(
            {
                "stage": stage,
                "allowed_index_doc_ids": sorted(allowed_doc_ids),
                "candidate_count_before_scope": candidate_count_before_scope,
                "candidate_count_after_scope": candidate_count_after_scope,
            },
            ensure_ascii=False,
        ),
    )
