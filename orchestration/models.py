"""Typed planning models shared by the source-aware control layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .web_source_policy import WebSourcePolicy


class SourceType(str, Enum):
    SQL = "sql"
    RAG = "rag"
    WEB = "web"
    DIRECT = "direct"
    LOCAL_COMPUTE = "local_compute"
    MODEL_PRIOR = "model_prior"


class PlanMode(str, Enum):
    SINGLE_SOURCE = "single_source"
    MULTI_SOURCE = "multi_source"
    AUTO = "auto"


@dataclass(frozen=True)
class EntityResolution:
    """Business-level entity match; physical index IDs never appear here."""

    company_name: str | None = None
    matched_alias: str | None = None
    catalog_doc_id: str | None = None
    document_type: str | None = None

    @property
    def matched(self) -> bool:
        return self.catalog_doc_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_name": self.company_name,
            "matched_alias": self.matched_alias,
            "catalog_doc_id": self.catalog_doc_id,
            "document_type": self.document_type,
        }


@dataclass(frozen=True)
class RouterDecision:
    """Validated source-only decision returned by the lightweight LLM router."""

    primary_source: SourceType
    required_sources: tuple[SourceType, ...]
    web_fallback_allowed: bool = True
    reason: str = ""


@dataclass(frozen=True)
class SourcePlan:
    """Declarative source plan; execution and index scoping happen later."""

    mode: PlanMode
    primary_source: SourceType | None
    required_sources: tuple[SourceType, ...]
    company_name: str | None = None
    catalog_doc_ids: tuple[str, ...] = ()
    document_type: str | None = None
    web_fallback_allowed: bool = True
    web_source_policy: WebSourcePolicy = WebSourcePolicy()
    routing_method: str = "rules"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "primary_source": self.primary_source.value if self.primary_source else None,
            "required_sources": [source.value for source in self.required_sources],
            "company_name": self.company_name,
            "catalog_doc_ids": list(self.catalog_doc_ids),
            "document_type": self.document_type,
            "web_fallback_allowed": self.web_fallback_allowed,
            "web_source_policy": self.web_source_policy.to_dict(),
            "routing_method": self.routing_method,
            "reason": self.reason,
        }
