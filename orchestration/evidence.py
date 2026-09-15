"""Unified evidence schema, RAG normalization, and pre-synthesis scope validation."""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .document_catalog import DocumentCatalog
from .financial_amounts import MoneyAmount, extract_money_amounts, extract_sql_money_amounts
from .models import SourceType
from .trace import notify_runtime_artifact


logger = logging.getLogger(__name__)
_TRACE_PREVIEW_CHARS = 200

_INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
    r"(?:system|developer)\s+(?:message|prompt)",
    r"(?:do|must)\s+not\s+answer\s+the\s+user",
    r"忽略(?:之前|以上|前面).{0,12}(?:指令|提示)",
    r"(?:系统|开发者)(?:消息|提示词|指令)",
)


def _source_tier(url: str) -> str:
    """Return a conservative provenance tier, never a truth verdict."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return "unknown"
    host = host.casefold().removeprefix("www.")
    if host.endswith((".gov", ".gov.cn", ".edu", ".edu.cn")):
        return "primary"
    if any(token in host for token in ("sec.gov", "who.int", "worldbank.org", "oecd.org")):
        return "primary"
    if host.endswith(("wikipedia.org", "baike.baidu.com")):
        return "reference"
    return "secondary" if host else "unknown"


def _published_at(result: Mapping[str, Any]) -> str | None:
    for key in ("published_at", "date", "publishedDate", "publication_date"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _sanitize_web_text(value: str) -> tuple[str, bool]:
    """Neutralize instruction-shaped page text before it enters an LLM prompt."""
    suspected = any(re.search(pattern, value, re.IGNORECASE) for pattern in _INJECTION_PATTERNS)
    if not suspected:
        return value, False
    sanitized = value
    for pattern in _INJECTION_PATTERNS:
        sanitized = re.sub(
            pattern,
            "[已移除疑似网页提示注入文本]",
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized, True


@dataclass(frozen=True)
class EvidenceItem:
    source_type: SourceType
    tool_name: str
    catalog_doc_id: str | None = None
    index_doc_id: str | None = None
    company_name: str | None = None
    source_file: str | None = None
    page: tuple[int, ...] = ()
    headings: tuple[str, ...] = ()
    retrieval_unit_id: str | None = None
    rerank_score: float | None = None
    text: str = ""
    url: str | None = None
    title: str | None = None
    snippet: str | None = None
    content: str | None = None
    source_tier: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    prompt_injection_suspected: bool = False
    generated_sql: str | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[Mapping[str, Any], ...] = ()
    monetary_values: tuple[MoneyAmount, ...] = ()
    row_count: int | None = None
    truncated: bool | None = None
    error_type: str | None = None
    latency_ms: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)
    verified: bool = True

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        output = {
            "source_type": self.source_type.value,
            "tool_name": self.tool_name,
            "catalog_doc_id": self.catalog_doc_id,
            "index_doc_id": self.index_doc_id,
            "company_name": self.company_name,
            "source_file": self.source_file,
            "page": list(self.page),
            "headings": list(self.headings),
            "retrieval_unit_id": self.retrieval_unit_id,
            "rerank_score": self.rerank_score,
            "text": self.text,
            "verified": self.verified,
            "monetary_values": [amount.to_dict() for amount in self.monetary_values],
        }
        if self.source_type is SourceType.WEB:
            output.update(
                url=self.url,
                title=self.title,
                snippet=self.snippet,
                content=self.content,
                source_tier=self.source_tier,
                published_at=self.published_at,
                retrieved_at=self.retrieved_at,
                prompt_injection_suspected=self.prompt_injection_suspected,
            )
        elif self.source_type is SourceType.SQL:
            output.update(
                generated_sql=self.generated_sql,
                columns=list(self.columns),
                rows=[copy.deepcopy(dict(row)) for row in self.rows],
                row_count=self.row_count,
                truncated=self.truncated,
                error_type=self.error_type,
                latency_ms=self.latency_ms,
            )
        if include_raw:
            output["raw"] = copy.deepcopy(dict(self.raw))
        return output


def model_prior_evidence(
    value: str,
    *,
    retrieval_unit_id: str = "model-prior-1",
) -> "EvidenceBundle":
    """Represent a planning hypothesis without promoting it to final evidence."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("model prior value must be non-empty")
    item = EvidenceItem(
        source_type=SourceType.MODEL_PRIOR,
        tool_name="model_prior",
        retrieval_unit_id=retrieval_unit_id,
        text=value.strip(),
        verified=False,
    )
    return EvidenceBundle(
        source_type=SourceType.MODEL_PRIOR,
        tool_name="model_prior",
        items=(item,),
        raw_tool_result={"value": value.strip(), "verified": False},
        execution_success=True,
        source_match=True,
    )


@dataclass(frozen=True)
class EvidenceBundle:
    source_type: SourceType | None
    tool_name: str
    items: tuple[EvidenceItem, ...]
    raw_tool_result: Mapping[str, Any]
    execution_success: bool
    source_match: bool
    scope_violation_count: int = 0

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        output = {
            "source_type": self.source_type.value if self.source_type else "mixed",
            "tool_name": self.tool_name,
            "items": [item.to_dict(include_raw=include_raw) for item in self.items],
            "execution_success": self.execution_success,
            "source_match": self.source_match,
            "scope_violation_count": self.scope_violation_count,
        }
        if include_raw:
            output["raw_tool_result"] = copy.deepcopy(dict(self.raw_tool_result))
        return output


def normalize_rag_tool_result(
    result: Mapping[str, Any],
    catalog: DocumentCatalog,
    *,
    tool_name: str = "retrieve_document",
) -> EvidenceBundle:
    """Project legacy RAG evidence into the common schema without mutating it."""

    raw_tool_result = copy.deepcopy(dict(result))
    normalized: list[EvidenceItem] = []
    all_sources_match = True
    evidence = result.get("evidence", [])
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise ValueError("RAG tool evidence must be a sequence")

    for position, raw_item in enumerate(evidence):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"RAG evidence[{position}] must be an object")
        index_doc_id = raw_item.get("index_doc_id", raw_item.get("doc_id"))
        if not isinstance(index_doc_id, str) or not index_doc_id:
            raise ValueError(f"RAG evidence[{position}] has no physical doc_id")
        document = catalog.get_by_index_doc_id(index_doc_id)
        source_file = raw_item.get("source_file", raw_item.get("file_name"))
        if source_file is not None and not isinstance(source_file, str):
            raise ValueError(f"RAG evidence[{position}].source_file must be a string or null")
        source_matches = document is not None and source_file == document.source_file
        all_sources_match = all_sources_match and source_matches

        pages = raw_item.get("page", raw_item.get("pages", []))
        headings = raw_item.get("headings", [])
        if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
            raise ValueError(f"RAG evidence[{position}].page must be a sequence")
        if not isinstance(headings, Sequence) or isinstance(headings, (str, bytes)):
            raise ValueError(f"RAG evidence[{position}].headings must be a sequence")
        text = raw_item.get("text")
        if not isinstance(text, str):
            raise ValueError(f"RAG evidence[{position}].text must be a string")
        rerank_score = raw_item.get("rerank_score")
        if rerank_score is not None and not isinstance(rerank_score, (int, float)):
            raise ValueError(f"RAG evidence[{position}].rerank_score must be numeric or null")

        normalized.append(
            EvidenceItem(
                source_type=SourceType.RAG,
                tool_name=tool_name,
                catalog_doc_id=document.catalog_doc_id if document else None,
                index_doc_id=index_doc_id,
                company_name=document.company_name if document else None,
                source_file=document.source_file if document else source_file,
                page=tuple(pages),
                headings=tuple(str(item) for item in headings),
                retrieval_unit_id=raw_item.get("retrieval_unit_id"),
                rerank_score=float(rerank_score) if rerank_score is not None else None,
                text=text,
                monetary_values=extract_money_amounts(text),
                raw=copy.deepcopy(dict(raw_item)),
            )
        )

    bundle = EvidenceBundle(
        source_type=SourceType.RAG,
        tool_name=tool_name,
        items=tuple(normalized),
        raw_tool_result=raw_tool_result,
        execution_success=bool(result.get("success")),
        source_match=all_sources_match,
    )
    return bundle


def validate_rag_evidence_scope(
    bundle: EvidenceBundle,
    allowed_index_doc_ids: Sequence[str],
) -> EvidenceBundle:
    """Drop out-of-scope or catalog-mismatched evidence before synthesis."""

    if not allowed_index_doc_ids:
        raise ValueError("allowed_index_doc_ids must not be empty")
    allowed = frozenset(allowed_index_doc_ids)
    kept: list[EvidenceItem] = []
    violations: list[EvidenceItem] = []
    for item in bundle.items:
        expected_source_file = None
        raw_source_file = item.raw.get("source_file", item.raw.get("file_name"))
        if item.catalog_doc_id is not None:
            expected_source_file = item.source_file
        valid = (
            item.index_doc_id in allowed
            and item.catalog_doc_id is not None
            and raw_source_file == expected_source_file
        )
        (kept if valid else violations).append(item)

    payload = {
        "allowed_index_doc_ids": sorted(allowed),
        "candidate_count_before_scope": len(bundle.items),
        "candidate_count_after_scope": len(kept),
        "violation_count": len(violations),
        "violations": [
            {
                "index_doc_id": item.index_doc_id,
                "catalog_doc_id": item.catalog_doc_id,
                "source_file": item.source_file,
                "text_preview": item.text[:_TRACE_PREVIEW_CHARS],
            }
            for item in violations[:10]
        ],
    }
    logger.debug("SOURCE_SCOPE_VALIDATION: %s", json.dumps(payload, ensure_ascii=False))
    for violation in violations:
        logger.debug(
            "SOURCE_SCOPE_VIOLATION: %s",
            json.dumps(
                {
                    "index_doc_id": violation.index_doc_id,
                    "catalog_doc_id": violation.catalog_doc_id,
                    "source_file": violation.source_file,
                    "text_preview": violation.text[:_TRACE_PREVIEW_CHARS],
                },
                ensure_ascii=False,
            ),
        )
    validated = replace(
        bundle,
        items=tuple(kept),
        source_match=(bundle.source_match and not violations),
        scope_violation_count=len(violations),
    )
    notify_runtime_artifact("evidence.bundle", validated)
    return validated


def normalize_web_tool_result(
    search_results: Sequence[Mapping[str, Any]],
    fetched_content: Mapping[str, str] | None = None,
) -> EvidenceBundle:
    """Normalize bounded search/fetch output without treating Web as local evidence."""

    fetched = dict(fetched_content or {})
    items: list[EvidenceItem] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for position, result in enumerate(search_results, start=1):
        url = result.get("url")
        title = result.get("title")
        snippet = result.get("snippet")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if not isinstance(title, str):
            title = ""
        if not isinstance(snippet, str):
            snippet = ""
        safe_title, title_injection = _sanitize_web_text(title)
        safe_snippet, snippet_injection = _sanitize_web_text(snippet)
        content = fetched.get(url)
        if not isinstance(content, str):
            content = None
        original_text = content.strip() if content and content.strip() else safe_snippet.strip()
        text, content_injection = _sanitize_web_text(original_text)
        injection_suspected = title_injection or snippet_injection or content_injection
        if not text:
            continue
        identifier = f"web-{position}"
        raw = copy.deepcopy(dict(result))
        if content is not None:
            raw["content"] = content
        items.append(
            EvidenceItem(
                source_type=SourceType.WEB,
                tool_name="fetch" if content else "search",
                retrieval_unit_id=identifier,
                text=text,
                url=url,
                title=safe_title,
                snippet=safe_snippet,
                content=text if content is not None else None,
                source_tier=(
                    str(result.get("source_tier"))
                    if result.get("source_tier")
                    else _source_tier(url)
                ),
                published_at=_published_at(result),
                retrieved_at=(
                    str(result.get("retrieved_at"))
                    if result.get("retrieved_at")
                    else retrieved_at
                ),
                prompt_injection_suspected=injection_suspected,
                monetary_values=extract_money_amounts(text),
                raw=raw,
            )
        )

    raw_tool_result = {
        "search_results": copy.deepcopy([dict(item) for item in search_results]),
        "fetched_content": copy.deepcopy(fetched),
    }
    bundle = EvidenceBundle(
        source_type=SourceType.WEB,
        tool_name="controlled_web_fallback",
        items=tuple(items),
        raw_tool_result=raw_tool_result,
        execution_success=bool(items),
        source_match=True,
    )
    notify_runtime_artifact("evidence.bundle", bundle)
    return bundle


def normalize_sql_tool_result(
    result: Mapping[str, Any],
    *,
    tool_name: str = "query_financial_db",
) -> EvidenceBundle:
    """Project the unchanged SQL Tool result into unified evidence."""

    raw_tool_result = copy.deepcopy(dict(result))
    success = bool(result.get("success"))
    items: tuple[EvidenceItem, ...] = ()
    if success:
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        row_count = result.get("row_count", 0)
        truncated = result.get("truncated", False)
        generated_sql = result.get("generated_sql")
        latency_ms = result.get("latency_ms")
        if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
            raise ValueError("SQL columns must be a sequence")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("SQL rows must be a sequence")
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("every SQL row must be an object")
        if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
            raise ValueError("SQL row_count must be a non-negative integer")
        if type(truncated) is not bool:
            raise ValueError("SQL truncated must be a boolean")
        if generated_sql is not None and not isinstance(generated_sql, str):
            raise ValueError("generated_sql must be a string or null")
        if latency_ms is not None and not isinstance(latency_ms, (int, float)):
            raise ValueError("latency_ms must be numeric or null")
        normalized_rows = tuple(copy.deepcopy(dict(row)) for row in rows)
        monetary_values = extract_sql_money_amounts(normalized_rows)
        items = (
            EvidenceItem(
                source_type=SourceType.SQL,
                tool_name=tool_name,
                retrieval_unit_id="sql-result-1",
                text=json.dumps(list(normalized_rows), ensure_ascii=False, default=str),
                generated_sql=generated_sql,
                columns=tuple(str(column) for column in columns),
                rows=normalized_rows,
                monetary_values=monetary_values,
                row_count=row_count,
                truncated=truncated,
                error_type=result.get("error_type"),
                latency_ms=float(latency_ms) if latency_ms is not None else None,
                raw=raw_tool_result,
            ),
        )
    bundle = EvidenceBundle(
        source_type=SourceType.SQL,
        tool_name=tool_name,
        items=items,
        raw_tool_result=raw_tool_result,
        execution_success=success,
        source_match=True,
    )
    notify_runtime_artifact("evidence.bundle", bundle)
    return bundle


def merge_evidence_bundles(*bundles: EvidenceBundle) -> EvidenceBundle:
    """Merge source bundles while retaining each item's provenance boundary."""

    if not bundles:
        raise ValueError("at least one evidence bundle is required")
    items: list[EvidenceItem] = []
    seen: set[tuple[SourceType, str | None]] = set()
    for bundle in bundles:
        for item in bundle.items:
            key = (item.source_type, item.retrieval_unit_id)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    source_types = {item.source_type for item in items}
    return EvidenceBundle(
        source_type=next(iter(source_types)) if len(source_types) == 1 else None,
        tool_name="multi_source_synthesis",
        items=tuple(items),
        raw_tool_result={
            "bundles": [bundle.to_dict(include_raw=True) for bundle in bundles],
        },
        execution_success=all(bundle.execution_success for bundle in bundles),
        source_match=all(bundle.source_match for bundle in bundles),
        scope_violation_count=sum(bundle.scope_violation_count for bundle in bundles),
    )
