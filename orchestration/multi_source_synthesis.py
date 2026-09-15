"""Strict evidence-partitioned synthesis for explicit two-source queries."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .evidence import EvidenceBundle, EvidenceItem
from .llm_errors import diagnose_llm_error
from .models import SourceType
from prompts.multi_source_synthesis_prompt import build_multi_source_synthesis_prompt


logger = logging.getLogger(__name__)
_TRACE_PREVIEW_CHARS = 300
_ALLOWED_SOURCES = frozenset({"sql", "rag", "web"})
_SOURCE_LABELS = {
    "rag": ("本地文档", "招股说明书", "招股书", "年报", "财报"),
    "sql": ("数据库", "结构化", "SQL", "sql"),
    "web": ("Web", "web", "公开信息", "网页", "新闻"),
}


class ClaimEvidenceValidationError(ValueError):
    """Raised when synthesis cites evidence outside the supplied source blocks."""


@dataclass(frozen=True)
class MultiSourceClaim:
    claim: str
    supported_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "supported_evidence_ids": list(self.supported_evidence_ids),
        }


@dataclass(frozen=True)
class MultiSourceSynthesisResult:
    status: str
    answer: str | None
    claims: tuple[MultiSourceClaim, ...]
    source_summary: Mapping[str, str]
    error_type: str | None = None
    latency_ms: int = 0
    failure_kind: str | None = None

    @property
    def successful(self) -> bool:
        return self.status == "success"


class MultiSourceSynthesizer:
    def __init__(self, llm_call: Callable[..., str] | None = None, *, timeout: int = 60) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._llm_call = llm_call
        self.timeout = timeout

    def synthesize(
        self,
        question: str,
        evidence_by_source: Mapping[str, EvidenceBundle],
        required_sources: set[str] | frozenset[str],
    ) -> MultiSourceSynthesisResult:
        started = time.perf_counter()
        required = frozenset(required_sources)
        try:
            prepared, evidence_sources = _prepare_evidence(evidence_by_source, required)
            call = self._llm_call
            if call is None:
                from config import call_llm

                call = call_llm
            prompt = build_multi_source_synthesis_prompt(question, prepared)
            logger.debug(
                "MULTI_SOURCE_SYNTHESIS: %s",
                json.dumps(
                    {
                        "status": "started",
                        "required_sources": sorted(required),
                        "evidence_count_by_source": {
                            source: len(items) for source, items in prepared.items()
                        },
                    },
                    ensure_ascii=False,
                ),
            )
            kwargs = {"temperature": 0.0, "timeout": self.timeout}
            if self._llm_call is None:
                kwargs["trace_operation"] = "multi_source_synthesis"
            raw = call(prompt, **kwargs)
            parsed = self.parse_result(
                raw,
                evidence_sources=evidence_sources,
                required_sources=required,
            )
            result = MultiSourceSynthesisResult(
                status=parsed.status,
                answer=parsed.answer,
                claims=parsed.claims,
                source_summary=parsed.source_summary,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            logger.debug(
                "CLAIM_EVIDENCE_VALIDATION: %s",
                json.dumps(
                    {
                        "status": "success",
                        "claim_count": len(result.claims),
                        "evidence_ids": sorted(evidence_sources),
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as error:
            diagnostic = diagnose_llm_error(error)
            failure_kind = diagnostic.failure_kind
            status = "provider_content_block" if failure_kind == "provider_content_block" else "generation_failure"
            result = MultiSourceSynthesisResult(
                status=status,
                answer=None,
                claims=(),
                source_summary={},
                error_type=type(error).__name__,
                latency_ms=round((time.perf_counter() - started) * 1000),
                failure_kind=failure_kind,
            )
            logger.debug(
                "CLAIM_EVIDENCE_VALIDATION: %s",
                json.dumps(
                    {
                        "status": "failure",
                        "error_type": result.error_type,
                        "failure_kind": failure_kind,
                    },
                    ensure_ascii=False,
                ),
            )
        logger.debug(
            "MULTI_SOURCE_SYNTHESIS: %s",
            json.dumps(
                {
                    "status": result.status,
                    "claim_count": len(result.claims),
                    "answer_preview": (result.answer or "")[:_TRACE_PREVIEW_CHARS],
                    "error_type": result.error_type,
                    "latency_ms": result.latency_ms,
                },
                ensure_ascii=False,
            ),
        )
        return result

    @staticmethod
    def parse_result(
        raw: str,
        *,
        evidence_sources: Mapping[str, str],
        required_sources: frozenset[str],
    ) -> MultiSourceSynthesisResult:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("multi-source synthesis output must be a non-empty string")
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("multi-source synthesis output must be valid JSON") from error
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise ValueError("multi-source synthesis status must be success")
        answer = payload.get("answer")
        claims_raw = payload.get("claims")
        summaries = payload.get("source_summary")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("multi-source synthesis answer must be non-empty")
        if not isinstance(claims_raw, list) or not claims_raw:
            raise ValueError("multi-source synthesis claims must be a non-empty list")
        if not isinstance(summaries, dict):
            raise ValueError("source_summary must be an object")
        if set(summaries) != set(required_sources):
            raise ValueError("source_summary must contain exactly the required sources")
        if any(not isinstance(value, str) or not value.strip() for value in summaries.values()):
            raise ValueError("every source summary must be non-empty")

        claims: list[MultiSourceClaim] = []
        cited_sources: set[str] = set()
        for position, raw_claim in enumerate(claims_raw):
            if not isinstance(raw_claim, dict):
                raise ValueError(f"claims[{position}] must be an object")
            claim = raw_claim.get("claim")
            supported = raw_claim.get("supported_evidence_ids")
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError(f"claims[{position}].claim must be non-empty")
            if not isinstance(supported, list) or not supported or any(
                not isinstance(identifier, str) for identifier in supported
            ):
                raise ValueError(f"claims[{position}].supported_evidence_ids must be non-empty strings")
            if len(set(supported)) != len(supported):
                raise ValueError(f"claims[{position}] contains duplicate evidence IDs")
            unknown = set(supported).difference(evidence_sources)
            if unknown:
                raise ClaimEvidenceValidationError(
                    f"claims[{position}] references unknown evidence IDs: {sorted(unknown)}"
                )
            cited_sources.update(evidence_sources[identifier] for identifier in supported)
            claims.append(MultiSourceClaim(claim.strip(), tuple(supported)))
        if not cited_sources.issuperset(required_sources):
            missing = sorted(required_sources.difference(cited_sources))
            raise ClaimEvidenceValidationError(f"claims do not cite required sources: {missing}")
        for source in required_sources:
            if not any(label in answer for label in _SOURCE_LABELS[source]):
                raise ClaimEvidenceValidationError(
                    f"answer does not explicitly distinguish source: {source}"
                )
        return MultiSourceSynthesisResult(
            status="success",
            answer=answer.strip(),
            claims=tuple(claims),
            source_summary={source: summaries[source].strip() for source in sorted(required_sources)},
        )


def _prepare_evidence(
    evidence_by_source: Mapping[str, EvidenceBundle],
    required_sources: frozenset[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if len(required_sources) != 2 or not required_sources.issubset(_ALLOWED_SOURCES):
        raise ValueError("Phase D supports exactly two of sql, rag, and web")
    if not set(evidence_by_source).issuperset(required_sources):
        raise ValueError("all required source bundles must be present before synthesis")
    prepared: dict[str, list[dict[str, Any]]] = {}
    evidence_sources: dict[str, str] = {}
    for source in sorted(required_sources):
        bundle = evidence_by_source[source]
        if bundle.source_type is None or bundle.source_type.value != source:
            raise ValueError(f"{source} evidence bundle has the wrong source type")
        if not bundle.execution_success or not bundle.source_match or not bundle.items:
            raise ValueError(f"{source} evidence bundle is not successful and source-matched")
        prepared[source] = []
        for position, item in enumerate(bundle.items, start=1):
            if not item.verified or item.source_type is SourceType.MODEL_PRIOR:
                continue
            if item.source_type.value != source:
                raise ValueError(f"{source} evidence block contains a foreign source item")
            raw_id = item.retrieval_unit_id or f"item-{position}"
            qualified_id = f"{source}:{raw_id}"
            if qualified_id in evidence_sources:
                raise ValueError(f"duplicate qualified evidence ID: {qualified_id}")
            evidence_sources[qualified_id] = source
            prepared[source].append(_synthesis_item(item, qualified_id))
        if not prepared[source]:
            raise ValueError(f"{source} has no verified evidence for synthesis")
    return prepared, evidence_sources


def _synthesis_item(item: EvidenceItem, qualified_id: str) -> dict[str, Any]:
    output = item.to_dict()
    output["evidence_id"] = qualified_id
    output.pop("raw", None)
    return output
