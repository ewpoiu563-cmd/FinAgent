"""Strict lightweight LLM adapter used only for ambiguous source plans."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable

from prompts.router_prompt import build_router_prompt

from .models import EntityResolution, RouterDecision, SourceType
from .tool_registry import ToolRegistry


logger = logging.getLogger(__name__)


class LLMSourceRouter:
    """Return a source-only decision from one bounded, non-ReAct LLM call."""

    def __init__(self, llm_call: Callable[..., str] | None = None, *, timeout: int = 30) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._llm_call = llm_call
        self.timeout = timeout

    def route(
        self,
        question: str,
        entity: EntityResolution,
        registry: ToolRegistry,
        *,
        allowed_sources: tuple[SourceType, ...] | None = None,
        rejected_sources: tuple[SourceType, ...] = (),
        rejection_reason: str | None = None,
    ) -> RouterDecision:
        started = time.perf_counter()
        call = self._llm_call
        if call is None:
            # Lazy import keeps Phase A models and deterministic rules independent
            # from runtime credentials and the current LLM transport.
            from config import call_llm

            call = call_llm
        kwargs = {"temperature": 0.0, "timeout": self.timeout}
        if self._llm_call is None:
            kwargs["trace_operation"] = "source_routing"
        raw = call(
            build_router_prompt(
                question,
                entity,
                registry,
                allowed_sources=allowed_sources,
                rejected_sources=rejected_sources,
                rejection_reason=rejection_reason,
            ),
            **kwargs,
        )
        decision = self.parse_decision(raw)
        if allowed_sources is not None and (
            decision.primary_source not in allowed_sources
            or any(source not in allowed_sources for source in decision.required_sources)
        ):
            raise ValueError("router selected a source outside allowed_sources")
        if rejected_sources and (
            decision.primary_source in rejected_sources
            or any(source in rejected_sources for source in decision.required_sources)
        ):
            raise ValueError("router repeated a rejected source")
        logger.debug(
            "LIGHTWEIGHT_SOURCE_ROUTER: %s",
            json.dumps(
                {
                    "primary_source": decision.primary_source.value,
                    "required_sources": [source.value for source in decision.required_sources],
                    "web_fallback_allowed": decision.web_fallback_allowed,
                    "reason_preview": decision.reason[:300],
                    "tool_names": [tool.name for tool in registry.all()],
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                },
                ensure_ascii=False,
            ),
        )
        return decision

    @staticmethod
    def parse_decision(raw: str) -> RouterDecision:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("router output must be a non-empty string")
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("router output must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("router output must be a JSON object")

        required = payload.get("required_sources")
        if not isinstance(required, list) or not required or any(not isinstance(item, str) for item in required):
            raise ValueError("required_sources must be a non-empty list of strings")
        try:
            required_sources = tuple(SourceType(item) for item in required)
            primary_source = SourceType(payload.get("primary_source"))
        except (TypeError, ValueError) as error:
            raise ValueError("router sources must be sql, rag, or web") from error
        executable_sources = {SourceType.SQL, SourceType.RAG, SourceType.WEB}
        if primary_source not in executable_sources or any(
            source not in executable_sources for source in required_sources
        ):
            raise ValueError("router sources must be sql, rag, or web")
        if len(set(required_sources)) != len(required_sources):
            raise ValueError("required_sources must not contain duplicates")
        if primary_source not in required_sources:
            raise ValueError("primary_source must be included in required_sources")
        fallback = payload.get("web_fallback_allowed", True)
        reason = payload.get("reason", "")
        if type(fallback) is not bool or not isinstance(reason, str):
            raise ValueError("invalid router fallback or reason field")
        return RouterDecision(
            primary_source=primary_source,
            required_sources=required_sources,
            web_fallback_allowed=fallback,
            reason=reason.strip(),
        )
