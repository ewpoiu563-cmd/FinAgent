"""Sanitized diagnostics and retry classification for LLM failures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


_BODY_PREVIEW_CHARS = 1000


@dataclass(frozen=True)
class LLMErrorDiagnostic:
    error_type: str
    status_code: int | None
    response_body_preview: str | None
    provider_error_code: str | None
    failure_kind: str


def diagnose_llm_error(error: Exception, *, secrets: Iterable[str | None] = ()) -> LLMErrorDiagnostic:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    status_code = status if isinstance(status, int) else None
    body: str | None = None
    provider_error_code: str | None = None
    if response is not None:
        try:
            raw = response.text
        except Exception:
            raw = None
        if isinstance(raw, str) and raw.strip():
            body = raw.strip()[:_BODY_PREVIEW_CHARS]
            provider_error_code = _provider_error_code(error, body)
            for secret in secrets:
                if secret:
                    body = body.replace(secret, "[redacted]")
    if provider_error_code is None:
        provider_error_code = _provider_error_code(error, None)
    return LLMErrorDiagnostic(
        error_type=type(error).__name__,
        status_code=status_code,
        response_body_preview=body,
        provider_error_code=provider_error_code,
        failure_kind=classify_llm_failure(
            type(error).__name__,
            status_code,
            provider_error_code=provider_error_code,
        ),
    )


def classify_llm_failure(
    error_type: str | None,
    status_code: int | None,
    *,
    provider_error_code: str | None = None,
) -> str:
    """Map low-level failures to the four control-layer outcomes."""

    if status_code == 400 and provider_error_code == "data_inspection_failed":
        return "provider_content_block"
    if is_retryable_llm_error(
        error_type,
        status_code,
        provider_error_code=provider_error_code,
    ):
        return "transport_failure"
    if error_type in {"ValueError", "JSONDecodeError"}:
        return "parse_failure"
    return "generation_failure"


def is_retryable_llm_error(
    error_type: str | None,
    status_code: int | None,
    *,
    provider_error_code: str | None = None,
) -> bool:
    """Retry only transport failures, 408/429, and server-side HTTP errors."""

    if provider_error_code == "data_inspection_failed":
        return False
    if status_code is not None:
        return status_code in {408, 429} or 500 <= status_code <= 599
    return error_type in {
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "TimeoutError",
        "ChunkedEncodingError",
        "ProxyError",
        "SSLError",
        "ConnectionResetError",
        "APIConnectionError",
        "APITimeoutError",
    }


def _provider_error_code(error: Exception, body: str | None) -> str | None:
    direct = getattr(error, "code", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    nested = payload.get("error")
    candidates = (
        nested.get("code") if isinstance(nested, dict) else None,
        payload.get("code"),
    )
    return next(
        (value.strip() for value in candidates if isinstance(value, str) and value.strip()),
        None,
    )
