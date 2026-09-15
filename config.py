"""Runtime constants from environment variables + LLM call helper."""

import logging
import os
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from threading import Thread

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── LLM ──────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-max")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

# ── Search ───────────────────────────────────────────────────────────
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# ── Debug ────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "0") == "1"
NO_TIMEOUT = os.getenv("NO_TIMEOUT", "0") == "1"
DEBUG_UI_ENABLED = os.getenv("DEBUG_UI_ENABLED", "0") == "1"

# Structured trace persistence is deliberately independent from DEBUG logging.
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "0") == "1"
TRACE_INCLUDE_CONTENT = os.getenv("TRACE_INCLUDE_CONTENT", "0") == "1"
TRACE_DIR = os.getenv(
    "TRACE_DIR",
    os.path.join(os.path.dirname(__file__), "outputs", "traces"),
)

# ── Session persistence ─────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))

# ── Resource controls ────────────────────────────────────────────────
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "30"))
MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "10"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))            # total LLM call budget (seconds)
MAX_TOTAL_SECONDS = 999999 if NO_TIMEOUT else int(os.getenv("TOTAL_TIMEOUT", "600"))
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", "8"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))  # max output tokens per LLM call

# ── Page fetch controls ─────────────────────────────────────────────
MAX_FETCH_PAGES = int(os.getenv("MAX_FETCH_PAGES", "6"))    # max pages fetched per question
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "15"))       # single page timeout (seconds)
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "15000"))   # max chars sent to LLM per page

# ── LLM call via requests ────────────────────────────────────────────
# Cloudflare blocks the default openai-python User-Agent on this endpoint,
# so we bypass the SDK entirely and use requests with a browser-like UA.
import time as _time  # noqa: E402
import requests as _requests  # noqa: E402

_LLM_ENDPOINT = f"{BASE_URL}/chat/completions"
_LLM_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

_LLM_MAX_RETRIES = 2
_LLM_BACKOFF = [3, 5, 10]


def _retryable_llm_failure(error) -> bool:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in {408, 429} or 500 <= status <= 599
    return isinstance(error, (_requests.Timeout, _requests.ConnectionError, ConnectionResetError))


def _llm_usage(data):
    """Return provider-reported usage only; never estimate missing values."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    def token(*names):
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return {
        "input_tokens": token("prompt_tokens", "input_tokens"),
        "output_tokens": token("completion_tokens", "output_tokens"),
        "total_tokens": token("total_tokens"),
    }


def _record_llm_event(**event):
    """Import tracing lazily and keep observability strictly fail-open."""
    try:
        from orchestration.trace import record_trace_event

        record_trace_event(**event)
    except Exception:
        return None


def _llm_status_code(error):
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def call_llm(
    prompt: str,
    temperature: float = 0.1,
    timeout: int = LLM_TIMEOUT,
    trace_operation: str = "unspecified",
) -> str:
    """Return content within one wall-clock budget, including retries/backoff.

    Requests timeouts limit socket inactivity, not total response duration.
    A daemon worker bounds caller waiting even for a slowly trickling response;
    an in-flight request cannot be forcibly cancelled and may finish afterwards.
    """
    call_started = _time.monotonic()
    deadline = call_started + timeout
    attempt_count = 0
    common_attributes = {
        "model": LLM_MODEL,
        "operation": trace_operation,
        "timeout": timeout,
        "prompt_char_count": len(prompt) if isinstance(prompt, str) else None,
    }
    _record_llm_event(
        event_type="llm.call.started",
        stage="llm_call",
        status="started",
        attributes={
            **common_attributes,
            # No HTTP attempt exists yet.  ``None`` keeps lifecycle state
            # distinct from the terminal count reported by completed/failed.
            "attempt_count": None,
            "retry_count": 0,
            "provider_status": None,
            "status_code": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    )

    def remaining():
        value = deadline - _time.monotonic()
        if value <= 0:
            raise TimeoutError("call_llm total time budget exhausted")
        return value

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": LLM_MAX_TOKENS,
        "enable_thinking": False,  # Disable thinking mode for speed
        "stop": ["\nObservation:", "\nObservation", "\nResult:", "\nResult"],
    }

    def request_content(future):
        try:
            resp = _requests.post(
                _LLM_ENDPOINT,
                json=payload,
                headers=_LLM_HEADERS,
                timeout=remaining(),
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]

            # Strip <think>...</think> tags in case they still appear
            if "</think>" in content:
                idx = content.index("</think>")
                content = content[idx + len("</think>"):]

            response_status = getattr(resp, "status_code", None)
            provider_status = data.get("status") if isinstance(data, dict) else None
            future.set_result(
                {
                    "content": content.strip(),
                    "usage": _llm_usage(data),
                    "status_code": (
                        response_status
                        if isinstance(response_status, int) and not isinstance(response_status, bool)
                        else None
                    ),
                    "provider_status": (
                        provider_status if isinstance(provider_status, str) else None
                    ),
                }
            )
        except Exception as e:
            future.set_exception(e)

    last_err = None
    try:
        for attempt in range(_LLM_MAX_RETRIES):
            remaining()
            future = Future()
            attempt_count += 1
            Thread(target=request_content, args=(future,), daemon=True).start()
            try:
                response_data = future.result(timeout=remaining())
                remaining()
                usage = response_data["usage"]
                _record_llm_event(
                    event_type="llm.call.completed",
                    stage="llm_call",
                    status="completed",
                    duration_ms=round((_time.monotonic() - call_started) * 1000),
                    attributes={
                        **common_attributes,
                        "attempt_count": attempt_count,
                        "retry_count": max(0, attempt_count - 1),
                        "provider_status": response_data["provider_status"],
                        "status_code": response_data["status_code"],
                        **usage,
                        "response_char_count": len(response_data["content"]),
                    },
                )
                return response_data["content"]
            except Exception as e:
                # A worker still running means the caller's total wait expired.
                if isinstance(e, FutureTimeoutError) and not future.done():
                    raise TimeoutError("call_llm total time budget exhausted") from e
                last_err = e
                if not _retryable_llm_failure(e):
                    raise
                budget = remaining()
                if attempt < _LLM_MAX_RETRIES - 1:
                    _time.sleep(min(_LLM_BACKOFF[attempt], budget))
                    remaining()

        raise last_err  # type: ignore
    except Exception as error:
        _record_llm_event(
            event_type="llm.call.failed",
            stage="llm_call",
            status="failed",
            duration_ms=round((_time.monotonic() - call_started) * 1000),
            error_type=type(error).__name__,
            attributes={
                **common_attributes,
                "attempt_count": attempt_count,
                "retry_count": max(0, attempt_count - 1),
                "provider_status": None,
                "status_code": _llm_status_code(error),
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        )
        raise


# ── Logging setup ──────────────────────────────────────────────────
def setup_logging(log_file: str | None = None):
    """统一配置日志系统。

    - DEBUG=0: 控制台只输出 WARNING+，无文件输出
    - DEBUG=1: 控制台输出 INFO+，文件输出 DEBUG+（全量）
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if DEBUG else logging.WARNING)

    # 清理已有 handlers（避免重复配置）
    root.handlers.clear()

    # 格式
    fmt_console = logging.Formatter("[%(levelname).1s] %(message)s")
    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台 handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if DEBUG else logging.WARNING)
    console.setFormatter(fmt_console)
    root.addHandler(console)

    # 文件 handler（仅 DEBUG 模式且指定了路径）
    if DEBUG and log_file:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        root.addHandler(fh)
