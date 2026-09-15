"""Offline deadline tests: fake HTTP and monotonic clock, no API calls."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from requests import HTTPError

import config
from orchestration.llm_errors import diagnose_llm_error


def _http_error(status_code):
    return HTTPError(
        f"{status_code} response",
        response=Mock(status_code=status_code, text="response body"),
    )


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(config, "_time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=sleep))
    return now, sleeps


def test_retries_share_deadline(monkeypatch, clock):
    now, sleeps = clock
    timeouts = []

    def post(*args, timeout, **kwargs):
        timeouts.append(timeout)
        now[0] += 10 if len(timeouts) == 1 else timeout
        raise config._requests.Timeout("socket timeout")

    monkeypatch.setattr(config._requests, "post", post)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        config.call_llm("test", timeout=30)
    assert timeouts == [30, 17]
    assert sleeps == [3]
    assert now[0] == 30


@pytest.mark.parametrize("elapsed,expected_sleep", [(30, []), (29, [1])])
def test_exhausted_budget_stops_retry(monkeypatch, clock, elapsed, expected_sleep):
    now, sleeps = clock

    def post(*args, **kwargs):
        now[0] += elapsed
        raise config._requests.Timeout("socket timeout")

    post = Mock(side_effect=post)
    monkeypatch.setattr(config._requests, "post", post)
    with pytest.raises(TimeoutError, match="budget exhausted"):
        config.call_llm("test", timeout=30)
    post.assert_called_once()
    assert sleeps == expected_sleep


def test_non_transport_or_parse_error_is_not_retried(monkeypatch, clock):
    post = Mock(side_effect=ValueError("invalid response"))
    monkeypatch.setattr(config._requests, "post", post)
    with pytest.raises(ValueError, match="invalid response"):
        config.call_llm("test", timeout=60)
    assert post.call_count == 1
    assert clock[1] == []


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_deterministic_http_client_errors_are_not_retried(monkeypatch, clock, status_code):
    post = Mock(side_effect=_http_error(status_code))
    monkeypatch.setattr(config._requests, "post", post)

    with pytest.raises(HTTPError):
        config.call_llm("test", timeout=60)

    assert post.call_count == 1
    assert clock[1] == []


@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
def test_retryable_http_statuses_retry_at_most_once(monkeypatch, clock, status_code):
    post = Mock(side_effect=_http_error(status_code))
    monkeypatch.setattr(config._requests, "post", post)

    with pytest.raises(HTTPError):
        config.call_llm("test", timeout=60)

    assert post.call_count == 2
    assert clock[1] == [3]


def test_zero_budget_never_calls_http(monkeypatch, clock):
    post = Mock()
    monkeypatch.setattr(config._requests, "post", post)
    with pytest.raises(TimeoutError):
        config.call_llm("test", timeout=0)
    post.assert_not_called()


def test_blocking_http_cannot_hold_caller_past_deadline(monkeypatch):
    release = Event()
    done = Event()

    def post(*args, **kwargs):
        try:
            release.wait(2)
            raise ValueError("late response")
        finally:
            done.set()

    post = Mock(side_effect=post)
    monkeypatch.setattr(config._requests, "post", post)
    started = config._time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="budget exhausted"):
            config.call_llm("test", timeout=0.05)
        assert config._time.monotonic() - started < 0.5
        post.assert_called_once()
    finally:
        release.set()
        assert done.wait(1)


def test_success_preserves_payload_and_user_agent(monkeypatch, clock):
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "<think>x</think> answer "}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(config._requests, "post", post)
    assert config.call_llm("test", temperature=0.2, timeout=30) == "answer"
    kwargs = post.call_args.kwargs
    assert kwargs["timeout"] == 30
    assert kwargs["headers"]["User-Agent"].startswith("Mozilla/5.0")
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "test"}]
    assert kwargs["json"]["temperature"] == 0.2


def test_llm_failure_kinds_are_explicit():
    content_block = diagnose_llm_error(_http_error_with_body(
        400,
        '{"error":{"code":"data_inspection_failed","message":"blocked"}}',
    ))
    assert content_block.failure_kind == "provider_content_block"
    assert content_block.provider_error_code == "data_inspection_failed"

    assert diagnose_llm_error(config._requests.Timeout("timeout")).failure_kind == "transport_failure"
    assert diagnose_llm_error(ValueError("bad JSON")).failure_kind == "parse_failure"
    assert diagnose_llm_error(_http_error_with_body(400, '{"code":"bad_request"}')).failure_kind == "generation_failure"


def _http_error_with_body(status_code, body):
    return HTTPError(
        f"{status_code} response",
        response=Mock(status_code=status_code, text=body),
    )
