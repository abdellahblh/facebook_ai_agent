"""Tests for retry. Weighted toward what must NOT be retried."""
import asyncio
import time

import pytest

from app.util.retry import (
    is_retryable, is_retryable_send, with_retry, with_retry_send,
)


# ── fakes that look like real SDK exceptions ─────────────────────────────────
class _Resp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}


class HTTPStatusError(Exception):          # httpx shape
    def __init__(self, status, headers=None):
        super().__init__(f"HTTP {status}")
        self.response = _Resp(status, headers)


class ClientError(Exception):              # google-genai shape
    def __init__(self, code):
        super().__init__(f"code {code}")
        self.code = code


class ReadTimeout(Exception): pass         # httpx timeout
class ConnectError(Exception): pass
class ResourceExhausted(Exception): pass   # google.api_core 429
class ValueError_(ValueError): pass


# ── classification ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("exc", [
    HTTPStatusError(429), HTTPStatusError(500), HTTPStatusError(502),
    HTTPStatusError(503), HTTPStatusError(504), HTTPStatusError(408),
    ClientError(429), ClientError(503),
    ReadTimeout(), ConnectError(), ResourceExhausted(),
])
def test_transient_is_retryable(exc):
    assert is_retryable(exc)


@pytest.mark.parametrize("exc", [
    HTTPStatusError(400), HTTPStatusError(401), HTTPStatusError(403),
    HTTPStatusError(404), HTTPStatusError(422),
    ClientError(403), ClientError(400),
    ValueError("bad input"), KeyError("missing"), TypeError("nope"),
])
def test_permanent_is_not_retryable(exc):
    assert not is_retryable(exc)


def test_explicit_status_beats_class_name():
    """A 403 whose class is named ConnectionError must NOT retry.

    Status is authoritative. Without this ordering, an SDK that wraps auth
    failures in a connection-ish class would make every bad token a 4x retry.
    """
    class ConnectionError_(Exception):
        def __init__(self):
            self.response = _Resp(403)
    assert not is_retryable(ConnectionError_())


# ── behaviour ────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_succeeds_first_try_no_delay():
    calls = []
    async def ok():
        calls.append(1); return "fine"
    t0 = time.monotonic()
    assert await with_retry(ok) == "fine"
    assert len(calls) == 1
    assert time.monotonic() - t0 < 0.1


@pytest.mark.asyncio
async def test_recovers_on_second_attempt():
    calls = []
    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise HTTPStatusError(503)
        return "recovered"
    assert await with_retry(flaky, base_delay=0.01) == "recovered"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_permanent_error_tried_exactly_once():
    """The 403 lesson: a bad token must fail instantly, not after 4 attempts."""
    calls = []
    async def forbidden():
        calls.append(1)
        raise HTTPStatusError(403)
    with pytest.raises(HTTPStatusError):
        await with_retry(forbidden, attempts=4, base_delay=0.01)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_raises_last_exception_after_exhausting():
    async def always():
        raise HTTPStatusError(500)
    with pytest.raises(HTTPStatusError):
        await with_retry(always, attempts=3, base_delay=0.01)


@pytest.mark.asyncio
async def test_respects_attempt_count():
    calls = []
    async def always():
        calls.append(1); raise HTTPStatusError(500)
    with pytest.raises(HTTPStatusError):
        await with_retry(always, attempts=3, base_delay=0.01)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_budget_stops_early():
    """Budget is wall-clock, so a slow call cannot blow past it."""
    calls = []
    async def slow_fail():
        calls.append(1)
        await asyncio.sleep(0.15)
        raise HTTPStatusError(500)
    t0 = time.monotonic()
    with pytest.raises(HTTPStatusError):
        await with_retry(slow_fail, attempts=10, base_delay=0.1, budget=0.4)
    assert time.monotonic() - t0 < 1.0
    assert len(calls) < 10


@pytest.mark.asyncio
async def test_retry_after_header_is_honoured():
    calls = []
    async def limited():
        calls.append(1)
        if len(calls) == 1:
            raise HTTPStatusError(429, {"retry-after": "0.2"})
        return "ok"
    t0 = time.monotonic()
    assert await with_retry(limited, base_delay=5.0) == "ok"
    elapsed = time.monotonic() - t0
    # Slept ~0.2s (the header), not 5s (our backoff)
    assert 0.1 < elapsed < 1.0


@pytest.mark.asyncio
async def test_absurd_retry_after_is_ignored():
    """A 3600 Retry-After must not park the coroutine for an hour."""
    calls = []
    async def limited():
        calls.append(1)
        if len(calls) == 1:
            raise HTTPStatusError(429, {"retry-after": "3600"})
        return "ok"
    t0 = time.monotonic()
    assert await with_retry(limited, base_delay=0.01, max_delay=0.05) == "ok"
    assert time.monotonic() - t0 < 1.0


# ── sends: the ambiguity rule ────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_send_does_not_retry_on_timeout():
    """A timed-out POST may already have delivered. Duplicate > missing."""
    calls = []
    async def timing_out():
        calls.append(1); raise ReadTimeout()
    with pytest.raises(ReadTimeout):
        await with_retry_send(timing_out, attempts=3)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_send_retries_on_503():
    """503 means it definitively did not land, so a resend cannot duplicate."""
    calls = []
    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise HTTPStatusError(503, {"retry-after": "0.05"})
        return "sent"
    assert await with_retry_send(flaky) == "sent"
    assert len(calls) == 2


def test_send_classification_is_stricter_than_generic():
    assert is_retryable(ReadTimeout()) and not is_retryable_send(ReadTimeout())
    assert is_retryable(HTTPStatusError(408)) and not is_retryable_send(HTTPStatusError(408))
    assert is_retryable_send(HTTPStatusError(503))
