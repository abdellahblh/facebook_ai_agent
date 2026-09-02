"""Retry for the flaky network calls in the agent pipeline.

WHY NOT A LIBRARY
    tenacity would do this. This is ~80 lines with no dependency, and more
    importantly the two rules below are project-specific — a generic retry
    decorator gets them both wrong by default.

RULE 1 — RETRY ONLY WHAT CAN SUCCEED
    429, 500, 502, 503, 504, timeouts and connection resets are transient.
    400, 401, 403, 404 and 422 are verdicts: the same request will be refused
    identically forever. Retrying them turns one fast error into four slow
    ones while a customer waits.

    This is the mistake that hurt earlier in this project: the empty
    CHATWOOT_API_TOKEN produced a 403. A blind retry would have made that a
    12-second silence instead of an instant failure, and hidden the cause.

RULE 2 — RETRIES ARE NOT FREE WHEN THE CALL HAS A SIDE EFFECT
    Retrying `generate_reply` is safe: worst case you spend tokens twice.
    Retrying `send_reply` is NOT: if the first POST reached Chatwoot and only
    the RESPONSE was lost, the customer already has the message and the retry
    sends it twice. Read the note on TIMEOUT_IS_AMBIGUOUS below before
    wrapping any send.

WHY THE BUDGET IS SHORT
    The webhook already answered 200, so nothing upstream is blocked — but a
    human is looking at a chat window. 8 seconds of retrying then a graceful
    "let me check with a colleague" beats 60 seconds of silence followed by a
    perfect answer.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Transient HTTP statuses. 429 included: providers mean "slow down", not "no".
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Matched on class NAME, not by importing every SDK. google-genai, httpx,
# openai and aiohttp all have their own exception trees; string matching keeps
# this module dependency-free and survives an SDK swap.
RETRYABLE_EXC_NAMES = (
    "Timeout", "TimeoutError", "TimeoutException", "ReadTimeout",
    "ConnectTimeout", "ConnectError", "ConnectionError", "ConnectionReset",
    "RemoteProtocolError", "ServiceUnavailable", "ResourceExhausted",
    "InternalServerError", "DeadlineExceeded", "ServerError", "Unavailable",
)


def _status_of(exc: BaseException) -> int | None:
    """Dig a status code out of whatever the SDK threw.

    Four shapes seen in practice: httpx.HTTPStatusError.response.status_code,
    google.genai errors' .code, a bare .status_code, and google.api_core's
    .grpc_status_code. Checked in order, first hit wins.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    for attr in ("status_code", "code", "status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    return None


def _retry_after(exc: BaseException) -> float | None:
    """Honour a Retry-After header when the provider sends one.

    Gemini and Meta both do under rate limiting. Their number is better than
    our guess: it reflects when the quota window actually rolls over.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        # Seconds form only. The HTTP-date form is legal but rare, and
        # mis-parsing a date into a huge sleep is worse than ignoring it.
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if 0 < value <= 30 else None


def is_retryable(exc: BaseException) -> bool:
    """True for transient failures only."""
    status = _status_of(exc)
    if status is not None:
        # An explicit status is authoritative: a 403 is never retryable even
        # if its exception class happens to be named ...ConnectionError.
        return status in RETRYABLE_STATUS
    name = type(exc).__name__
    return any(marker in name for marker in RETRYABLE_EXC_NAMES)


async def with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    budget: float = 8.0,
    label: str = "call",
) -> T:
    """Await func(), retrying transient failures with jittered backoff.

    Raises the LAST exception when every attempt fails, so the caller's
    existing error handling still sees a real exception rather than None.

    JITTER IS NOT DECORATION: the debounce window means several customers can
    fire at the same instant. Without jitter their retries stay in lockstep and
    hit the provider together, reproducing the 429 that caused the retry.

    BUDGET is wall-clock across all attempts. Backoff alone can overshoot
    badly: 3 attempts at 0.5/1/2s looks like 3.5s but each attempt may itself
    take 10s to time out.
    """
    started = time.monotonic()
    last: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if not is_retryable(exc):
                logger.warning(
                    "%s failed permanently (%s: %s) — not retrying",
                    label, type(exc).__name__, exc,
                )
                raise
            if attempt == attempts:
                break

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay = _retry_after(exc) or delay * (0.5 + random.random())

            elapsed = time.monotonic() - started
            if elapsed + delay > budget:
                logger.warning(
                    "%s out of retry budget after %.1fs (attempt %d/%d)",
                    label, elapsed, attempt, attempts,
                )
                break

            logger.info(
                "%s attempt %d/%d failed (%s) — retrying in %.2fs",
                label, attempt, attempts, type(exc).__name__, delay,
            )
            await asyncio.sleep(delay)

    assert last is not None
    raise last


# ── the send problem ─────────────────────────────────────────────────────────
# TIMEOUT_IS_AMBIGUOUS
#     A timeout on a POST does not tell you whether the server processed it.
#     Retrying a send after a timeout is how customers get the same reply
#     twice — the exact duplicate-message bug that a blocking webhook caused
#     earlier in this project, arriving by a different route.
#
#     So sends retry on statuses only (a 503 means it definitively did not
#     land) and NOT on timeouts. One duplicate reply damages trust more than
#     one missing reply, because a missing reply still has the handoff path.
SEND_SAFE_STATUS = frozenset({429, 500, 502, 503, 504})


def is_retryable_send(exc: BaseException) -> bool:
    """Stricter rule for calls that create something customer-visible."""
    status = _status_of(exc)
    return status in SEND_SAFE_STATUS if status is not None else False


async def with_retry_send(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    label: str = "send",
) -> T:
    """Retry a side-effecting call. Statuses only, never timeouts."""
    started = time.monotonic()
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not is_retryable_send(exc) or attempt == attempts:
                if not is_retryable_send(exc):
                    logger.warning(
                        "%s not retried (%s) — may have already been delivered",
                        label, type(exc).__name__,
                    )
                break
            delay = (_retry_after(exc) or 1.0) * (0.5 + random.random())
            if time.monotonic() - started + delay > 5.0:
                break
            logger.info("%s attempt %d/%d failed — retrying in %.2fs",
                        label, attempt, attempts, delay)
            await asyncio.sleep(delay)
    assert last is not None
    raise last


__all__ = [
    "with_retry",
    "with_retry_send",
    "is_retryable",
    "is_retryable_send",
    "RETRYABLE_STATUS",
]
