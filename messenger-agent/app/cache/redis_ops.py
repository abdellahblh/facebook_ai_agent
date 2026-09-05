"""Redis: dedupe, debounce, per-user locks. YOU implement these.
(Step 4 of the path)

Honest note from the architecture review: we do NOT cache LLM responses here —
support conversations are too varied for useful hit rates. Redis earns its
place on the webhook hot path with these three jobs.

Definition of done: pytest tests/test_redis_ops.py
"""

from __future__ import annotations

import asyncio
import time
import secrets
import redis.asyncio as aioredis

DEDUPE_TTL_SECONDS = 3600
LOCK_TTL_SECONDS = 120  # auto-release: a crashed worker must not lock a user forever
DEBOUNCE_TTL_SECONDS = 120


async def is_duplicate(r: aioredis.Redis, mid: str) -> bool:
    """True if we've already seen this message id. Meta redelivers webhooks —
    without this, every redelivery makes the bot answer the customer twice.
    """
    was_set = await r.set(f"seen:{mid}", "1", nx=True, ex=DEDUPE_TTL_SECONDS)
    return not bool(was_set)


async def buffer_and_wait(r: aioredis.Redis, psid: str, text: str, debounce_seconds: int | float) -> str | None:
    """The debounce — buffer rapid messages, wait for silence, answer ONCE with merged text."""
    buffer_key = f"buf:{psid}"
    last_key = f"last:{psid}"
    await r.rpush(buffer_key, text)
    my_token = str(time.time_ns())
    # Expiry avoids a crashed worker merging a message from yesterday.
    await r.set(last_key, my_token, ex=max(DEBOUNCE_TTL_SECONDS, int(debounce_seconds) * 3))
    await r.expire(buffer_key, max(DEBOUNCE_TTL_SECONDS, int(debounce_seconds) * 3))

    await asyncio.sleep(debounce_seconds)

    current_stamp = await r.get(last_key)
    if isinstance(current_stamp, bytes):
        current_stamp = current_stamp.decode("utf-8")

    if current_stamp != my_token:
        return None

    # Compare/read/delete is one Redis operation. Without it, an arrival
    # between LRANGE and DELETE could be silently erased.
    script = """
    if redis.call('GET', KEYS[2]) ~= ARGV[1] then return false end
    local values = redis.call('LRANGE', KEYS[1], 0, -1)
    redis.call('DEL', KEYS[1], KEYS[2])
    return values
    """
    parts = await r.eval(script, 2, buffer_key, last_key, my_token)
    if not parts:
        return None
    decoded_parts = [p.decode("utf-8") if isinstance(p, bytes) else p for p in parts]
    return " ".join(decoded_parts)


async def acquire_user_lock(r: aioredis.Redis, psid: str) -> bool:
    """One reply pipeline per customer at a time."""
    was_set = await r.set(f"lock:{psid}", "1", nx=True, ex=LOCK_TTL_SECONDS)
    return bool(was_set)


async def acquire_user_lock_token(r: aioredis.Redis, psid: str) -> str | None:
    """Acquire a lock with an ownership token for safe release after TTLs."""
    token = secrets.token_urlsafe(16)
    acquired = await r.set(f"lock:{psid}", token, nx=True, ex=LOCK_TTL_SECONDS)
    return token if acquired else None


async def release_user_lock(r: aioredis.Redis, psid: str, token: str | None = None) -> None:
    """Delete only our lock; never delete one acquired after our TTL expired."""
    if token is None:
        await r.delete(f"lock:{psid}")
        return
    await r.eval(
        "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end return 0",
        1,
        f"lock:{psid}",
        token,
    )
