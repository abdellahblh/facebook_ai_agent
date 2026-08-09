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
import redis.asyncio as aioredis

DEDUPE_TTL_SECONDS = 3600
LOCK_TTL_SECONDS = 120  # auto-release: a crashed worker must not lock a user forever


async def is_duplicate(r: aioredis.Redis, mid: str) -> bool:
    """True if we've already seen this message id. Meta redelivers webhooks —
    without this, every redelivery makes the bot answer the customer twice.
    """
    was_set = await r.set(f"seen:{mid}", "1", nx=True, ex=DEDUPE_TTL_SECONDS)
    return not bool(was_set)


async def buffer_and_wait(r: aioredis.Redis, psid: str, text: str, debounce_seconds: int | float) -> str | None:
    """The debounce — buffer rapid messages, wait for silence, answer ONCE with merged text."""
    await r.rpush(f"buf:{psid}", text)
    my_token = str(time.time_ns())
    await r.set(f"last:{psid}", my_token)

    await asyncio.sleep(debounce_seconds)

    current_stamp = await r.get(f"last:{psid}")
    if isinstance(current_stamp, bytes):
        current_stamp = current_stamp.decode("utf-8")

    if current_stamp != my_token:
        return None

    parts = await r.lrange(f"buf:{psid}", 0, -1)
    await r.delete(f"buf:{psid}", f"last:{psid}")
    decoded_parts = [p.decode("utf-8") if isinstance(p, bytes) else p for p in parts]
    return " ".join(decoded_parts)


async def acquire_user_lock(r: aioredis.Redis, psid: str) -> bool:
    """One reply pipeline per customer at a time."""
    was_set = await r.set(f"lock:{psid}", "1", nx=True, ex=LOCK_TTL_SECONDS)
    return bool(was_set)


async def release_user_lock(r: aioredis.Redis, psid: str) -> None:
    """Delete the lock key."""
    await r.delete(f"lock:{psid}")

