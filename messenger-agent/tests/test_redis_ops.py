"""Step 4: Redis ops on fakeredis — no server needed."""

import asyncio

from app.cache.redis_ops import acquire_user_lock, buffer_and_wait, is_duplicate, release_user_lock


async def test_first_time_is_not_duplicate(fake_redis):
    assert await is_duplicate(fake_redis, "m_1") is False


async def test_second_time_is_duplicate(fake_redis):
    await is_duplicate(fake_redis, "m_1")
    assert await is_duplicate(fake_redis, "m_1") is True


async def test_different_mids_independent(fake_redis):
    await is_duplicate(fake_redis, "m_1")
    assert await is_duplicate(fake_redis, "m_2") is False


async def test_debounce_merges_burst_into_one_flush(fake_redis):
    """3 rapid messages -> exactly ONE flush containing all three, owned by the last."""
    results = await asyncio.gather(
        buffer_and_wait(fake_redis, "PSID", "hi", 1),
        _delayed(0.1, buffer_and_wait(fake_redis, "PSID", "i want to ask", 1)),
        _delayed(0.2, buffer_and_wait(fake_redis, "PSID", "about the red jacket", 1)),
    )
    flushes = [r for r in results if r is not None]
    assert len(flushes) == 1
    assert flushes[0] == "hi i want to ask about the red jacket"
    assert results[0] is None and results[1] is None  # earlier messages ceded ownership


async def test_single_message_flushes_itself(fake_redis):
    assert await buffer_and_wait(fake_redis, "PSID", "just one question", 1) == "just one question"


async def test_lock_is_exclusive_then_releasable(fake_redis):
    assert await acquire_user_lock(fake_redis, "PSID") is True
    assert await acquire_user_lock(fake_redis, "PSID") is False  # held
    await release_user_lock(fake_redis, "PSID")
    assert await acquire_user_lock(fake_redis, "PSID") is True  # free again


async def _delayed(seconds: float, coro):
    await asyncio.sleep(seconds)
    return await coro
