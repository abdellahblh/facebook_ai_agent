"""Focused tests for the multi-tier cache controller."""

from __future__ import annotations

import asyncio

from app.cache.layers.l0_memory import L0InMemoryCache
from app.cache.layers.l2_redis import L2RedisKVCache, exact_cache_key
from app.cache.manager import CacheManager
from app.cache.schemas import CachePayload, CacheRequest


def _request(
    query: str = "What is the return policy?",
    tenant_id: str = "page-1",
    user_id: str = "customer-1",
) -> CacheRequest:
    return CacheRequest(
        tenant_id=tenant_id,
        user_id=user_id,
        auth_role="user",
        query=query,
    )


async def test_cache_manager_reuses_exact_response(fake_redis):
    manager = CacheManager(
        L0InMemoryCache(max_size=10, default_ttl_seconds=60),
        L2RedisKVCache(fake_redis, default_ttl_seconds=60),
    )
    calls = 0

    async def compute() -> CachePayload:
        nonlocal calls
        calls += 1
        return CachePayload(response="Returns are accepted within 14 days.")

    assert (await manager.get_or_compute(_request(), compute)).response.startswith("Returns")
    assert (await manager.get_or_compute(_request(), compute)).response.startswith("Returns")
    assert calls == 1


async def test_cache_key_isolation_includes_tenant_and_user():
    first = exact_cache_key(_request(tenant_id="page-1", user_id="customer-1"))
    other_tenant = exact_cache_key(_request(tenant_id="page-2", user_id="customer-1"))
    other_user = exact_cache_key(_request(tenant_id="page-1", user_id="customer-2"))
    assert first != other_tenant
    assert first != other_user


async def test_single_flight_computes_once_for_concurrent_requests(fake_redis):
    manager = CacheManager(
        L0InMemoryCache(max_size=10, default_ttl_seconds=60),
        L2RedisKVCache(fake_redis, default_ttl_seconds=60),
        lock_wait_seconds=2,
        poll_interval_seconds=0.01,
    )
    calls = 0

    async def compute() -> CachePayload:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return CachePayload(response="one computation")

    results = await asyncio.gather(
        manager.get_or_compute(_request("same"), compute),
        manager.get_or_compute(_request("same"), compute),
    )
    assert [result.response for result in results] == ["one computation", "one computation"]
    assert calls == 1


async def test_namespace_invalidation_clears_l0_and_l2(fake_redis):
    manager = CacheManager(
        L0InMemoryCache(max_size=10, default_ttl_seconds=60),
        L2RedisKVCache(fake_redis, default_ttl_seconds=60),
    )
    request = _request()
    await manager.get_or_compute(request, lambda: _payload("cached"))
    counts = await manager.invalidate_namespace("agent-response")
    assert counts["L0"] == 1
    assert counts["L2"] == 1
    assert await manager.get(request) is None


async def _payload(response: str) -> CachePayload:
    return CachePayload(response=response)
