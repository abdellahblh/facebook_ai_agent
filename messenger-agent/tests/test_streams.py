"""Queue layer tests: producer enqueue + dedupe, consumer read/ack/reclaim,
sent-flag idempotency, attempt counting, and backlog metrics. Runs against
fakeredis (decode_responses=True like the app's CI fixture)."""

from __future__ import annotations

import pytest

from app.cache import streams
from app.config import get_settings
from app.schemas import InboundMessage


def _msg(mid="m1", channel="chatwoot", **kwargs) -> InboundMessage:
    kwargs.setdefault("kind", "text")
    kwargs.setdefault("psid", "user-1")
    kwargs.setdefault("text", "hello")
    return InboundMessage(channel=channel, page_id="page-1", mid=mid, **kwargs)


@pytest.mark.asyncio
async def test_producer_dedupe_first_delivery_only(fake_redis):
    msg = _msg()
    assert await streams.try_reserve(msg, fake_redis) is True
    assert await streams.try_reserve(msg, fake_redis) is False
    # different channel/page id -> independent promise
    other = _msg(channel="messenger")
    assert await streams.try_reserve(other, fake_redis) is True


@pytest.mark.asyncio
async def test_producer_allows_missing_mid(fake_redis):
    # A message without a mid cannot be deduplicated; always allow enqueue.
    assert await streams.try_reserve(_msg(mid=None), fake_redis) is True


@pytest.mark.asyncio
async def test_enqueue_read_round_trip(fake_redis):
    await streams.ensure_consumer_group(fake_redis)
    msg = _msg(mid="m-roundtrip")
    entry_id = await streams.enqueue(fake_redis, msg)
    assert entry_id

    entries = await streams.read_next(
        fake_redis,
        get_settings().redis_stream_group,
        "test-consumer",
        count=16,
        block_ms=50,
    )
    assert len(entries) == 1
    got_id, payload = entries[0]
    assert got_id == entry_id
    parsed = InboundMessage.model_validate_json(payload)
    assert parsed.mid == "m-roundtrip"
    assert parsed.channel == "chatwoot"


@pytest.mark.asyncio
async def test_ack_removes_from_pending(fake_redis):
    await streams.ensure_consumer_group(fake_redis)
    settings = get_settings()
    await streams.enqueue(fake_redis, _msg("m-ack"))
    (entry_id, _payload) = (
        await streams.read_next(
            fake_redis, settings.redis_stream_group, "test-consumer", block_ms=50
        )
    )[0]

    assert await streams.ack(fake_redis, settings.redis_stream_group, entry_id) == 1
    # reclaimed = nothing left pending
    pending = await streams.claim_pending(
        fake_redis, settings.redis_stream_group, "reclaimer", min_idle_ms=0, count=10
    )
    assert pending == []


@pytest.mark.asyncio
async def test_unacked_entry_is_reclaimable(fake_redis):
    await streams.ensure_consumer_group(fake_redis)
    settings = get_settings()
    await streams.enqueue(fake_redis, _msg("m-orphan"))
    (entry_id, _payload) = (
        await streams.read_next(
            fake_redis, settings.redis_stream_group, "dead-consumer", block_ms=50
        )
    )[0]

    # consumer "died" without acking -> a survivor reclaims the entry
    reclaimed = await streams.claim_pending(
        fake_redis, settings.redis_stream_group, "survivor", min_idle_ms=0, count=10
    )
    assert [eid for eid, _ in reclaimed] == [entry_id]


@pytest.mark.asyncio
async def test_sent_flag_guard(fake_redis):
    msg = _msg("m-sent")
    assert await streams.is_sent(fake_redis, msg) is False
    await streams.mark_sent(fake_redis, msg)
    assert await streams.is_sent(fake_redis, msg) is True
    # no mid -> never "sent"
    assert await streams.is_sent(fake_redis, _msg(mid=None)) is False


@pytest.mark.asyncio
async def test_attempts_count_and_clear(fake_redis):
    await streams.ensure_consumer_group(fake_redis)
    settings = get_settings()
    await streams.enqueue(fake_redis, _msg("m-poison"))
    (entry_id, _payload) = (
        await streams.read_next(
            fake_redis, settings.redis_stream_group, "test-consumer", block_ms=50
        )
    )[0]

    assert await streams.bump_attempts(fake_redis, settings.redis_stream_group, entry_id) == 1
    assert await streams.bump_attempts(fake_redis, settings.redis_stream_group, entry_id) == 2
    await streams.clear_attempts(fake_redis, settings.redis_stream_group, entry_id)
    assert await streams.bump_attempts(fake_redis, settings.redis_stream_group, entry_id) == 1


@pytest.mark.asyncio
async def test_success_path_sets_sent_and_acks(fake_redis):
    """End-to-end shape the consumer uses: enqueue -> read -> mark sent -> ack."""
    await streams.ensure_consumer_group(fake_redis)
    settings = get_settings()
    msg = _msg("m-flow")

    await streams.enqueue(fake_redis, msg)
    (entry_id, payload) = (
        await streams.read_next(
            fake_redis, settings.redis_stream_group, "test-consumer", block_ms=50
        )
    )[0]
    parsed = InboundMessage.model_validate_json(payload)
    assert parsed.mid == msg.mid

    await streams.mark_sent(fake_redis, parsed)
    await streams.ack(fake_redis, settings.redis_stream_group, entry_id)

    # a redelivered duplicate webhook is now "already answered": is_sent blocks reply
    redelivery = _msg("m-flow")
    assert await streams.is_sent(fake_redis, redelivery) is True


@pytest.mark.asyncio
async def test_backlog_reports_length_and_pending(fake_redis):
    await streams.ensure_consumer_group(fake_redis)
    stats = await streams.backlog(fake_redis)
    assert stats["stream_length"] == 0
    assert stats["group_pending"] == 0
    assert stats["redis_available"] is True

    await streams.enqueue(fake_redis, _msg("m-backlog"))
    stats = await streams.backlog(fake_redis)
    assert stats["stream_length"] == 1
    assert stats["group_pending"] == 0  # enqueued but not yet delivered

    # delivered to a consumer and not ACKed -> pending
    await streams.read_next(
        fake_redis, get_settings().redis_stream_group, "test-consumer", block_ms=50
    )
    stats = await streams.backlog(fake_redis)
    assert stats["stream_length"] == 1
    assert stats["group_pending"] == 1
    assert stats["redis_available"] is True
