"""The pipeline: everything between "webhook accepted" and "reply sent".

    dedupe -> handoff check -> media->text -> debounce -> lock
           -> embedding guardrail -> log/cache/agent -> send -> log

Media is converted to text BEFORE debounce on purpose: a voice note and the
typed follow-up that comes after it must merge into ONE turn, exactly like two
typed messages do.
"""

from __future__ import annotations

import asyncio
import logging
from time import monotonic

import httpx
import re
from app import messenger_api
from app.agent.graph import run_turn
from app.cache import redis_ops, streams
from app.cache.schemas import CachePayload, CacheRequest
from app.config import get_settings
from app.db import repo
from app.media import media_to_text
from app.schemas import InboundMessage

logger = logging.getLogger(__name__)

FALLBACK_REPLY = "I'm sorry, I couldn't process your request right now."
ERROR_REPLY = "Something went wrong on our end. A teammate will follow up."
HANDOFF_REPLY = "A teammate will reply here shortly."
INPUT_GUARDRAIL_REPLY = "I can help with customer-support questions, but I can't help with that request."
UNSUPPORTED_REPLY = (
    "I can understand voice messages and photos! "
    "For other files, please describe your request in text."
)
GREETING_REPLIES = {
    "ar": "وعليكم السلام، مرحبا بيك! واش نقدر نعاونك اليوم؟ 😊",
    "en": "Hello! How can I help you today?",
}
_GREETING_RE_AR = re.compile(
    r"^\s*سلام(?:\s+عليكم)?(?:\s+ورحمة\s+الله(?:\s+وبركاته)?)?\s*[!.؟?]*\s*$"
)
_GREETING_RE_EN = re.compile(
    r"^\s*(hi+|hello+|hey+)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


async def _send_text(
    http_client: httpx.AsyncClient,
    inbound: InboundMessage,
    text: str,
) -> None:
    """Route outgoing reply to Chatwoot or Meta Graph API depending on channel."""
    if inbound.channel == "chatwoot":
        from app import chatwoot

        account_id = inbound.account_id
        conversation_id = inbound.conversation_id or inbound.psid
        logger.info("Sending Chatwoot reply to account=%s conv=%s: %s", account_id, conversation_id, text[:60])
        await chatwoot.send_reply(
            http_client, account_id, conversation_id, text
        )
    else:
        logger.info("Sending Messenger reply to psid=%s: %s", inbound.psid, text[:60])
        await messenger_api.send_text(http_client, inbound.psid, text)




def try_handle_greeting(text: str) -> str | None:
    """
    Returns:
        "ar" or "en" if this message is a greeting,
        None if it's not a greeting at all — caller should proceed to the LLM/agent as normal.
    """
    stripped = text.strip()
    if _GREETING_RE_AR.match(stripped):
        return "ar"
    elif _GREETING_RE_EN.match(stripped):
        return "en"
    return None  

async def _send_typing(
    http_client: httpx.AsyncClient,
    inbound: InboundMessage,
) -> None:
    """Route typing indicator to Chatwoot or Meta Graph API."""
    if inbound.channel == "chatwoot":
        from app import chatwoot

        account_id = inbound.account_id 
        conversation_id = inbound.conversation_id or inbound.psid
        await chatwoot.send_typing(
            http_client, account_id, conversation_id
        )
    else:
        await messenger_api.send_typing(http_client, inbound.psid)


async def process_event(inbound: InboundMessage, dedupe: bool = False) -> None:
    """One inbound event, end to end.

    `dedupe=True` is only used by the QUEUE_ENABLED=false escape hatch (direct
    in-process tasks) where the webhook does not do producer-side dedupe. The
    queue consumer keeps dedupe False — that path is guarded by the producer's
    `seen:` SETNX and the consumer's `sent:` flag, and a reclaimed entry MUST
    be reprocessed even though `seen:` is already set.
    """
    from app.main import app

    convo_key = inbound.conversation_id or inbound.psid
    settings = get_settings()
    redis = getattr(app.state, "redis", None)
    session_factory = getattr(app.state, "session_factory", None)
    http_client = getattr(app.state, "http", None)
    llm = getattr(app.state, "llm", None)
    saver = getattr(app.state, "saver", None)

    # ── 1. DEDUPE ────────────────────────────────────────────────────────────
    # Only used by the QUEUE_ENABLED=false escape hatch. Uses the same
    # seen:{channel}:{page_id}:{mid} key as streams.seen_key so a mid-flight
    # toggle from queue to direct (or vice versa) shares one dedupe regime.
    if dedupe and redis:
        if await redis_ops.is_duplicate(redis, inbound.channel, inbound.page_id, inbound.mid):
            logger.info("Duplicate message %s skipped.", inbound.mid)
            return

#if i get off topic question i should block it and answer with "I'm sorry, I can only answer questions about company policies or products. Please ask a question that is on topic."

    # ── 3. HANDOFF CHECK ─────────────────────────────────────────────────────
    # For Meta, customers.handoff_active in PostgreSQL is the source of truth.
    # For Chatwoot, handoff is conversation-scoped and managed in should_process().
    if session_factory and inbound.channel != "chatwoot":
        async with session_factory() as session:
            customer = await repo.get_or_create_customer(
                session, inbound.page_id, inbound.psid
            )
            if customer.handoff_active:
                logged_text = inbound.text
                if not logged_text and inbound.attachment_types:
                    kinds = ", ".join(inbound.attachment_types)
                    logged_text = f"[customer sent attachment(s): {kinds}]"
                if logged_text:
                    await repo.save_message(
                        session,
                        inbound.page_id,
                        inbound.psid,
                        "user",
                        logged_text,
                        meta={
                            "attachment_types": inbound.attachment_types,
                            "during_handoff": True,
                        },
                    )
                logger.info(
                    "Handoff active for %s. Message logged without AI reply.", inbound.psid
                )
                return

    # ── 4. POSTBACK: explicit "talk to a human" button ──────────────────────
    if inbound.kind == "postback" and inbound.postback_payload == "HANDOFF":
        if session_factory:
            async with session_factory() as session:
                await repo.set_handoff(session, inbound.page_id, inbound.psid, True)
        if (
            inbound.channel == "chatwoot"
            and inbound.account_id
            and inbound.conversation_id
            and http_client
        ):
            from app import chatwoot

            await chatwoot.escalate_to_human(
                http_client,
                inbound.account_id,
                inbound.conversation_id,
                note="Customer requested a human via HANDOFF postback.",
            )
        if session_factory and inbound.channel != "chatwoot":
            async with session_factory() as session:
                await repo.set_handoff(session, inbound.page_id, inbound.psid, True)
        if http_client:
            await _send_text(http_client, inbound, HANDOFF_REPLY)
        return

    # ── 3b. MEDIA -> TEXT ────────────────────────────────────────────────────
    media_parts: list[str] = []
    if inbound.media and http_client and llm:
        for attachment_type, attachment_url in inbound.media:
            fragment = await media_to_text(llm, http_client, attachment_type, attachment_url)
            if fragment:
                media_parts.append(fragment)

    text_parts: list[str] = []
    if inbound.text:
        text_parts.append(inbound.text)
    text_parts.extend(media_parts)
    merged_text = "\n".join(text_parts).strip()

    if not merged_text:
        if inbound.attachment_types and http_client:
            await _send_text(http_client, inbound, UNSUPPORTED_REPLY)
        return

    # ── 4. DEBOUNCE ──────────────────────────────────────────────────────────
    if redis:
        debounced = await redis_ops.buffer_and_wait(
            redis,convo_key, merged_text, debounce_seconds=settings.debounce_seconds
        )
        if debounced is None:
            return  # a newer message in the burst owns the flush
        merged_text = debounced.strip()

    if not merged_text:
        return
    if http_client:
        await _send_typing(http_client, inbound)
        logger.info("Typing indicator sent for message %s.", inbound.mid)

    # ── 5. LOCK ──────────────────────────────────────────────────────────────
    lock_token = None
    if redis:
        lock_token = await redis_ops.acquire_user_lock_token(redis, convo_key)
        if lock_token is None:
            logger.info("Worker lock for %s already held.", inbound.psid)
            return

    try:
        greeting_lang = try_handle_greeting(merged_text)
        if greeting_lang and greeting_lang in GREETING_REPLIES:
            greeting_reply = GREETING_REPLIES[greeting_lang]
            if http_client:
                await _send_text(http_client, inbound, greeting_reply)
            if session_factory:
                async with session_factory() as session:
                    await repo.save_message(
                        session, inbound.page_id, inbound.psid, "user", merged_text
                    )
                    await repo.save_message(
                        session, inbound.page_id, inbound.psid, "assistant", greeting_reply
                    )
            return

        # ── 6. TOPIC CONTROL GUARDRAIL ───────────────────────────────────────
        reply_text = FALLBACK_REPLY
        agent_graph = getattr(app.state, "agent_graph", None)
        cache_manager = getattr(app.state, "cache_manager", None)
        inbound_guardrail = getattr(app.state, "inbound_guardrail", None)
        if inbound_guardrail is None:
            raise RuntimeError("Topic control input guardrail is unavailable")

        decision = await inbound_guardrail.inspect(merged_text)
        logger.info(
            "Topic guardrail inspection for %s: blocked=%s similarity=%.3f example=%s",
            inbound.mid,
            decision.blocked,
            decision.similarity or 0.0,
            decision.example_id,
        )
        if decision.blocked:
            logger.warning(
                "Topic control guardrail blocked inbound message %s (similarity=%.3f, example=%s).",
                inbound.mid,
                decision.similarity or 0.0,
                decision.example_id,
            )
            guard = app.state.guard_model
            result = await guard.check(merged_text)
            if not result.is_safe:       
                reply_text = INPUT_GUARDRAIL_REPLY
                if session_factory:
                    async with session_factory() as session:
                        await repo.save_message(
                            session,
                            inbound.page_id,
                            inbound.psid,
                            "user",
                            merged_text,
                            meta={
                                "guardrail_violation": True,
                                "guardrail_type": "topic_control",
                                "guardrail_similarity": decision.similarity,
                                "guardrail_example_id": decision.example_id,
                                "attachment_types": inbound.attachment_types,
                            },
                        )
            if http_client:
                await _send_text(http_client, inbound, reply_text)
            if session_factory:
                async with session_factory() as session:
                    await repo.save_message(
                        session, inbound.page_id, inbound.psid, "assistant", reply_text
                    )
            return

        # ── 7. LOG SAFE USER MESSAGE ─────────────────────────────────────────
        if session_factory:
            async with session_factory() as session:
                await repo.save_message(
                    session,
                    inbound.page_id,
                    inbound.psid,
                    "user",
                    merged_text,
                    meta={"attachment_types": inbound.attachment_types},
                )

        # ── 8. AGENT TURN / SEMANTIC CACHE ───────────────────────────────────
        if cache_manager and llm and session_factory and agent_graph:
            thread_id = inbound.conversation_id or inbound.psid
            cache_request = CacheRequest(
                tenant_id=str(inbound.page_id),
                user_id=str(convo_key),
                auth_role="user",
                query=merged_text,
                metadata={"channel": inbound.channel},
            )

            async def compute_reply() -> CachePayload:
                generated = await run_turn(
                    agent_graph,
                    merged_text,
                    psid=inbound.psid,
                    thread_id=thread_id,
                    config_context={
                        "page_id": inbound.page_id,
                        "psid": inbound.psid,
                        "channel": inbound.channel,
                        "account_id": inbound.account_id,
                        "conversation_id": inbound.conversation_id,
                        "redis": redis,
                        "embedder": app.state.embedder,
                    },
                )
                if not generated or not generated.strip():
                    raise RuntimeError("Agent returned an empty response")
                return CachePayload(response=generated, source="agent")

            payload = await cache_manager.get_or_compute(
                cache_request, compute_reply, embedding=decision.embedding
            )
            reply_text = payload.response
        elif llm and session_factory and agent_graph:
            reply_text = await run_turn(
                agent_graph,
                merged_text,
                psid=inbound.psid,
                thread_id=inbound.conversation_id or inbound.psid,
                config_context={
                    "page_id": inbound.page_id,
                    "psid": inbound.psid,
                    "channel": inbound.channel,
                    "account_id": inbound.account_id,
                    "conversation_id": inbound.conversation_id,
                    "redis": redis,
                    "embedder": app.state.embedder
                },
            )
        else:
            logger.error(
                "Agent skipped: llm=%s session_factory=%s graph=%s — check the lifespan wiring.",
                bool(llm),
                bool(session_factory),
                bool(agent_graph),
            )

        if not reply_text or not reply_text.strip():
            logger.warning("Agent returned an empty reply for %s.", inbound.psid)
            reply_text = FALLBACK_REPLY

        # ── 9. SEND ──────────────────────────────────────────────────────────
        if http_client:
            await _send_text(http_client, inbound, reply_text)

        # ── 10. LOG THE ASSISTANT MESSAGE ────────────────────────────────────
        if session_factory:
            async with session_factory() as session:
                await repo.save_message(
                    session, inbound.page_id, inbound.psid, "assistant", reply_text
                )

    except Exception as err:
        logger.exception("Error processing message for %s: %s", inbound.psid, err)
        if http_client:
            try:
                await _send_text(http_client, inbound, ERROR_REPLY)
            except Exception:
                logger.exception("Could not even send the error reply to %s.", inbound.psid)
            if inbound.channel == "chatwoot" and inbound.account_id and inbound.conversation_id:
                from app import chatwoot

                note_ok = await chatwoot.send_reply(
                    http_client,
                    inbound.account_id,
                    inbound.conversation_id,
                    f"Bot error processing message: {type(err).__name__}. "
                    f"A human teammate will review. Reason: {str(err)[:200]}",
                    private=True,
                )
                if not note_ok:
                    logger.warning(
                        "Failed to send error private note for conversation %s",
                        inbound.conversation_id,
                    )
    finally:
        if redis and lock_token:
            await redis_ops.release_user_lock(redis, convo_key, lock_token)


# ── queue consumer ───────────────────────────────────────────────────────────
# Bounded concurrency: entries are claimed in batches of 16, but a turn can take
# 10-30s (debounce sleep + LLM). Processing them one-at-a-time serializes the
# whole batch behind the slowest turn — so we fan out with a semaphore cap.
# SBFLOCK: the per-user `lock:{psid}` key keeps two turns for the SAME customer
# from running concurrently; this semaphore only bounds total in-flight work.
QUEUE_MAX_CONCURRENT_TURNS = 16

# How often the consumer polls XAUTOCLAIM for orphaned entries. This is the
# recovery cadence, NOT the idle threshold — the actual "is it dead?" threshold
# is Queue.reclaim_min_idle_s from settings (default 180s), which must exceed
# worst-case turn duration so a slow-but-alive consumer is never tripped.
QUEUE_RECLAIM_POLL_S = 10.0


async def _record_failure(
    redis,
    session_factory,
    stream: str,
    group: str,
    entry_id: str,
    payload: str | None,
    error: str,
) -> None:
    """Bump the per-entry attempt counter; ACK + dead-letter once it exceeds
    queue_max_attempts so an always-failing entry cannot grow the stream."""
    settings = get_settings()
    attempts = await streams.bump_attempts(redis, stream, entry_id)
    if attempts > settings.queue_max_attempts:
        await streams.ack(redis, group, entry_id, stream=stream)
        logger.error(
            "Poison entry %s exceeded %s attempts — dead-lettering.",
            entry_id,
            settings.queue_max_attempts,
        )
        if session_factory:
            try:
                async with session_factory() as session:
                    await repo.save_dead_letter(session, payload or "", error)
            except Exception:
                logger.exception("Failed to dead-letter entry %s.", entry_id)


async def _process_queued(
    redis,
    session_factory,
    stream: str,
    group: str,
    entry_id: str,
    payload: str | None,
) -> None:
    """Process one stream entry; XACK only once the message is fully handled."""
    try:
        inbound = InboundMessage.model_validate_json(payload)
    except Exception:
        logger.error("Malformed queue payload for entry %s — dead-lettering.", entry_id)
        await streams.ack(redis, group, entry_id, stream=stream)
        if session_factory:
            try:
                async with session_factory() as session:
                    await repo.save_dead_letter(session, payload or "", "malformed InboundMessage")
            except Exception:
                logger.exception("Failed to dead-letter malformed entry %s.", entry_id)
        return

    # Idempotency: a crash between send and XACK must not double-reply.
    if await streams.is_sent(redis, inbound):
        await streams.ack(redis, group, entry_id, stream=stream)
        logger.info(
            "Entry %s already answered (%s) — acknowledging without reply.",
            entry_id,
            inbound.mid,
        )
        return

    await process_event(inbound)

    await streams.mark_sent(redis, inbound)
    await streams.clear_attempts(redis, stream, entry_id)
    await streams.ack(redis, group, entry_id, stream=stream)


async def _claim_and_process(
    redis, session_factory, stream: str, group: str, consumer: str
) -> None:
    """Reclaim and replay entries orphaned by a crashed/restarted consumer.

    Entries are only claimed after they have been idle for
    settings.queue_reclaim_min_idle_s (default 180s) — well above worst-case
    turn duration — so a live (but slow) consumer is never tripped by its own
    in-flight work."""
    settings = get_settings()
    try:
        reclaimed = await streams.claim_pending(
            redis,
            group,
            consumer,
            stream=stream,
            min_idle_ms=int(settings.queue_reclaim_min_idle_s * 1000),
            count=50,
        )
    except Exception as exc:
        logger.debug("XAUTOCLAIM unavailable (%s) — skipping reclaim.", exc)
        return
    for entry_id, payload in reclaimed:
        try:
            await _process_queued(redis, session_factory, stream, group, entry_id, payload)
        except Exception as exc:
            logger.exception("Reclaimed entry %s failed.", entry_id)
            await _record_failure(redis, session_factory, stream, group, entry_id, payload, str(exc))


async def _log_backlog(redis, stream: str, group: str) -> None:
    try:
        logger.info("Queue backlog: %s", await streams.backlog(redis, stream=stream))
    except Exception:
        logger.debug("Queue stats unavailable.", exc_info=True)


async def run_queue_consumer(
    redis,
    session_factory,
    consumer_name: str = "",
) -> None:
    """Drain the webhook stream in-process. Runs until cancelled. Blocking
    reads idle the task between bursts; the caller cancels it in shutdown
    BEFORE closing Redis/DB/HTTP so no acknowledged work is in flight.

    Entries are fanned out into concurrent tasks (bounded by a semaphore) so
    a slow turn for one customer does not block everyone behind them in the
    same batch."""
    settings = get_settings()
    stream = settings.redis_stream
    group = settings.redis_stream_group
    consumer = consumer_name or f"default-{id(redis)}"

    # Reclaim cadence is independent of the idle threshold: we poll XAUTOCLAIM
    # every 10s but only claim truly orphaned (idle > reclaim_min_idle_s) entries.
    reclaim_interval_s = QUEUE_RECLAIM_POLL_S

    await streams.ensure_consumer_group(redis, stream, group)

    semaphore = asyncio.Semaphore(QUEUE_MAX_CONCURRENT_TURNS)

    async def _process_with_semaphore(entry_id: str, payload: str | None) -> None:
        async with semaphore:
            try:
                await _process_queued(
                    redis, session_factory, stream, group, entry_id, payload
                )
            except Exception as exc:
                logger.exception("Queue entry %s failed.", entry_id)
                await _record_failure(
                    redis, session_factory, stream, group, entry_id, payload, str(exc)
                )

    # Track in-flight tasks so shutdown can await them before closing clients.
    in_flight: set[asyncio.Task] = set()

    def _spawn(entry_id: str, payload: str | None) -> None:
        task = asyncio.create_task(_process_with_semaphore(entry_id, payload))
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)

    last_claim = monotonic()
    last_stats = monotonic()
    while True:
        try:
            # Cloud Redis: shorter block_ms (500ms) to avoid network timeout issues.
            # Local Redis can use 2000ms. The shorter block just means more poll cycles.
            entries = await streams.read_next(
                redis, group, consumer, stream=stream, count=16, block_ms=500
            )
            for entry_id, payload in entries:
                _spawn(entry_id, payload)

            now = monotonic()
            if now - last_claim >= reclaim_interval_s:
                last_claim = now
                await _claim_and_process(redis, session_factory, stream, group, consumer)
            if now - last_stats >= 60:
                last_stats = now
                await _log_backlog(redis, stream, group)
        except asyncio.CancelledError:
            # Drain in-flight entries before re-raising, so the caller's
            # "cancel then wait" (main.py shutdown) finishes cleanly.
            if in_flight:
                await asyncio.gather(*list(in_flight), return_exceptions=True)
            raise
        except Exception as exc:
            logger.error("Queue consumer loop iteration failed: %s", exc)
            await asyncio.sleep(2)
