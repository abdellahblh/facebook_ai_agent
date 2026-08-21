"""The pipeline: everything between "webhook accepted" and "reply sent".

    dedupe -> handoff check -> media->text -> debounce -> lock
           -> log -> typing -> agent -> send -> log

Media is converted to text BEFORE debounce on purpose: a voice note and the
typed follow-up that comes after it must merge into ONE turn, exactly like two
typed messages do.
"""

from __future__ import annotations

import logging

from app import messenger_api
from app.agent.graph import build_graph, run_turn
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import make_tools
from app.cache import redis_ops
from app.config import get_settings
from app.db import repo
from app.media import media_to_text
from app.schemas import InboundMessage

logger = logging.getLogger(__name__)

FALLBACK_REPLY = "I'm sorry, I couldn't process your request right now."
ERROR_REPLY = "Something went wrong on our end. A teammate will follow up."
HANDOFF_REPLY = "A teammate will reply here shortly."
UNSUPPORTED_REPLY = (
    "I can understand voice messages and photos! "
    "For other files, please describe your request in text."
)


async def process_event(inbound: InboundMessage) -> None:
    """One inbound event, end to end."""
    from app.main import app

    settings = get_settings()
    redis = getattr(app.state, "redis", None)
    session_factory = getattr(app.state, "session_factory", None)
    http_client = getattr(app.state, "http", None)
    llm = getattr(app.state, "llm", None)
    saver = getattr(app.state, "saver", None)

    # ── 1. DEDUPE ────────────────────────────────────────────────────────────
    if redis and inbound.mid:
        if await redis_ops.is_duplicate(redis, inbound.mid):
            logger.info("Duplicate message %s skipped.", inbound.mid)
            return

    # ── 2. HANDOFF CHECK ─────────────────────────────────────────────────────
    if session_factory:
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

    # ── 3a. POSTBACK: explicit "talk to a human" button ──────────────────────
    if inbound.kind == "postback" and inbound.postback_payload == "HANDOFF":
        if session_factory:
            async with session_factory() as session:
                await repo.set_handoff(session, inbound.page_id, inbound.psid, True)
        if http_client:
            await messenger_api.send_text(http_client, inbound.psid, HANDOFF_REPLY)
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
            await messenger_api.send_text(http_client, inbound.psid, UNSUPPORTED_REPLY)
        return

    # ── 4. DEBOUNCE ──────────────────────────────────────────────────────────
    if redis:
        debounced = await redis_ops.buffer_and_wait(
            redis, inbound.psid, merged_text, debounce_seconds=settings.debounce_seconds
        )
        if debounced is None:
            return  # a newer message in the burst owns the flush
        merged_text = debounced.strip()

    if not merged_text:
        return

    # ── 5. LOCK ──────────────────────────────────────────────────────────────
    if redis:
        if not await redis_ops.acquire_user_lock(redis, inbound.psid):
            logger.info("Worker lock for %s already held.", inbound.psid)
            return

    try:
        # ── 6. LOG THE USER MESSAGE ──────────────────────────────────────────
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

        # ── 7. TYPING INDICATOR ──────────────────────────────────────────────
        if http_client:
            await messenger_api.send_typing(http_client, inbound.psid)

        # ── 8. AGENT TURN ────────────────────────────────────────────────────
        reply_text = FALLBACK_REPLY
        if llm and session_factory:
            tools = make_tools(inbound.psid, inbound.page_id, session_factory)
            agent_graph = build_graph(llm, tools, SYSTEM_PROMPT, checkpointer=saver)
            reply_text = await run_turn(agent_graph, merged_text, psid=inbound.psid)
        else:
            logger.error(
                "Agent skipped: llm=%s session_factory=%s — check the lifespan wiring.",
                bool(llm),
                bool(session_factory),
            )

        if not reply_text or not reply_text.strip():
            logger.warning("Agent returned an empty reply for %s.", inbound.psid)
            reply_text = FALLBACK_REPLY

        # ── 9. SEND ──────────────────────────────────────────────────────────
        if http_client:
            await messenger_api.send_text(http_client, inbound.psid, reply_text)

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
                await messenger_api.send_text(http_client, inbound.psid, ERROR_REPLY)
            except Exception:
                logger.exception("Could not even send the error reply to %s.", inbound.psid)
    finally:
        if redis:
            await redis_ops.release_user_lock(redis, inbound.psid)

