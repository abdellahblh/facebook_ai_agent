"""The pipeline: everything between "webhook accepted" and "reply sent".

    dedupe -> handoff check -> media->text -> debounce -> lock
           -> log -> typing -> agent -> send -> log

Media is converted to text BEFORE debounce on purpose: a voice note and the
typed follow-up that comes after it must merge into ONE turn, exactly like two
typed messages do.
"""

from __future__ import annotations

import logging

import httpx

from app import messenger_api
from app.agent.graph import build_graph, run_turn
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import make_tools
from app.chatwoot import send_reply, send_typing, escalate_to_human
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


async def _send_text(
    http_client: httpx.AsyncClient,
    inbound: InboundMessage,
    text: str,
) -> None:
    """Route outgoing reply to Chatwoot or Meta Graph API depending on channel."""
    if inbound.channel == "chatwoot":
        from app import chatwoot

        account_id = inbound.account_id if inbound.account_id is not None else 1
        conversation_id = inbound.conversation_id or inbound.psid
        logger.info("Sending Chatwoot reply to account=%s conv=%s: %s", account_id, conversation_id, text[:60])
        await chatwoot.send_reply(
            http_client, account_id, conversation_id, text
        )
    else:
        logger.info("Sending Messenger reply to psid=%s: %s", inbound.psid, text[:60])
        await messenger_api.send_text(http_client, inbound.psid, text)


async def _send_typing(
    http_client: httpx.AsyncClient,
    inbound: InboundMessage,
) -> None:
    """Route typing indicator to Chatwoot or Meta Graph API."""
    if inbound.channel == "chatwoot":
        from app import chatwoot

        account_id = inbound.account_id if inbound.account_id is not None else 1
        conversation_id = inbound.conversation_id or inbound.psid
        await chatwoot.send_typing(
            http_client, account_id, conversation_id
        )
    else:
        await messenger_api.send_typing(http_client, inbound.psid)


async def process_event(inbound: InboundMessage) -> None:
    """One inbound event, end to end."""
    from app.main import app

    convo_key = inbound.conversation_id or inbound.psid
    settings = get_settings()
    redis = getattr(app.state, "redis", None)
    session_factory = getattr(app.state, "session_factory", None)
    http_client = getattr(app.state, "http", None)
    llm = getattr(app.state, "llm", None)
    saver = getattr(app.state, "saver", None)

    # ── 1. DEDUPE ────────────────────────────────────────────────────────────
    if redis and inbound.mid:
        dedupe_id = f"{inbound.channel}:{inbound.page_id}:{inbound.mid}"
        if await redis_ops.is_duplicate(redis, dedupe_id):
            logger.info("Duplicate message %s skipped.", inbound.mid)
            return

    # ── 2. SEND TYPING ─────────────────────────────────────────────────────
    if http_client:
        await _send_typing(http_client, inbound)
        logger.info("Typing indicator sent for message %s.", inbound.mid)

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

    # ── 5. LOCK ──────────────────────────────────────────────────────────────
    lock_token = None
    if redis:
        lock_token = await redis_ops.acquire_user_lock_token(redis, convo_key)
        if lock_token is None:
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
            await _send_typing(http_client, inbound)

        # ── 8. AGENT TURN ────────────────────────────────────────────────────
        reply_text = FALLBACK_REPLY
        agent_graph = getattr(app.state, "agent_graph", None)
        if llm and session_factory and agent_graph:
            thread_id = inbound.conversation_id or inbound.psid
            reply_text = await run_turn(
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
    finally:
        if redis and lock_token:
            await redis_ops.release_user_lock(redis, convo_key, lock_token)
