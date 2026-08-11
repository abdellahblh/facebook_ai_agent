"""The pipeline: everything between "webhook accepted" and "reply sent".
YOU implement this LAST — it composes every piece you built before it.
(Step 8 of the path)

    dedupe → handoff check → debounce → lock → log → agent → send → log
"""

from __future__ import annotations

import logging

from app import messenger_api
from app.agent.graph import build_graph, run_turn
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import make_tools
from app.cache import redis_ops
from app.db import repo
from app.schemas import InboundMessage

logger = logging.getLogger(__name__)



async def process_event(inbound: InboundMessage) -> None:
    """One inbound event, end to end.

    Pipeline sequence:
      1. Dedupe (is_duplicate)
      2. Handoff check (customer.handoff_active)
      3. Special content (postbacks / attachments)
      4. Debounce (buffer_and_wait)
      5. Lock (acquire_user_lock)
      6-10. Log user message -> Typing indicator -> Agent turn -> Send reply -> Log assistant message
    """
    from app.main import app

    redis = getattr(app.state, "redis", None)
    session_factory = getattr(app.state, "session_factory", None)
    http_client = getattr(app.state, "http", None)
    llm = getattr(app.state, "llm", None)
    saver = getattr(app.state, "saver", None)

    # 1. DEDUPE
    if redis and inbound.mid:
        if await redis_ops.is_duplicate(redis, inbound.mid):
            logger.info("Duplicate message %s skipped.", inbound.mid)
            return

    # 2. HANDOFF CHECK
    if session_factory:
        async with session_factory() as session:
            customer = await repo.get_or_create_customer(session, inbound.page_id, inbound.psid)
            if customer.handoff_active:
                if inbound.text or inbound.attachment_urls:
                    await repo.save_message(session, inbound.page_id, inbound.psid, "user", inbound.text)
                logger.info("Handoff active for %s. Message logged without AI reply.", inbound.psid)
                return

    # 3. SPECIAL CONTENT (Postback / Media)
    if inbound.kind == "postback" and inbound.postback_payload == "HANDOFF":
        if session_factory:
            async with session_factory() as session:
                await repo.set_handoff(session, inbound.page_id, inbound.psid, True)
        if http_client:
            await messenger_api.send_text(http_client, inbound.psid, "A teammate will reply here shortly.")
        return

    if inbound.attachment_urls and not inbound.text:
        if http_client:
            await messenger_api.send_text(http_client, inbound.psid, "I received your attachment. A teammate will review it soon.")
        return

    # 4. DEBOUNCE
    merged_text = inbound.text or ""
    if redis:
        debounced = await redis_ops.buffer_and_wait(redis, inbound.psid, merged_text, debounce_seconds=2)
        if debounced is None:
            return  # A newer message in the burst owns the flush
        merged_text = debounced

    if not merged_text:
        return

    # 5. LOCK
    if redis:
        locked = await redis_ops.acquire_user_lock(redis, inbound.psid)
        if not locked:
            logger.info("Worker lock for %s already held.", inbound.psid)
            return

    try:
        # 6. LOG USER MESSAGE
        if session_factory:
            async with session_factory() as session:
                await repo.save_message(session, inbound.page_id, inbound.psid, "user", merged_text)

        # 7. TYPING INDICATOR
        if http_client:
            await messenger_api.send_typing(http_client, inbound.psid)

        # 8. AGENT TURN
        reply_text = "I'm sorry, I couldn't process your request right now."
        if llm and session_factory:
            tools = make_tools(inbound.psid,inbound.page_id, session_factory)
            agent_graph = build_graph(llm, tools, SYSTEM_PROMPT, checkpointer=saver)
            reply_text = await run_turn(agent_graph, merged_text, psid=inbound.psid)

        # 9. SEND REPLY
        if http_client:
            await messenger_api.send_text(http_client, inbound.psid, reply_text)

        # 10. LOG ASSISTANT MESSAGE
        if session_factory:
            async with session_factory() as session:
                await repo.save_message(session, inbound.page_id, inbound.psid, "assistant", reply_text)

    except Exception as err:
        logger.exception("Error processing message for %s: %s", inbound.psid, err)
        if http_client:
            try:
                await messenger_api.send_text(
                    http_client, inbound.psid, "Something went wrong on our end. A teammate will follow up."
                )
            except Exception:
                pass
    finally:
        if redis:
            await redis_ops.release_user_lock(redis, inbound.psid)


