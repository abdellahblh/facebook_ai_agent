"""The Meta webhook — verification (GET) and events (POST). YOU implement both.
(Steps 2-3 of the path)

Definition of done: pytest tests/test_webhook.py
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from app.security import verify_signature
from app.config import get_settings  # noqa: F401  (pre-imported for your implementation)
from app.schemas import InboundMessage, MessagingEvent, WebhookPayload  # noqa: F401
from app.worker import process_event

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/webhook")
async def verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
) -> Response:
    if hub_mode != "subscribe":
        return Response(status_code=400)
    if hub_verify_token != get_settings().facebook_verify_token:
        return Response(status_code=403)
    if not hub_challenge:
        return Response(status_code=400)
    return PlainTextResponse(hub_challenge)


@router.post("/webhook")
async def receive(request: Request) -> dict:
    raw = await request.body()
    if not verify_signature(
        get_settings().facebook_app_secret, raw, request.headers.get("X-Hub-Signature-256")
    ):
        return Response(status_code=403)
    try:
        payload = WebhookPayload.model_validate_json(raw)
    except Exception as e:
        logger.warning(f"Failed to parse webhook payload: {e}")
        return {"status": "ok"}
    for entry in payload.entry:
        for event in entry.messaging:
            normalized = normalize(entry.id, event)
            if normalized is None:
                continue
            await process_event(normalized)
    return {"status": "ok"}


def normalize(page_id: str, event: MessagingEvent) -> InboundMessage | None:
    if event.message and event.message.is_echo:
        return None
    if event.kind in ("delivery", "read", "reaction", "unknown"):
        return None
    psid = event.sender.id
    page_id = page_id
    kind = event.kind
    text = event.message.text if (event.message and event.kind == "message") else None
    attachments = event.message.attachments if (event.message and event.message.attachments) else []
    attachment_types = [a.type for a in attachments]
    attachment_urls = [a.payload.get("url") for a in attachments if a.payload]
    quick_reply_payload = event.message.quick_reply.payload if (event.message and event.message.quick_reply) else None
    postback_payload = event.postback.payload if (event.postback and event.kind == "postback") else None
    mid = event.message.mid if event.message else None

    return InboundMessage(
        page_id=page_id,
        psid=psid,
        kind=kind,
        text=text,
        attachment_types=attachment_types,
        attachment_urls=attachment_urls,
        quick_reply_payload=quick_reply_payload,
        postback_payload=postback_payload,
        mid=mid,
    )

