"""Voice notes and images -> text, BEFORE the agent runs. (Phase 2)

DESIGN (decided in the architecture review):
  Media is converted to text in the worker, and the agent only ever sees text.

  Why not feed raw media into the graph? Because the checkpointer would then
  store image/audio bytes in the conversation state and replay them into EVERY
  later turn of that conversation. Token cost grows with conversation length
  and your Postgres fills with base64. Transcribe once, keep the text.

  Cost of this choice: the agent can't "look again" at the photo later. For a
  support bot that is the right trade. Revisit only if customers start asking
  visual follow-ups ("is the stitching on the left the same as the right?").

VERIFIED against langchain-google-genai 4.3.2 (_convert_to_parts):
    {"type": "media", "mime_type": "audio/mp4", "data": <RAW base64 str>}   OK
    {"type": "media", "mime_type": "image/jpeg", "data": <RAW base64 str>}  OK
    {"type": "media", ... "data": "data:audio/mp4;base64,AAAA"}   ValueError!
  -> pass RAW base64. Do NOT prefix it with a data: URL for media blocks.

Definition of done: pytest tests/test_media.py
"""

from __future__ import annotations

import base64
import logging

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Gemini rejects oversized inline payloads, and a 20-minute voice note is not a
# support question. Refuse early rather than burning latency then failing.
MAX_MEDIA_BYTES = 12 * 1024 * 1024

DOWNLOAD_TIMEOUT = 20.0

TRANSCRIBE_PROMPT = (
    "Transcribe this voice message exactly, in its original language. "
    "Output only the transcription, no commentary, no translation. "
    "If the audio is unintelligible or silent, output exactly: [inaudible]"
)

DESCRIBE_PROMPT = (
    "You are helping a clothing store assistant understand a photo a customer "
    "sent. Describe it in 1-2 sentences: what product it shows, colour, "
    "distinctive details, and any visible text, label or damage. "
    "If it is not a product photo (a screenshot, a receipt, a person, a meme), "
    "say what it is instead. Be factual, do not guess a brand."
)


async def fetch_media(client: httpx.AsyncClient, url: str) -> tuple[bytes, str] | None:
    """Download an attachment from Meta's CDN. Returns (bytes, mime_type) or None.

    ⚠️ Meta's attachment URLs EXPIRE quickly (they are signed, short-lived
    links). Download at the moment the webhook arrives — never store the URL
    and fetch it later, and never put the URL in a retry queue that runs in
    ten minutes. Store the transcription, not the link.
    """
    try:
        resp = await client.get(url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("Failed to fetch media from %s (status %s)", url, resp.status_code)
            return None
        if len(resp.content) > MAX_MEDIA_BYTES:
            logger.warning("Media payload too large (%d bytes)", len(resp.content))
            return None
        mime = resp.headers.get("content-type", "").split(";")[0].strip() or "application/octet-stream"
        return (resp.content, mime)
    except httpx.HTTPError as e:
        logger.warning("HTTP error fetching media: %s", e)
        return None


def build_media_block(data: bytes, mime_type: str) -> dict:
    """Wrap raw bytes into the content block Gemini accepts."""
    b64 = base64.b64encode(data).decode()
    return {"type": "media", "mime_type": mime_type, "data": b64}


def message_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


async def transcribe(llm: BaseChatModel, data: bytes, mime_type: str) -> str | None:
    """Voice note -> text, in the customer's own language."""
    try:
        message = HumanMessage(content=[
            {"type": "text", "text": TRANSCRIBE_PROMPT},
            build_media_block(data, mime_type),
        ])
        response = await llm.ainvoke([message])
    except Exception as err:
        logger.exception("Error transcribing media: %s", err)
        return None

    text = message_text(response).strip()
    if not text or text.lower() == "[inaudible]":
        return None
    return text


async def describe_image(llm: BaseChatModel, data: bytes, mime_type: str) -> str | None:
    """Photo -> short factual description for the agent."""
    try:
        message = HumanMessage(content=[
            {"type": "text", "text": DESCRIBE_PROMPT},
            build_media_block(data, mime_type),
        ])
        response = await llm.ainvoke([message])
    except Exception as err:
        logger.exception("Error describing image: %s", err)
        return None

    text = message_text(response).strip()
    if not text or text.lower() == "[inaudible]":
        return None
    return text


async def media_to_text(
    llm: BaseChatModel,
    client: httpx.AsyncClient,
    attachment_type: str,
    url: str,
) -> str | None:
    """One attachment -> a text fragment the agent can read. Orchestrates the above."""
    try:
        fetched = await fetch_media(client, url)
        if fetched is None:
            return None
        data, mime = fetched
        if attachment_type == "audio":
            text = await transcribe(llm, data, mime)
            return f"[Voice message: {text}]" if text else None
        elif attachment_type == "image":
            text = await describe_image(llm, data, mime)
            return f"[Photo: {text}]" if text else None
        else:
            return None
    except Exception as err:
        logger.exception("Error in media_to_text: %s", err)
        return None
