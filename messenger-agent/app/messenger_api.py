"""Outbound: the Send API. YOU implement it. (Step 7 of the path)

Docs: Meta Graph API POST /{version}/me/messages?access_token=PAGE_ACCESS_TOKEN
Body: {"recipient": {"id": PSID}, "message": {"text": "..."}}
"""

from __future__ import annotations

import logging
import re
import httpx

from app.config import get_settings  # noqa: F401  (pre-imported for your implementation)

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 2000  # Messenger's hard limit per message


async def send_typing(client: httpx.AsyncClient, psid: str) -> None:
    """Show the typing indicator while the LLM thinks — the difference
    between a bot that feels broken and one that feels alive.

    TODO(you) — steps:
      1. settings = get_settings(); url =
         f"https://graph.facebook.com/{settings.graph_api_version}/me/messages"
      2. POST json={"recipient": {"id": psid}, "sender_action": "typing_on"},
         params={"access_token": settings.page_access_token}
      3. Fire-and-forget: log failures (resp.status_code != 200) but never
         raise — a failed typing indicator must not kill the reply.
    """
    settings = get_settings(); url =f"https://graph.facebook.com/{settings.graph_api_version}/me/messages"
    response = await client.post(url, json={"recipient": {"id": psid}, "sender_action": "typing_on"}, params={"access_token": settings.page_access_token})
    if response.status_code != 200:
        logger.warning("Failed to send typing indicator: %s", response.text)



async def send_text(client: httpx.AsyncClient, psid: str, text: str) -> None:
    """Deliver the reply.

    TODO(you) — steps:
      1. Split text into chunks of MAX_MESSAGE_CHARS (prefer splitting on
         newlines/sentence ends — read the tests for the expected behavior).
      2. POST each chunk in order to the same url as send_typing, body
         {"recipient": {"id": psid}, "message": {"text": chunk}}.
      3. On non-200: log resp.text — Meta's error body tells you exactly
         what's wrong (expired token, out-of-window, ...). Raise on hard
         failure so the worker can decide.
    """
    settings = get_settings(); url =f"https://graph.facebook.com/{settings.graph_api_version}/me/messages"
    chunks = split_message(text)
    for chunk in chunks:
        response = await client.post(url, json={"recipient": {"id": psid}, "message": {"text": chunk}}, params={"access_token": settings.page_access_token})
        if response.status_code != 200:
            logger.warning("Failed to send message: %s", response.text)



def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]

    # Split into lines, then sentences — smallest units we're allowed to break at
    units = []
    for line in text.split('\n'):
        units.extend(re.split(r'(?<=[.!?])\s+', line) if len(line) > limit else [line])

    result, current = [], ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                result.append(current)
            current = unit  # may still exceed limit — acceptable per spec ("if avoidable")

    if current:
        result.append(current)
    return result