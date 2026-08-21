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

    """
    settings = get_settings(); url =f"https://graph.facebook.com/{settings.graph_api_version}/me/messages"
    response = await client.post(url, json={"recipient": {"id": psid}, "sender_action": "typing_on"}, params={"access_token": settings.page_access_token})
    if response.status_code != 200:
        logger.warning("Failed to send typing indicator: %s", response.text)



async def send_text(client: httpx.AsyncClient, psid: str, text: str) -> None:
    """Deliver the reply.

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