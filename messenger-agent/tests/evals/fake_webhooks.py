import httpx

import csv
from app.config import get_settings

from app.chatwoot import (
    ChatwootEvent)
CONV = {"id": 7, "account_id": 1, "inbox_id": 2, "status": "pending"}
CONTACT = {"id": 55, "name": "Ali", "type": "contact"}
def event(**overrides) -> ChatwootEvent:
    base = {
        "event": "message_created",
        "id": 101,
        "content": "salam, chhal taman?",
        "message_type": "incoming",
        "private": False,
        "conversation": dict(CONV),
        "sender": dict(CONTACT),
        "attachments": [],
    }
    base.update(overrides)
    return ChatwootEvent.model_validate(base)

async def test_webhook_endpoint():
    with open ("algerian_test_dataset.csv", "r", encoding="utf-8") as f:
        data = csv.reader(f)
        for input in data["text"]:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:8000/webhook/{get_settings().webhook_secret}",
                    json={"text": input},
                )
                assert response.status_code == 200
