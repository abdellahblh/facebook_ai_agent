"""Chatwoot adapter. No network, no Chatwoot instance needed.

Every test here is a payload shape that either broke the first schema or would
have caused a production incident (the reply loop, a human being talked over).
"""

import httpx

from app.chatwoot import (
    ChatwootEvent,
    escalate_to_human,
    send_reply,
    send_typing,
    should_process,
    to_inbound,
)

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


# ── the three shapes that 422'd the original schema ───────────────────────────

def test_message_type_as_integer_is_accepted():
    """Chatwoot ships 0/1 in some versions. `str` rejected every message."""
    assert event(message_type=0).direction == "incoming"
    assert event(message_type=1).direction == "outgoing"
    assert event(message_type=2).direction == "other"  # activity


def test_message_type_as_string_is_accepted():
    assert event(message_type="incoming").direction == "incoming"
    assert event(message_type="outgoing").direction == "outgoing"


def test_account_id_only_at_top_level():
    """Some payloads omit conversation.account_id and carry `account` instead."""
    ev = event(
        conversation={"id": 7, "inbox_id": 2, "status": "pending"},
        account={"id": 1, "name": "My Store"},
    )
    assert ev.account_id == 1
    assert ev.inbox_id == 2


def test_event_without_conversation_parses():
    """contact_created has no conversation. Required field -> 422 for all of them."""
    ev = ChatwootEvent.model_validate({"event": "contact_created", "id": 55})
    assert ev.conversation is None
    process, reason = should_process(ev)
    assert process is False and "not a new message" in reason


def test_unknown_future_fields_ignored():
    ev = event(some_field_from_chatwoot_v9={"nested": True})
    assert ev.direction == "incoming"


# ── the loop guard: the bug that makes a bot talk to itself forever ───────────

def test_incoming_customer_message_is_processed():
    process, reason = should_process(event())
    assert process is True, reason


def test_our_own_outgoing_reply_is_skipped():
    process, reason = should_process(event(message_type="outgoing"))
    assert process is False
    assert "outgoing" in reason


def test_agent_bot_sender_is_skipped():
    process, _ = should_process(
        event(sender={"id": 9, "name": "AI", "type": "agent_bot"})
    )
    assert process is False


def test_private_agent_note_is_skipped():
    process, reason = should_process(event(private=True))
    assert process is False and "private" in reason


def test_activity_message_is_skipped():
    """message_type 2 = activity ('Conversation was marked resolved')."""
    process, _ = should_process(event(message_type=2))
    assert process is False


# ── handoff is Chatwoot's job now ─────────────────────────────────────────────

def test_bot_stays_quiet_when_status_is_open():
    """`open` means a human took it. Two sources of truth is the bug we avoid."""
    process, reason = should_process(
        event(conversation={**CONV, "status": "open"})
    )
    assert process is True and reason == "ok"


def test_bot_stays_quiet_when_resolved():
    process, _ = should_process(event(conversation={**CONV, "status": "resolved"}))
    assert process is False


def test_bot_stays_quiet_when_assigned_to_an_agent():
    process, reason = should_process(
        event(conversation={**CONV, "meta": {"assignee": {"id": 3, "name": "Sara"}}})
    )
    assert process is False and "assigned" in reason


def test_empty_message_skipped():
    process, reason = should_process(event(content="   ", attachments=[]))
    assert process is False and "empty" in reason


# ── mapping onto the internal type the worker already understands ─────────────

def test_maps_to_inbound_message():
    inbound = to_inbound(event())
    assert inbound.page_id == "2"            # inbox = tenant
    assert inbound.psid == "55"              # contact = customer
    assert inbound.conversation_id == "7"    # conversation = memory scope
    assert inbound.channel == "chatwoot"
    assert inbound.account_id == 1
    assert inbound.text == "salam, chhal taman?"
    assert inbound.mid == "101"


def test_chatwoot_attachment_shape_becomes_media_pairs():
    """Chatwoot uses file_type + data_url, not Meta's payload.url."""
    inbound = to_inbound(
        event(
            content="",
            attachments=[
                {"id": 9, "file_type": "audio", "data_url": "https://cw/voice.mp4"},
                {"id": 10, "file_type": "image", "data_url": "https://cw/photo.jpg"},
            ],
        )
    )
    assert inbound.media == [
        ("audio", "https://cw/voice.mp4"),
        ("image", "https://cw/photo.jpg"),
    ]
    assert inbound.attachment_types == ["audio", "image"]
    assert inbound.text is None


def test_localhost_attachment_url_uses_public_chatwoot_base(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chatwoot.example.com")
    from app.config import get_settings

    get_settings.cache_clear()
    inbound = to_inbound(
        event(
            content="",
            attachments=[
                {
                    "file_type": "audio",
                    "data_url": "http://localhost:3000/rails/blob.ogg?disposition=inline",
                }
            ],
        )
    )
    assert inbound.media == [
        ("audio", "https://chatwoot.example.com/rails/blob.ogg?disposition=inline")
    ]


def test_attachment_without_data_url_is_not_paired():
    """Same misalignment lesson as Meta: types keep it, media never gets a None."""
    inbound = to_inbound(
        event(
            attachments=[
                {"id": 9, "file_type": "file"},  # no data_url
                {"id": 10, "file_type": "image", "data_url": "https://cw/p.jpg"},
            ]
        )
    )
    assert inbound.attachment_types == ["file", "image"]
    assert inbound.media == [("image", "https://cw/p.jpg")]


def test_falls_back_to_conversation_meta_sender():
    inbound = to_inbound(
        event(sender=None, conversation={**CONV, "meta": {"sender": {"id": 77}}})
    )
    assert inbound.psid == "77"


# ── outbound API calls ────────────────────────────────────────────────────────

def api_stub(status=200, capture=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        return httpx.Response(status, json={"id": 999})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_send_reply_posts_outgoing_message(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://cw.example")
    monkeypatch.setenv("CHATWOOT_API_TOKEN", "tok")
    from app.config import get_settings

    get_settings.cache_clear()

    captured: list[httpx.Request] = []
    async with api_stub(capture=captured) as client:
        assert await send_reply(client, 1, 7, "3500 DA") is True

    request = captured[0]
    assert request.url.path == "/api/v1/accounts/1/conversations/7/messages"
    assert request.headers["api_access_token"] == "tok"
    body = request.content.decode()
    assert '"message_type":"outgoing"' in body.replace(" ", "")
    assert '"private":false' in body.replace(" ", "")


async def test_send_reply_returns_false_on_error(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://cw.example")
    from app.config import get_settings

    get_settings.cache_clear()
    async with api_stub(status=401) as client:
        assert await send_reply(client, 1, 7, "hi") is False


async def test_send_typing_toggles_typing_status(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://cw.example")
    monkeypatch.setenv("CHATWOOT_API_TOKEN", "tok")
    from app.config import get_settings

    get_settings.cache_clear()

    captured: list[httpx.Request] = []
    async with api_stub(capture=captured) as client:
        assert await send_typing(client, 1, 7) is True

    request = captured[0]
    assert request.url.path == "/api/v1/accounts/1/conversations/7/toggle_typing_status"
    assert request.headers["api_access_token"] == "tok"
    body = request.content.decode()
    assert '"typing_status":"on"' in body.replace(" ", "")


async def test_escalation_toggles_status_to_open(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://cw.example")
    from app.config import get_settings

    get_settings.cache_clear()
    captured: list[httpx.Request] = []
    async with api_stub(capture=captured) as client:
        assert await escalate_to_human(client, 1, 7, note="customer angry") is True

    paths = [r.url.path for r in captured]
    assert "/api/v1/accounts/1/conversations/7/toggle_status" in paths
    # the note must be PRIVATE — agents only, never shown to the customer
    note_request = [r for r in captured if r.url.path.endswith("/messages")][0]
    assert '"private":true' in note_request.content.decode().replace(" ", "")

