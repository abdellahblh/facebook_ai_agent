"""Webhook models — provided COMPLETE, because every line encodes a lesson
from the review of your first version. Read the comments; don't skip this file.

The rule: model the ENVELOPE strictly, the VARIANTS tolerantly.
Meta adds fields and event types without notice. A strict schema at this
boundary means Meta's next feature launch breaks your bot.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Tolerant(BaseModel):
    # Meta WILL send fields you've never seen. Ignore them, never reject.
    model_config = ConfigDict(extra="ignore")


class Attachment(Tolerant):
    # str, NOT Literal["image", "audio", ...]:
    # "sticker" appeared in 2026; the next type will appear without warning.
    # Known values belong in your dispatch logic, not in schema constraints.
    type: str = "fallback"
    payload: dict[str, Any] = Field(default_factory=dict)


class QuickReply(Tolerant):
    payload: str | None = None


class Message(Tolerant):
    mid: str | None = None
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    quick_reply: QuickReply | None = None
    # True = this is YOUR page's own message echoed back.
    # Process echoes as user input and your bot replies to itself forever.
    is_echo: bool = False


class Postback(Tolerant):
    title: str | None = None
    payload: str | None = None


class Party(Tolerant):
    id: str  # the only truly required field: without it we can't reply


class MessagingEvent(Tolerant):
    """One event. Messenger has NO 'type' field —
    the event type is WHICH KEY EXISTS (message vs postback vs delivery...)."""

    sender: Party
    recipient: Party
    timestamp: int = 0
    message: Message | None = None
    postback: Postback | None = None
    delivery: dict[str, Any] | None = None
    read: dict[str, Any] | None = None
    reaction: dict[str, Any] | None = None

    @property
    def kind(self) -> str:
        for name in ("message", "postback", "delivery", "read", "reaction"):
            if getattr(self, name) is not None:
                return name
        return "unknown"


class Entry(Tolerant):
    id: str  # the PAGE id — your tenant key if you go multi-client later
    time: int = 0
    messaging: list[MessagingEvent] = Field(default_factory=list)


class WebhookPayload(Tolerant):
    """The envelope. Meta ALWAYS posts this shape — never a bare message."""

    object: str
    entry: list[Entry] = Field(default_factory=list)


# ── Your internal type: Meta's shapes STOP here ──────────────────────────────
class InboundMessage(BaseModel):
    """What the rest of the bot sees. When you add WhatsApp/Instagram later,
    each platform gets a ~50-line adapter that emits this same type —
    and the agent, worker, and database never change."""

    page_id: str
    psid: str
    kind: str  # "message" | "postback"
    text: str | None = None
    # ALL attachment types Meta/Chatwoot sent — used for logging and for deciding
    # whether to tell the customer "I can't read that file type".
    attachment_types: list[str] = Field(default_factory=list)

    # Only attachments that actually HAVE a url, as (type, url) PAIRS.
    # Two parallel lists could drift out of sync (they did: a sticker has a
    # payload but no url, an unknown attachment can have an empty payload —
    # filtering one list and not the other silently mispaired them and dropped
    # a customer's photo). One list of pairs makes that impossible.
    media: list[tuple[str, str]] = Field(default_factory=list)

    # Kept for backwards compatibility with existing tests/callers.
    @property
    def attachment_urls(self) -> list[str]:
        return [url for _, url in self.media]

    quick_reply_payload: str | None = None
    postback_payload: str | None = None
    # Channel source: "messenger" | "chatwoot"
    channel: str = "messenger"
    # Chatwoot account ID for API calls
    account_id: int | None = None
    # Chatwoot: the conversation is the natural memory scope (thread_id).
    # None for the Meta adapter, which keys memory by psid.
    conversation_id: str | None = None
    mid: str | None = None
