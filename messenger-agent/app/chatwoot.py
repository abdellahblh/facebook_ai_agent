"""Chatwoot integration: inbound webhook + outbound replies + handoff.

ARCHITECTURE
    Chatwoot owns the Facebook connection. Messages arrive here as Chatwoot
    webhooks; replies go back through the Chatwoot API, which forwards them to
    Messenger. One write, both effects (customer sees it, dashboard shows it).

    Chatwoot event -> ChatwootEvent (tolerant) -> InboundMessage -> the SAME
    worker pipeline. That adapter boundary is why the worker barely changes.

FOUR THINGS THAT DIFFER FROM META:

 1. NO HMAC SIGNATURE. Chatwoot does not sign webhooks. The endpoint is
    protected by a long random secret in the URL path, compared in constant
    time. Generate it with `secrets.token_urlsafe(32)` — base64 secrets can
    contain "/", which splits the path and 404s before your check ever runs.

 2. THE LOOP IS BACK. Chatwoot fires `message_created` for incoming, outgoing
    (bot), outgoing (human agent) AND private notes. Reply to your own outgoing
    message and the bot talks to itself forever.

 3. HANDOFF IS CHATWOOT'S JOB. Status (pending/open/resolved) + assignee.
    Because new conversations arrive as `open`, status alone cannot gate the
    bot — the ASSIGNEE is what silences it. So escalation must assign.

 4. ATTACHMENTS: `file_type` + `data_url`, not Meta's `payload.url`.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from app.util.retry import with_retry_send
import httpx
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.schemas import InboundMessage

logger = logging.getLogger(__name__)
router = APIRouter()


def _log_worker_failure(task: asyncio.Task[Any]) -> None:
    """Log failures from the detached webhook worker."""
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Chatwoot worker failed after webhook ACK", exc_info=error)


# ── Models: tolerant at the boundary ─────────────────────────────────────────
class Tolerant(BaseModel):
    """Ignore unknown keys. Never reject a real message because Chatwoot
    shipped a new field."""

    model_config = ConfigDict(extra="ignore")


class ChatwootSender(Tolerant):
    id: int | None = None
    name: str | None = None
    email: str | None = None
    # "contact" (customer) | "user" (human agent) | "agent_bot" (us)
    type: str | None = None


class ChatwootAssignee(Tolerant):
    id: int | None = None
    name: str | None = None
    email: str | None = None
    type: str | None = None

 
class ChatwootTeam(Tolerant):
    id: int | None = None
    name: str | None = None


class ChatwootConversationMeta(Tolerant):
    assignee: ChatwootAssignee | None = None
    sender: ChatwootSender | None = None
    team: ChatwootTeam | None = None


class ChatwootConversation(Tolerant):
    id: int
    # OPTIONAL: some payload shapes carry the account only at the top level.
    account_id: int | None = None
    inbox_id: int | None = None
    status: str | None = None  # open | pending | resolved | snoozed
    channel: str | None = None
    meta: ChatwootConversationMeta | None = None
    additional_attributes: dict[str, Any] = Field(default_factory=dict)


class ChatwootAccount(Tolerant):
    id: int | None = None
    name: str | None = None


class ChatwootInbox(Tolerant):
    id: int | None = None
    name: str | None = None


class ChatwootAttachment(Tolerant):
    id: int | None = None
    file_type: str | None = None  # image | audio | video | file
    data_url: str | None = None
    thumb_url: str | None = None


class ChatwootEvent(Tolerant):
    """One webhook delivery.

    `conversation` is OPTIONAL: contact_created / contact_updated carry none,
    and a required field there 422s all of them.
    """

    event: str
    id: int | None = None
    content: str | None = None
    # int OR str: Chatwoot ships 0/1/2/3 in some versions and
    # "incoming"/"outgoing" in others. Pydantic will not coerce int -> str.
    message_type: int | str | None = None
    content_type: str | None = None
    private: bool = False
    created_at: Any = None
    conversation: ChatwootConversation | None = None
    sender: ChatwootSender | None = None
    account: ChatwootAccount | None = None
    inbox: ChatwootInbox | None = None
    attachments: list[ChatwootAttachment] = Field(default_factory=list)

    # ── derived helpers ──────────────────────────────────────────────────────
    @property
    def direction(self) -> str:
        """Normalise to incoming | outgoing | other.

        Membership tests, not isinstance: a payload carrying the STRING "0"
        must land on "incoming" too. `str(mt).lower()` alone would return "0".
        """
        mt = self.message_type
        if isinstance(mt, str):
            mt = mt.strip().lower()
        if mt in (0, "0", "incoming"):
            return "incoming"
        if mt in (1, "1", "outgoing"):
            return "outgoing"
        if mt in (2, "2", "activity"):
            return "other"
        if mt in (3, "3", "template"):
            return "template"
        return "unknown"

    @property
    def account_id(self) -> int | None:
        if self.conversation and self.conversation.account_id is not None:
            return self.conversation.account_id
        return self.account.id if self.account else None

    @property
    def inbox_id(self) -> int | None:
        if self.conversation and self.conversation.inbox_id is not None:
            return self.conversation.inbox_id
        return self.inbox.id if self.inbox else None


# ── Webhook guard ─────────────────────────────────────────────────────────────
def verify_secret(provided: str | None, configured: str | None) -> bool:
    """Constant-time compare of the URL-path secret.

    NO SECRET CONFIGURED = REJECT. A missing secret is a misconfiguration, not
    consent to run an open endpoint: anyone who guesses the URL could inject
    fake customer messages and burn your LLM budget. Fail closed and log why.
    """
    if not configured:
        logger.error(
            "CHATWOOT_WEBHOOK_SECRET is not set — rejecting the webhook. "
            "Generate one with: python3 -c \"import secrets; "
            'print(secrets.token_urlsafe(32))"'
        )
        return False
    if not provided:
        return False
    return secrets.compare_digest(provided, configured)


def should_process(event: ChatwootEvent) -> tuple[bool, str]:
    """Return (process, reason). The reason is logged, so "why didn't the bot
    answer?" is answerable from the logs instead of by guessing."""
    settings = get_settings()

    if event.event != "message_created":
        return False, f"event={event.event} (not a new message)"

    if event.conversation is None:
        return False, "no conversation on the payload"

    # THE LOOP GUARD.
    if event.direction != "incoming":
        return False, f"direction={event.direction} (our own or an agent's reply)"
    if event.private:
        return False, "private note between agents"
    if event.sender and event.sender.type in ("agent_bot", "user"):
        return False, f"sender type={event.sender.type} (not a customer)"

    status = (event.conversation.status or "").lower()
    if status == "resolved":
        return False, "conversation is resolved"

    # NOT rejecting `open`: new Chatwoot conversations arrive as `open`, so
    # gating on status would silence the bot for every fresh conversation.
    # The assignee/team checks below are the real handoff gate.
    if settings.chatwoot_bot_only_when_pending and status != "pending":
        return False, f"conversation status={status!r} (bot configured for pending only)"

    meta = event.conversation.meta
    assignee = meta.assignee if meta else None
    
    if assignee and (assignee.type or "").lower() != "agent_bot":
        return False, f"assigned to agent {assignee.id}"
    team = meta.team if meta else None
    if team and team.id:
        return False, f"assigned to team {team.id}"

    if not (event.content or "").strip() and not event.attachments:
        return False, "empty message"

    return True, "ok"


def _attachment_url(url: str) -> str:
    """Make Chatwoot's local development attachment URL reachable by us."""
    parsed = urlsplit(url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return url

    base = get_settings().chatwoot_base_url.rstrip("/")
    if not base:
        return url
    public = urlsplit(base)
    return urlunsplit((public.scheme, public.netloc, parsed.path, parsed.query, parsed.fragment))


def to_inbound(event: ChatwootEvent) -> InboundMessage:
    """Chatwoot event -> the internal type the worker already understands.

    KEY MAPPING
      page_id         = inbox_id   -> one inbox == one FB page == one tenant
      psid            = contact id -> stable per customer
      conversation_id = conv id    -> the checkpointer thread_id
    """
    conversation = event.conversation
    if conversation is None:  # should_process() guarantees this
        raise ValueError("to_inbound() requires an event with a conversation")

    sender = event.sender
    if (sender is None or sender.id is None) and conversation.meta:
        sender = conversation.meta.sender
    contact_id = sender.id if sender else None

    # ONE pass builds both lists. Two filtered comprehensions zipped together
    # is how attachments silently mispair when one lacks a data_url.
    attachment_types: list[str] = []
    media: list[tuple[str, str]] = []
    for attachment in event.attachments:
        kind = attachment.file_type or "file"
        attachment_types.append(kind)
        if attachment.data_url:
            media.append((kind, _attachment_url(attachment.data_url)))

    return InboundMessage(
        page_id=str(event.inbox_id if event.inbox_id is not None else "unknown"),
        psid=str(contact_id if contact_id is not None else conversation.id),
        kind="message",
        text=(event.content or "").strip() or None,
        attachment_types=attachment_types,
        media=media,
        channel="chatwoot",
        account_id=event.account_id,
        conversation_id=str(conversation.id),
        mid=str(event.id) if event.id is not None else None,
    )


# ── Inbound endpoint ─────────────────────────────────────────────────────────
# response_model=None: returns either a dict or a raw Response, which FastAPI
# cannot express as one response model.
@router.post("/webhook/{webhook_secret}", response_model=None)
async def chatwoot_webhook(webhook_secret: str, request: Request) -> Response | dict:
    """Configure in Chatwoot: Settings -> Integrations -> Webhooks ->
    https://your.host/webhook/<CHATWOOT_WEBHOOK_SECRET>   (no trailing slash)

    Everything past the secret check returns 200. A webhook consumer that
    errors gets retried — and a retry after a slow success is a DUPLICATE
    REPLY. Log, dead-letter, move on.
    """
    settings = get_settings()
    if not verify_secret(webhook_secret, settings.chatwoot_webhook_secret):
        logger.warning("Rejected Chatwoot webhook: bad secret in path.")
        return Response(status_code=403)  # the ONLY non-200

    raw = await request.body()
    try:
        event = ChatwootEvent.model_validate_json(raw)
    except Exception as exc:
        # Unknown shape: keep the bytes so the models can be extended from
        # evidence. Do NOT 400 — Chatwoot would retry a payload that will
        # never parse, forever.
        logger.warning("Unparseable Chatwoot payload: %s | %s", exc, raw[:800])
        return {"status": "ok"}

    try:
        process, reason = should_process(event)
        if not process:
            logger.info("Chatwoot event skipped (%s).", reason)
            return {"status": "ignored"}
        inbound = to_inbound(event)
    except Exception:
        logger.exception("Failed to process Chatwoot event — dropping, not retrying.")
        return {"status": "ok"}

    # ACK NOW, WORK LATER. Transcription + an agent turn takes many seconds;
    # awaiting it here blows Chatwoot's webhook timeout, Chatwoot retries, and
    # the customer gets answered twice.
    from app import worker

    task = asyncio.create_task(worker.process_event(inbound))
    task.add_done_callback(_log_worker_failure)
    return {"status": "ok"}


# ── Outbound: replies and handoff go through the Chatwoot API ─────────────────
def _api_headers() -> dict[str, str]:
    return {
        "api_access_token": get_settings().chatwoot_api_token,
        "Content-Type": "application/json",
    }


def _conversation_url(account_id: int | str | None, conversation_id: int | str) -> str:
    settings = get_settings()
    base = settings.chatwoot_base_url.rstrip("/")
    resolved = account_id if account_id is not None else settings.chatwoot_account_id
    if not resolved:
        # Fail loudly rather than defaulting to account 1. A silent guess
        # either 404s forever or writes into the WRONG tenant's conversation.
        raise ValueError(
            f"No account_id for conversation {conversation_id} and "
            "CHATWOOT_ACCOUNT_ID is unset. Refusing to guess — find the id in "
            "your Chatwoot dashboard URL: /app/accounts/<ID>/dashboard"
        )
    return f"{base}/api/v1/accounts/{resolved}/conversations/{conversation_id}"


async def send_reply(client: httpx.AsyncClient, account_id: int | str | None, conversation_id: int | str, text:str) -> bool:
    async def _post():
        response = await client.post(
            f"{_conversation_url(account_id, conversation_id)}/messages",
            headers=_api_headers(),
            json={"content": text, "message_type": "outgoing", "private": False},
            timeout=20.0,
        )
        response.raise_for_status()
        return response

    try:
        await with_retry_send(_post, label="chatwoot_send")
        return True
    except Exception:
        logger.exception("send_reply failed")
        return False

    if response.status_code >= 300:
        logger.error(
            "Chatwoot send rejected (%s) for conversation %s: %s — a 401/403 "
            "here means CHATWOOT_API_TOKEN is wrong or lacks agent access.",
            response.status_code, conversation_id, response.text[:400],
        )
        return False
    return True


async def send_typing(
    client: httpx.AsyncClient,
    account_id: int | str | None,
    conversation_id: int | str,
    status: Literal["on", "off"] = "on",
) -> bool:
    """Typing indicator. Cosmetic — failure must never break the reply."""
    try:
        response = await client.post(
            f"{_conversation_url(account_id, conversation_id)}/toggle_typing_status",
            headers=_api_headers(),
            json={"typing_status": status},
            timeout=5.0,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.debug("Typing status %s failed: %s", status, exc)
        return False
    return True


async def escalate_to_human(
    client: httpx.AsyncClient,
    account_id: int | str | None,
    conversation_id: int | str,
    assignee_id: int | None = None,
    team_id: int | None = None,
    note: str | None = "Handoff requested. Bot pausing.",
) -> bool:
    """Hand the conversation to a person.

    ASSIGNS it. Two separate endpoints, because Chatwoot has no combined one:
      POST .../assignments     {"assignee_id": N} or {"team_id": N}
      POST .../toggle_status   {"status": "open"}
    Posting {"status", "assignee_id"} to the conversation URL itself is not a
    Chatwoot route — it 404s, and the handoff silently does nothing.

    Assignment is what actually silences the bot: `open` is the status new
    conversations already have, so should_process() cannot gate on it.
    """
    settings = get_settings()
    target_assignee = assignee_id or settings.chatwoot_handoff_agent_id
    target_team = team_id or settings.chatwoot_handoff_team_id
    ok = True

    # 1. ASSIGN — the part that stops the bot. Team wins if both are set.
    assignment: dict[str, int] = {}
    if target_team:
        assignment["team_id"] = int(target_team)
    elif target_assignee:
        assignment["assignee_id"] = int(target_assignee)

    if assignment:
        try:
            response = await client.post(
                f"{_conversation_url(account_id, conversation_id)}/assignments",
                headers=_api_headers(),
                json=assignment,
                timeout=20.0,
            )
            if response.status_code >= 300:
                logger.error("Chatwoot assignment failed: %s", response.text[:400])
                ok = False
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception("Chatwoot assignment failed: %s", exc)
            ok = False
    else:
        logger.error(
            "HANDOFF IS A NO-OP for conversation %s: set CHATWOOT_HANDOFF_TEAM_ID "
            "or CHATWOOT_HANDOFF_AGENT_ID, otherwise the bot keeps replying over "
            "the human.",
            conversation_id,
        )
        ok = False

    # 2. status=open -> visible in the human inbox.
    try:
        await client.post(
            f"{_conversation_url(account_id, conversation_id)}/toggle_status",
            headers=_api_headers(),
            json={"status": "open"},
            timeout=20.0,
        )
    except (httpx.HTTPError, ValueError):
        logger.exception("Chatwoot toggle_status failed.")

    # 3. A PRIVATE note explaining why. Agents see it; the customer never does.
    if note:
        await send_reply(client, account_id, conversation_id, note, private=True)

    if ok:
        logger.info("Escalated conversation %s to a human.", conversation_id)
    return ok
