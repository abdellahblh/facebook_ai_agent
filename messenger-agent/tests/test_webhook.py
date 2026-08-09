"""Steps 2-3: the webhook endpoints. These are the EXACT tests that failed
against your first version — now they're your acceptance criteria.
"""

import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SECRET = "test_app_secret"


def signed(payload: dict) -> tuple[bytes, dict]:
    raw = json.dumps(payload).encode()
    sig = "sha256=" + hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return raw, {"X-Hub-Signature-256": sig, "Content-Type": "application/json"}


# ── GET /webhook (verification) ───────────────────────────────────────────────


def test_verify_with_metas_real_dotted_params():
    """Meta sends hub.mode with a DOT. Your v1 returned 400 here."""
    r = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.challenge": "1158201444",
            "hub.verify_token": "test_verify_token",
        },
    )
    assert r.status_code == 200
    assert r.text == "1158201444"  # exact echo, no JSON quotes


def test_verify_wrong_token_403():
    r = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.challenge": "123",
            "hub.verify_token": "wrong",
        },
    )
    assert r.status_code == 403


def test_verify_wrong_mode_400():
    r = client.get(
        "/webhook",
        params={
            "hub.mode": "unsubscribe",
            "hub.challenge": "123",
            "hub.verify_token": "test_verify_token",
        },
    )
    assert r.status_code == 400


# ── POST /webhook (events) ────────────────────────────────────────────────────

REAL_TEXT_EVENT = {
    "object": "page",
    "entry": [
        {
            "id": "PAGE1",
            "time": 1,
            "messaging": [
                {
                    "sender": {"id": "PSID_1"},
                    "recipient": {"id": "PAGE1"},
                    "timestamp": 1,
                    "message": {"mid": "m_1", "text": "hello"},
                }
            ],
        }
    ],
}


def test_receive_envelope_returns_200(monkeypatch):
    """Your v1 expected a bare MessagingItem -> 422 on every real event."""
    import app.worker

    async def fake_process(inbound):  # don't run the real pipeline in this test
        fake_process.called = True

    fake_process.called = False
    monkeypatch.setattr(app.worker, "process_event", fake_process)

    raw, headers = signed(REAL_TEXT_EVENT)
    r = client.post("/webhook", content=raw, headers=headers)
    assert r.status_code == 200


def test_bad_signature_403():
    raw = json.dumps(REAL_TEXT_EVENT).encode()
    r = client.post(
        "/webhook",
        content=raw,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 403


def test_unparseable_body_still_200():
    """RULE 2 (always-200): garbage inside a valid signature must NOT error —
    log it / dead-letter it, and ack. Meta disables webhooks that keep failing."""
    raw = b'{"object": "page", "entry": [{"THIS": ["IS", "NOT", {"the": "shape"}]}]}'
    sig = "sha256=" + hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()
    r = client.post(
        "/webhook",
        content=raw,
        headers={
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 200


# ── normalize() ───────────────────────────────────────────────────────────────


def test_normalize_filters_echoes():
    from app.schemas import MessagingEvent
    from app.webhook import normalize

    echo = MessagingEvent.model_validate(
        {
            "sender": {"id": "PAGE1"},
            "recipient": {"id": "PSID_1"},
            "message": {"mid": "m", "text": "bot's own reply", "is_echo": True},
        }
    )
    assert normalize("PAGE1", echo) is None


def test_normalize_keeps_text_AND_attachments():
    """Image + caption: your v1's elif chain dropped the caption."""
    from app.schemas import MessagingEvent
    from app.webhook import normalize

    ev = MessagingEvent.model_validate(
        {
            "sender": {"id": "PSID_1"},
            "recipient": {"id": "PAGE1"},
            "message": {
                "mid": "m",
                "text": "do you have this in size 42?",
                "attachments": [{"type": "image", "payload": {"url": "https://cdn/x.jpg"}}],
            },
        }
    )
    out = normalize("PAGE1", ev)
    assert out is not None
    assert out.text == "do you have this in size 42?"
    assert out.attachment_types == ["image"]
    assert out.attachment_urls == ["https://cdn/x.jpg"]


def test_normalize_skips_delivery_receipts():
    from app.schemas import MessagingEvent
    from app.webhook import normalize

    receipt = MessagingEvent.model_validate(
        {
            "sender": {"id": "PSID_1"},
            "recipient": {"id": "PAGE1"},
            "delivery": {"mids": ["m_1"]},
        }
    )
    assert normalize("PAGE1", receipt) is None
