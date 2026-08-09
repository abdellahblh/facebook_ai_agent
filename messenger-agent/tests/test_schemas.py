"""Schema tests — GREEN FROM DAY ONE (schemas are provided).

Each test is one bug from your first version. If you ever 'simplify'
schemas.py and one of these goes red, you are re-introducing a real bug.
"""

from app.schemas import WebhookPayload


def envelope(messaging_event: dict) -> dict:
    return {"object": "page", "entry": [{"id": "PAGE1", "time": 1, "messaging": [messaging_event]}]}


def test_real_text_message_parses():
    p = WebhookPayload.model_validate(
        envelope(
            {
                "sender": {"id": "PSID"},
                "recipient": {"id": "PAGE1"},
                "timestamp": 1,
                "message": {"mid": "m1", "text": "salam, combien la veste?"},
            }
        )
    )
    ev = p.entry[0].messaging[0]
    assert ev.kind == "message"
    assert ev.message.text == "salam, combien la veste?"


def test_sticker_id_arrives_as_int():
    # v1 bug: sticker_id typed str -> 422 on every sticker
    p = WebhookPayload.model_validate(
        envelope(
            {
                "sender": {"id": "PSID"},
                "recipient": {"id": "PAGE1"},
                "message": {
                    "mid": "m1",
                    "attachments": [{"type": "sticker", "payload": {"sticker_id": 369239263222822}}],
                },
            }
        )
    )
    assert p.entry[0].messaging[0].message.attachments[0].type == "sticker"


def test_postback_has_no_message_key():
    # v1 bug: message was required -> button clicks got 422
    p = WebhookPayload.model_validate(
        envelope(
            {
                "sender": {"id": "PSID"},
                "recipient": {"id": "PAGE1"},
                "postback": {"title": "Talk to human", "payload": "HANDOFF"},
            }
        )
    )
    ev = p.entry[0].messaging[0]
    assert ev.kind == "postback"
    assert ev.postback.payload == "HANDOFF"


def test_unknown_attachment_type_tolerated():
    # v1 bug: Literal[...] rejected any type Meta ships next
    p = WebhookPayload.model_validate(
        envelope(
            {
                "sender": {"id": "PSID"},
                "recipient": {"id": "PAGE1"},
                "message": {"mid": "m1", "attachments": [{"type": "type_from_2027", "payload": {}}]},
            }
        )
    )
    assert p.entry[0].messaging[0].message.attachments[0].type == "type_from_2027"


def test_unknown_event_kind_is_unknown_not_crash():
    p = WebhookPayload.model_validate(
        envelope(
            {
                "sender": {"id": "PSID"},
                "recipient": {"id": "PAGE1"},
                "totally_new_meta_feature": {"x": 1},
            }
        )
    )
    assert p.entry[0].messaging[0].kind == "unknown"


def test_echo_flag_surfaces():
    p = WebhookPayload.model_validate(
        envelope(
            {
                "sender": {"id": "PAGE1"},
                "recipient": {"id": "PSID"},
                "message": {"mid": "m1", "text": "reply from the page itself", "is_echo": True},
            }
        )
    )
    assert p.entry[0].messaging[0].message.is_echo is True
