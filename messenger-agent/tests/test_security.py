"""Step 1: signature validation. Green when app/security.py is implemented."""

import hashlib
import hmac

from app.security import verify_signature

SECRET = "test_app_secret"
BODY = b'{"object":"page","entry":[]}'


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_valid_signature_accepted():
    assert verify_signature(SECRET, BODY, sign(SECRET, BODY)) is True


def test_wrong_secret_rejected():
    assert verify_signature(SECRET, BODY, sign("attacker_guess", BODY)) is False


def test_tampered_body_rejected():
    assert verify_signature(SECRET, b'{"object":"page","entry":["INJECTED"]}', sign(SECRET, BODY)) is False


def test_missing_header_rejected():
    assert verify_signature(SECRET, BODY, None) is False


def test_malformed_header_rejected():
    assert verify_signature(SECRET, BODY, "md5=abc123") is False


def test_reserialized_json_would_fail():
    """THE lesson: the signature covers the exact raw bytes. The same JSON
    with different spacing = different bytes = invalid. This is why the
    webhook must hash request.body(), never a re-serialized parse."""
    reserialized = b'{"object": "page", "entry": []}'  # note the spaces
    assert verify_signature(SECRET, reserialized, sign(SECRET, BODY)) is False
