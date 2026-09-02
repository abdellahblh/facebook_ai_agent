"""Webhook signature validation. YOU implement this. (Step 1 of the path)

Why it exists: your webhook URL is public. Without this check, anyone who
finds the URL can POST fake "customer messages" and your bot will act on them.
Meta signs every delivery so you can prove it's really them.

How Meta signs: HMAC-SHA256 over the RAW REQUEST BODY BYTES, keyed with your
App Secret. The result arrives in the header:

    X-Hub-Signature-256: sha256=<64 hex chars>

Definition of done: pytest tests/test_security.py
"""
"""
from __future__ import annotations
import hashlib
import hmac

def verify_signature(app_secret: str, raw_body: bytes, header_value: str | None) -> bool:
    Return True only if header_value is a
      valid signature of raw_body.

    if header_value is None or not header_value.startswith("sha256="):
        return False
    their_sig = header_value[7:]
    our_sig = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(their_sig, our_sig)

"""