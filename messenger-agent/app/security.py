"""Webhook signature validation. YOU implement this. (Step 1 of the path)

Why it exists: your webhook URL is public. Without this check, anyone who
finds the URL can POST fake "customer messages" and your bot will act on them.
Meta signs every delivery so you can prove it's really them.

How Meta signs: HMAC-SHA256 over the RAW REQUEST BODY BYTES, keyed with your
App Secret. The result arrives in the header:

    X-Hub-Signature-256: sha256=<64 hex chars>

Definition of done: pytest tests/test_security.py
"""

from __future__ import annotations
import hashlib
import hmac


def verify_signature(app_secret: str, raw_body: bytes, header_value: str | None) -> bool:
    """Return True only if header_value is a valid signature of raw_body.

    TODO(you) — steps:
      1. If header_value is None or doesn't start with "sha256=", return False.
      2. Strip the "sha256=" prefix to get the hex signature Meta sent.
      3. Compute your own: hmac.new(app_secret.encode(), raw_body,
         hashlib.sha256).hexdigest()
      4. Compare yours to theirs with hmac.compare_digest(...) — NOT with ==.
         (== leaks timing information; compare_digest is constant-time.)
      5. Return the comparison result.

    ⚠️ The classic mistake: computing the HMAC over the PARSED-then-re-serialized
    JSON. json.dumps reorders/respaces — the bytes differ — signatures never
    match. That is why webhook.py must read `await request.body()` BEFORE any
    JSON parsing, and pass those exact bytes here.
    """
    if header_value is None or not header_value.startswith("sha256="):
        return False
    their_sig = header_value[7:]
    our_sig = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(their_sig, our_sig)
