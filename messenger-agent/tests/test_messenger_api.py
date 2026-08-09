"""Step 7: the pure helper (network calls get tested manually via ngrok)."""

from app.messenger_api import MAX_MESSAGE_CHARS, split_message


def test_short_message_untouched():
    assert split_message("salam!") == ["salam!"]


def test_exact_limit_untouched():
    text = "x" * MAX_MESSAGE_CHARS
    assert split_message(text) == [text]


def test_long_message_split_under_limit():
    text = "Une phrase. " * 400  # ~4800 chars
    chunks = split_message(text)
    assert len(chunks) >= 2
    assert all(len(c) <= MAX_MESSAGE_CHARS for c in chunks)


def test_split_preserves_all_content():
    text = ("ligne " * 300 + "\n") * 3
    chunks = split_message(text)
    reassembled = "".join(chunks).replace(" ", "").replace("\n", "")
    assert reassembled == text.replace(" ", "").replace("\n", "")


def test_prefers_newline_boundaries():
    part = "a" * 1500
    text = f"{part}\n{part}"
    chunks = split_message(text)
    assert len(chunks) == 2
    assert chunks[0].rstrip("\n") == part  # cut at the newline, not mid-word
