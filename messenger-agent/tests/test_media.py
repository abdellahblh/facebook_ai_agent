"""Phase 2: voice + image -> text. No API key, no network.

The LLM is the scripted fake from conftest; HTTP is a stubbed httpx transport.
Every failure path is tested, because in production media handling fails far
more often than text does: expired CDN links, huge files, silent audio.
"""

import base64

import httpx
from langchain_core.messages import AIMessage

from app.media import (
    MAX_MEDIA_BYTES,
    build_media_block,
    describe_image,
    fetch_media,
    media_to_text,
    transcribe,
)

AUDIO = b"\x00\x01fake-audio-bytes"
IMAGE = b"\x89PNG\r\n\x1a\nfake-image"


def stub_client(status=200, content=AUDIO, content_type="audio/mp4", boom=False):
    """An httpx client whose responses we control, no network involved."""

    def handler(request: httpx.Request) -> httpx.Response:
        if boom:
            raise httpx.ConnectError("simulated network failure")
        return httpx.Response(status, content=content, headers={"content-type": content_type})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── build_media_block: the format verified against the real library ───────────

def test_media_block_uses_raw_base64_not_data_url():
    block = build_media_block(AUDIO, "audio/mp4")
    assert block["type"] == "media"
    assert block["mime_type"] == "audio/mp4"
    assert not block["data"].startswith("data:")  # data: URL -> ValueError in the lib
    assert base64.b64decode(block["data"]) == AUDIO


def test_media_block_accepted_by_the_real_converter():
    """Guards against a library upgrade silently changing the accepted shape."""
    from langchain_google_genai.chat_models import _convert_to_parts

    parts = _convert_to_parts([build_media_block(IMAGE, "image/png")])
    assert parts[0].inline_data.mime_type == "image/png"


# ── fetch_media: every failure mode degrades, never raises ────────────────────

async def test_fetch_returns_bytes_and_mime():
    async with stub_client() as c:
        got = await fetch_media(c, "https://cdn.meta/x")
    assert got == (AUDIO, "audio/mp4")


async def test_fetch_strips_charset_from_content_type():
    async with stub_client(content_type="image/jpeg; charset=binary") as c:
        _, mime = await fetch_media(c, "https://cdn.meta/x")
    assert mime == "image/jpeg"


async def test_expired_url_returns_none_not_exception():
    """Meta attachment URLs expire. That is normal, not a crash."""
    async with stub_client(status=404) as c:
        assert await fetch_media(c, "https://cdn.meta/expired") is None


async def test_network_failure_returns_none():
    async with stub_client(boom=True) as c:
        assert await fetch_media(c, "https://cdn.meta/x") is None


async def test_oversized_media_refused():
    huge = b"x" * (MAX_MEDIA_BYTES + 1)
    async with stub_client(content=huge) as c:
        assert await fetch_media(c, "https://cdn.meta/huge") is None


# ── transcribe / describe ─────────────────────────────────────────────────────

async def test_transcribe_returns_text(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="wach kayen, chhal taman?")])
    assert await transcribe(llm, AUDIO, "audio/mp4") == "wach kayen, chhal taman?"


async def test_transcribe_sends_prompt_and_audio_together(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="ok")])
    await transcribe(llm, AUDIO, "audio/mp4")
    blocks = llm.calls[0][0].content
    assert blocks[0]["type"] == "text"          # the instruction
    assert blocks[1]["type"] == "media"         # the audio
    assert blocks[1]["mime_type"] == "audio/mp4"


async def test_inaudible_returns_none(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="[inaudible]")])
    assert await transcribe(llm, AUDIO, "audio/mp4") is None


async def test_empty_response_returns_none(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="   ")])
    assert await transcribe(llm, AUDIO, "audio/mp4") is None


async def test_handles_list_content_blocks(scripted_llm_factory):
    """Gemini can return content as a list of blocks, not a plain string."""
    llm = scripted_llm_factory([AIMessage(content=[{"type": "text", "text": "salam"}])])
    assert await transcribe(llm, AUDIO, "audio/mp4") == "salam"


async def test_llm_error_returns_none(scripted_llm_factory):
    class Exploding(type(scripted_llm_factory([AIMessage(content="x")]))):
        def _generate(self, *a, **k):
            raise RuntimeError("gemini 503")

    assert await transcribe(Exploding(script=[], calls=[]), AUDIO, "audio/mp4") is None


async def test_describe_image_returns_text(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="A red zip-up jacket, size tag visible.")])
    out = await describe_image(llm, IMAGE, "image/png")
    assert "red zip-up jacket" in out


# ── media_to_text: the orchestration the worker calls ─────────────────────────

async def test_audio_becomes_bracketed_voice_message(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="do you deliver to Oran")])
    async with stub_client(content=AUDIO, content_type="audio/mp4") as c:
        out = await media_to_text(llm, c, "audio", "https://cdn.meta/a")
    assert out == "[Voice message: do you deliver to Oran]"


async def test_image_becomes_bracketed_photo(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="A black leather boot.")])
    async with stub_client(content=IMAGE, content_type="image/jpeg") as c:
        out = await media_to_text(llm, c, "image", "https://cdn.meta/i")
    assert out.startswith("[Photo:") and "black leather boot" in out


async def test_unsupported_attachment_type_skipped(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="should not be called")])
    async with stub_client() as c:
        assert await media_to_text(llm, c, "video", "https://cdn.meta/v") is None
    assert llm.calls == []  # never wasted an LLM call


async def test_failed_download_skips_llm_entirely(scripted_llm_factory):
    llm = scripted_llm_factory([AIMessage(content="should not be called")])
    async with stub_client(status=404) as c:
        assert await media_to_text(llm, c, "audio", "https://cdn.meta/gone") is None
    assert llm.calls == []
