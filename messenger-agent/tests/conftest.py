"""Test fixtures — provided complete. Read them; they show testing patterns
you'll reuse forever (fake LLM, fake Redis, env isolation).
"""

from __future__ import annotations

import os

# Set BEFORE importing the app: tests must never touch real secrets.
os.environ.setdefault("FACEBOOK_VERIFY_TOKEN", "test_verify_token")
os.environ.setdefault("FACEBOOK_APP_SECRET", "test_app_secret")
os.environ.setdefault("PAGE_ACCESS_TOKEN", "test_page_token")

import fakeredis.aioredis
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedLLM(BaseChatModel):
    """Replays a fixed list of AIMessages — no API key, fully deterministic.
    The same pattern that smoke-tested the debugger project."""

    script: list = []
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        index = (len(self.calls) - 1) % len(self.script)
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def scripted_llm_factory():
    def make(script: list[AIMessage]) -> ScriptedLLM:
        return ScriptedLLM(script=script, calls=[])

    return make


@pytest.fixture
async def sqlite_session_factory():
    """Repo tests run on SQLite (aiosqlite) — no Postgres needed locally.
    The models use dialect-agnostic types (sqlalchemy.Uuid) so this works."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()
