"""FastAPI app assembly — wiring provided; lifespan startup is YOURS.

Run: uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver 
from app.webhook import router as webhook_router
from app.config import get_settings  
from app.db.engine import init_engine
import httpx
import redis.asyncio as aioredis
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agent.graph import build_graph
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_core.tools import tool
from app.agent.tools import make_tools
from app.agent.prompts import SYSTEM_PROMPT
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build everything ONCE at startup; tear down cleanly at shutdown.

    TODO(you) — as you complete each learning step, light these up in order
    (the shape is exactly your debugger's api.py lifespan):

      1. settings = get_settings()
      2. await init_engine(settings.async_database_url)          [step 5]
      3. app.state.redis = redis.asyncio.from_url(settings.redis_url)
         + await app.state.redis.ping() so a dead Redis fails LOUDLY at
         startup, not silently at the first customer message.   [step 4]
      4. Checkpointer — the agent's memory:                      [step 6]
             from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
             app.state.saver_cm = AsyncPostgresSaver.from_conn_string(settings.database_url)
             saver = await app.state.saver_cm.__aenter__()
             await saver.setup()          ← creates ITS tables, first run only
         (from_conn_string is a context manager — keep it open for the app's
         lifetime, exactly like the MCP AsyncExitStack lesson.)
      5. LLM + tools + graph:
             llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
             app.state.graph = build_graph(llm, make_tools(...), SYSTEM_PROMPT, saver)
         NOTE make_tools needs a psid → per-request tools. Two options,
         both fine: (a) build the graph per-request in the worker;
         (b) pass psid through tool args via InjectedState (advanced).
         Start with (a) — measure before optimizing.
      6. app.state.http = httpx.AsyncClient(timeout=30)  (Send API client)
      7. yield  ← the app runs
      8. Shutdown, reverse order: aclose http client, __aexit__ the saver,
         aclose redis, dispose_engine().
    """
    settings = get_settings()
    await init_engine(settings.async_database_url)
    app.state.redis = redis.asyncio.from_url(settings.redis_url)
    await app.state.redis.ping()
    app.state.saver_cm = AsyncPostgresSaver.from_conn_string(settings.database_url)
    await app.state.saver_cm.__aenter__()
    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
    saver = await app.state.saver_cm.__aenter__()
    await saver.setup()
    app.state.saver = saver
    app.state.http = httpx.AsyncClient(timeout=30)
    logger.info("skeleton mode: nothing initialized yet - implement lifespan as you progress")
    yield
    await app.state.http.aclose()
    await app.state.saver_cm.__aexit__(None, None, None)
    await app.state.redis.aclose()
    await dispose_engine()
    logger.info("all components shut down successfully")


app = FastAPI(title="Messenger AI Agent", lifespan=lifespan)
app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict:
    """Provided: extend it as you light up components (report db/redis/agent
    readiness the way the debugger's /health reported mcp_connected)."""
    return {"status": "ok", "mode": "skeleton"}
