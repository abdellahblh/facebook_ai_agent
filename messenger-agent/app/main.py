"""FastAPI app assembly — wiring provided; lifespan startup is YOURS.

Run: uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver 
from app.webhook import router as webhook_router
from app.config import get_settings  
from app.db.engine import init_engine
import httpx
import redis.asyncio as aioredis
from langchain_google_genai import ChatGoogleGenerativeAI
from app.db import engine as db_engine
from app.schemas import InboundMessage
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_core.tools import tool
from app.agent.tools import make_tools
from app.agent.graph import build_graph
from app.agent.prompts import SYSTEM_PROMPT
from app.db.engine import dispose_engine
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build everything ONCE at startup; tear down cleanly at shutdown.

  
    """
    settings = get_settings()
    await init_engine(settings.async_database_url)
    app.state.session_factory = db_engine.SessionFactory
    app.state.redis = aioredis.from_url(settings.redis_url)
    await app.state.redis.ping()
    app.state.saver_cm = AsyncPostgresSaver.from_conn_string(settings.database_url)
    app.state.llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
    app.state.session_factory = db_engine.SessionFactory
    app.state.saver = await app.state.saver_cm.__aenter__()
    await app.state.saver.setup()
    app.state.http = httpx.AsyncClient(timeout=30)
    yield
    await app.state.http.aclose()
    await app.state.saver_cm.__aexit__(None, None, None)
    await app.state.redis.aclose()
    await dispose_engine()


app = FastAPI(title="Messenger AI Agent", lifespan=lifespan)
app.include_router(webhook_router)

@app.get("/health/redis")
async def redis_health(request: Request):
    try:
        pong = await request.app.state.redis.ping()
        return {"redis": "connected", "ping": pong}
    except Exception as e:
        return {"redis": "disconnected", "error": str(e)}

@app.get("/health")
async def health() -> dict:
    """Provided: extend it as you light up components (report db/redis/agent
    readiness the way the debugger's /health reported mcp_connected)."""
    return {"status": "ok", "mode": "skeleton"}
