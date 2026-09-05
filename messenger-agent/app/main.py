"""FastAPI app assembly: build every shared client ONCE at startup.

Everything the worker reads off app.state is created here. When a customer got
a permanent apology reply for a whole evening, the cause was app.state.llm
never being assigned — so the health endpoint below reports what is actually
wired, instead of a hardcoded "ok".
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pinecone import AsyncPinecone
from app.chatwoot import router as chatwoot_router
from app.config import get_settings
from app.db import engine as db_engine
from app.db.engine import dispose_engine, init_engine
from app.kb import router as kb_router
from fastapi.middleware.cors import CORSMiddleware
from app.agent.security import nemo_guardrails

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    await init_engine(settings.async_database_url)
    app.state.session_factory = db_engine.SessionFactory

    app.state.pg_pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=2,
        max_size=10,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": None,
        },
        check=AsyncConnectionPool.check_connection,
        max_idle=60,
        max_lifetime=1800,
        open=False,
    )
    await app.state.pg_pool.open()
    # 3. Pinecone (Shared Client & Index)
    if AsyncPinecone and settings.pinecone_api_key:
        app.state.pinecone = AsyncPinecone(api_key=settings.pinecone_api_key)
        app.state.pinecone_index = await app.state.pinecone.index(settings.pinecone_index)
    else:
        app.state.pinecone = None
        app.state.pinecone_index = None

    app.state.saver = AsyncPostgresSaver(app.state.pg_pool)
    await app.state.saver.setup()

    app.state.redis = aioredis.from_url(settings.redis_url)
    await app.state.redis.ping()


    app.state.llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
    app.state.http = httpx.AsyncClient(timeout=30)

    from app.agent.graph import build_graph
    from app.agent.prompts import SYSTEM_PROMPT
    from app.agent.tools import get_tools

    app.state.agent_graph = build_graph(
        app.state.llm,
        get_tools(),
        SYSTEM_PROMPT.format(business_name=settings.business_name),
        checkpointer=app.state.saver,
    )

    nemo_guardrails.initialize()
    app.state.nemo_guardrails = nemo_guardrails

    logger.info("Startup complete: db, redis, checkpointer, llm, agent_graph, http, and NeMo Guardrails wired.")
    yield

    for name, closer in (
        ("http", app.state.http.aclose()),
        ("pg_pool", app.state.pg_pool.close()),
        ("redis", app.state.redis.aclose()),
        ("engine", dispose_engine()),
        ("pinecone", app.state.pinecone.close()),
        ("nemoguardrails", app.state.nemo_guardrails.close()),
    ):
        try:
            await closer
        except Exception:
            logger.exception("Error closing %s during shutdown.", name)


app = FastAPI(title="Messenger AI Agent", lifespan=lifespan)
app.include_router(chatwoot_router)
app.include_router(kb_router)

origins = [
    "http://localhost:3000",
    "https://chat.dzvoixoff.online"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)



@app.get("/health")
async def health(request: Request) -> dict:
    """Report what is ACTUALLY wired. A hardcoded {"status": "ok"} is how a
    half-started app looks healthy while every customer gets the apology."""
    state = request.app.state
    components = {
        "db": getattr(state, "session_factory", None) is not None,
        "pg_pool": getattr(state, "pg_pool", None) is not None and not state.pg_pool.closed,
        "redis": getattr(state, "redis", None) is not None,
        "checkpointer": getattr(state, "saver", None) is not None,
        "llm": getattr(state, "llm", None) is not None,
        "agent_graph": getattr(state, "agent_graph", None) is not None,
        "http": getattr(state, "http", None) is not None,
        "nemo_guardrails": getattr(state, "nemo_guardrails", None) is not None
        and state.nemo_guardrails.rails is not None,
    }
    
    return {"status": "ok" if all(components.values()) else "degraded", **components}
