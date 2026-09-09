"""FastAPI app assembly: build every shared client ONCE at startup.

Everything the worker reads off app.state is created here. When a customer got
a permanent apology reply for a whole evening, the cause was app.state.llm
never being assigned — so the health endpoint below reports what is actually
wired, instead of a hardcoded "ok".
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pinecone import AsyncPinecone
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agent.security import nemo_guardrails
from app.chatwoot import router as chatwoot_router
from app.config import get_settings
from app.db import engine as db_engine
from app.db.engine import dispose_engine, init_engine
from app.kb import router as kb_router

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

    # Cloud Redis configuration — network latency and connection limits matter
    app.state.redis = aioredis.from_url(
        settings.redis_url,
        decode_responses=False,
        socket_timeout=10.0,           # Cloud latency can be 50-200ms, need buffer
        socket_connect_timeout=5.0,    # Cloud connections take longer to establish
        socket_keepalive=True,         # Prevent cloud provider from dropping idle connections
        max_connections=10,            # Cloud providers often limit connections (10-50)
        retry_on_timeout=True,
        health_check_interval=30,      # Validate connections before use
    )
    await app.state.redis.ping()

    from app.cache.layers.l0_memory import L0InMemoryCache
    from app.cache.layers.l1_semantic import L1SemanticCache, LangChainEmbeddingProvider
    from app.cache.layers.l2_redis import L2RedisKVCache
    from app.cache.manager import CacheManager

    semantic_cache = None
    if settings.cache_embedding_model and settings.cache_embedding_dimensions > 0:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.cache_embedding_model,
                google_api_key=settings.google_api_key,
            )
            semantic_cache = L1SemanticCache(
                app.state.redis,
                LangChainEmbeddingProvider(embeddings),
                dimensions=settings.cache_embedding_dimensions,
                similarity_threshold=settings.cache_semantic_threshold,
                index_name=settings.cache_semantic_index,
            )
        except Exception:
            logger.exception("Semantic cache unavailable; continuing with L0/L2.")

    app.state.cache_manager = CacheManager(
        l0=L0InMemoryCache(
            max_size=settings.cache_l0_max_size,
            default_ttl_seconds=settings.agent_response_cache_ttl_seconds,
        ),
        l2=L2RedisKVCache(
            app.state.redis,
            default_ttl_seconds=settings.agent_response_cache_ttl_seconds,
        ),
        l1=semantic_cache,
        lock_ttl_seconds=settings.cache_lock_ttl_seconds,
        lock_wait_seconds=settings.cache_lock_wait_seconds,
    )
    await app.state.cache_manager.initialize()


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

    # Queue consumer: drain the webhook stream in-process. Started last so the
    # guardrail "fail fast" above means no consumer is left half-wired.
    if settings.queue_enabled:
        from app.worker import run_queue_consumer

        consumer_name = f"{socket.gethostname()}-{os.getpid()}"
        app.state.queue_consumer_name = consumer_name
        app.state.queue_consumer_task = asyncio.create_task(
            run_queue_consumer(
                app.state.redis,
                app.state.session_factory,
                consumer_name=consumer_name,
            )
        )
        logger.info("Queue consumer started (%s).", consumer_name)
    else:
        app.state.queue_consumer_task = None

    logger.info("Startup complete: db, redis, checkpointer, llm, agent_graph, http, NeMo Guardrails, and webhook queue wired.")
    yield

    # Stop the consumer FIRST: it must not be processing an entry while Redis,
    # Postgres, or HTTP are being torn down under it. Cancel, then wait.
    consumer_task = getattr(app.state, "queue_consumer_task", None)
    if consumer_task is not None:
        consumer_task.cancel()
        try:
            await asyncio.wait_for(consumer_task, timeout=10)
            logger.info("Queue consumer stopped.")
        except asyncio.CancelledError:
            logger.info("Queue consumer cancelled.")
        except TimeoutError:
            logger.warning("Queue consumer ignored cancellation — continuing shutdown.")

    closers = [
        ("http", app.state.http.aclose),
        ("pg_pool", app.state.pg_pool.close),
        ("redis", app.state.redis.aclose),
        ("engine", dispose_engine),
        ("nemoguardrails", nemo_guardrails.close),
    ]
    # Pinecone is optional: closing None would raise AttributeError and abort
    # the whole shutdown before the remaining closers run.
    if app.state.pinecone is not None:
        closers.append(("pinecone", app.state.pinecone.close))

    for name, closer in closers:
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Error closing %s during shutdown.", name)
    from langfuse import get_client
    langfuse = get_client()
    langfuse.flush()


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

    settings = get_settings()
    queue = {
        "queue_enabled": settings.queue_enabled,
        "consumer_alive": False,
    }
    consumer_task = getattr(state, "queue_consumer_task", None)
    if consumer_task is not None:
        queue["consumer_alive"] = not consumer_task.done()
    if settings.queue_enabled and getattr(state, "redis", None) is not None:
        try:
            from app.cache import streams

            queue.update(await streams.backlog(state.redis))
            queue["consumer"] = getattr(state, "queue_consumer_name", None)
        except Exception:
            logger.exception("Failed to read queue metrics for /health.")
            queue["redis_available"] = False

    cache = {"enabled": getattr(state, "cache_manager", None) is not None}
    cache_manager = getattr(state, "cache_manager", None)
    if cache_manager is not None:
        try:
            cache["metrics"] = (await cache_manager.snapshot()).model_dump()
        except Exception:
            logger.exception("Failed to read cache metrics for /health.")
            cache["metrics_available"] = False

    return {"status": "ok" if all(components.values()) else "degraded", **components, "queue": queue, "cache": cache}
