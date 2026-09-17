"""Agent tools. YOU implement the bodies. (Step 6 of the path)

Design (from the architecture review):
  exact facts   → product_lookup → SQL          (never embeddings)
    fuzzy known.  → policy_search  → REDIS VECTOR STORE
  escape hatch  → handoff_to_human

The factory pattern: tools need the psid and a DB session, but the LLM only
fills the (query/reason) arguments. A closure per request carries the context.

Definition of done: pytest tests/test_agent.py
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

import httpx
from langchain_core.runnables import RunnableConfig
from redis.commands.search.query import Query
from redis import Redis
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.nlp.arabizi import strip_darija_stopwords
from app.agent.guardrail import EmbedderProvider
from app.db import repo
from app.config import get_settings
INDEX_NAME = "store-policies-jina"
logger = logging.getLogger(__name__)


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


@tool
async def search_products(
    keywords: str | None = None,
    max_price: int | None = None,
    min_price: int | None = None,
    in_stock_only: bool = False,
    sort: str = "relevance",
    config: RunnableConfig = None,
) -> str:
    """Search this shop's catalog. Returns exact prices and stock counts.

    Use for ANY product question: a specific item, a budget, a colour, a
    size, "what is cheapest", "what do you have". Combine arguments
    freely — one call can express "red t-shirt under 1000 in stock".

    Args:
        keywords: Product words only, no filler. Colours and materials
            work because descriptions are searched too. Say "veste rouge",
            not "do you have the red jacket". Omit to browse everything.
        max_price: Upper bound in dinars, as a NUMBER: 1000, not "1000 DA".
        min_price: Lower bound in dinars.
        in_stock_only: True when the customer wants to buy now. Leave
            False when they are only asking about price.
        sort: "relevance" (default, in-stock first then cheapest),
            "price_asc" for "what is cheapest", "price_desc" for
            "what is your best/most expensive".

    Returns one line per product, plus a total count when more matched
    than are shown. Quote these prices EXACTLY. If it returns nothing,
    say so and offer the handoff — never estimate.
    """
    configurable = (config or {}).get("configurable", {})
    page_id = configurable.get("page_id", "")
    session_factory = configurable.get("session_factory")
    if not session_factory:
        from app.main import app
        session_factory = getattr(app.state, "session_factory", None)

    if not session_factory or not page_id:
        logger.error("search_products failed: missing session_factory or page_id")
        return "No products in the catalog."

    limit = 5 if (keywords or max_price or min_price) else 8
    cleaned_kw = strip_darija_stopwords(keywords) if keywords else None

    async with session_factory() as session:
        rows = await repo.find_products(
            session,
            page_id=page_id,
            keywords=cleaned_kw,
            max_price=max_price,
            min_price=min_price,
            in_stock_only=in_stock_only,
            sort=sort,
            limit=limit,
        )
        total = await repo.count_products(
            session,
            page_id=page_id,
            keywords=cleaned_kw,
            max_price=max_price,
            min_price=min_price,
            in_stock_only=in_stock_only,
        )

    if not rows:
        applied = []
        if keywords:      applied.append(f"matching '{keywords}'")
        if max_price:      applied.append(f"under {max_price} DA")
        if min_price:      applied.append(f"over {min_price} DA")
        if in_stock_only:  applied.append("in stock")
        return "No products " + (" and ".join(applied) or "in the catalog") + "."

    lines = [
        f"{p.name} — {p.price} — in stock: {p.stock}"
        + ("" if p.stock else "  (OUT OF STOCK)")
        for p in rows
    ]
    if total > len(rows):
        lines.append(f"({len(rows)} of {total} matching products shown.)")
    return "\n".join(lines)


@tool
async def policy_search(question: str, config: RunnableConfig = None) -> str:
    """Search the tenant's published policies.

    Policies are embedded in redis's ``faq`` namespace by the RAG ingest
    job. PostgreSQL remains the source for structured catalog and customer
    data, but policy retrieval needs semantic search for natural-language
    questions.
    """
    
    limit = 3
    configurable = (config or {}).get("configurable", {})
    embedder = configurable.get("embedder")
    redis = configurable.get("redis")

    if not embedder or not redis:
        from app.main import app
        embedder = embedder or getattr(app.state, "embedder", None)
        redis = redis or getattr(app.state, "redis", None)

    if not embedder or not redis:
        logger.error("policy_search failed: missing embedder or redis")
        return "Policy search unavailable."

    query_vector = await embedder.embed_query(question)

    index_name = configurable.get("INDEX_NAME", INDEX_NAME)
    query = (
        Query(f'(*)=>[KNN $limit @embedding $query_vector AS distance]')
        .sort_by("distance")
        .return_fields("text", "source", "section_title", "distance")
        .paging(0, limit)
        .dialect(2)
    )
    result = await redis.ft(index_name).search(
        query,
        query_params={"query_vector": _pack_vector(query_vector), "limit": limit},
    )
    logger.info(f"Policy search result: {result}")
    if not result.docs:
        return "No policy information found for that question."
    lines = []
    for doc in result.docs:
        if float(doc.distance) < 0.10:
            lines.append(f"[{doc.source}] {doc.text}")
    if not lines:
        return "No policy information found for that question."
    return "\n".join(lines)


@tool
async def handoff_to_human(reason: str, config: RunnableConfig = None) -> str:
    """Escalate this conversation to a human teammate. Use when the
    customer is angry, stuck, or explicitly asks for a person."""
    configurable = (config or {}).get("configurable", {})
    page_id = configurable.get("page_id", "")
    psid = configurable.get("psid", "")
    channel = configurable.get("channel", "messenger")
    account_id = configurable.get("account_id")
    conversation_id = configurable.get("conversation_id")

    session_factory = configurable.get("session_factory")
    http_client = configurable.get("http_client")
    if not session_factory or not http_client:
        from app.main import app
        session_factory = session_factory or getattr(app.state, "session_factory", None)
        http_client = http_client or getattr(app.state, "http", None)

    if session_factory and page_id and psid:
        async with session_factory() as session:
            await repo.set_handoff(session, page_id, psid, True)

    if http_client and account_id and conversation_id:
        from app import chatwoot

        if channel == "chatwoot":
            await chatwoot.escalate_to_human(
                http_client,
                account_id,
                conversation_id,
                note=f"AI escalated conversation to human. Reason: {reason}",
            )
        else:
            note_ok = await chatwoot.send_reply(
                http_client,
                account_id,
                conversation_id,
                f"AI escalated conversation to human. Reason: {reason}",
                private=True,
            )
            if not note_ok:
                logger.warning(
                    "Failed to send handoff private note for conversation %s",
                    conversation_id,
                )
    elif http_client:
        logger.warning(
            "handoff_to_human: missing account_id/conversation_id for channel %s — "
            "cannot send private note to Chatwoot.",
            channel,
        )

    return "Handoff activated."


def get_tools() -> list[BaseTool]:
    """Static tools list for graph compilation at application startup."""
    return [search_products, policy_search, handoff_to_human]


def make_tools(
    psid: str,
    page_id: str,
    session_factory: async_sessionmaker,
    channel: str = "messenger",
    account_id: int | None = None,
    conversation_id: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    redis_client=None,
    INDEX_NAME=None,
    embedder=None,
) -> list[BaseTool]:
    """Factory for tests and manual construction with bound fallback context."""

    @tool
    async def search_products_bound(
        keywords: str | None = None,
        max_price: int | None = None,
        min_price: int | None = None,
        in_stock_only: bool = False,
        sort: str = "relevance",
        config: RunnableConfig = None,
    ) -> str:
        """Search this shop's catalog. Returns exact prices and stock counts."""
        cfg = dict((config or {}).get("configurable", {}))
        cfg.setdefault("page_id", page_id)
        cfg.setdefault("session_factory", session_factory)
        return await search_products.ainvoke(
            {
                "keywords": keywords,
                "max_price": max_price,
                "min_price": min_price,
                "in_stock_only": in_stock_only,
                "sort": sort,
            },
            config={"configurable": cfg},
        )

    @tool
    async def policy_search_bound(
        question: str,
        config: RunnableConfig = None,
    ) -> str:
        """Search the tenant's published policies."""
        cfg = dict((config or {}).get("configurable", {}))
        cfg.setdefault("page_id", page_id)
        cfg.setdefault("session_factory", session_factory)
        cfg.setdefault("redis", redis_)
        cfg.setdefault("embedder", embedder)
        cfg.setdefault("INDEX_NAME", INDEX_NAME)
        return await policy_search.ainvoke(
            {"question": question},
            config={"configurable": cfg},
        )

    @tool
    async def handoff_to_human_bound(
        reason: str,
        config: RunnableConfig = None,
    ) -> str:
        """Escalate this conversation to a human teammate."""
        cfg = dict((config or {}).get("configurable", {}))
        cfg.setdefault("page_id", page_id)
        cfg.setdefault("psid", psid)
        cfg.setdefault("session_factory", session_factory)
        cfg.setdefault("http_client", http_client)
        cfg.setdefault("channel", channel)
        cfg.setdefault("account_id", account_id)
        cfg.setdefault("conversation_id", conversation_id)
        return await handoff_to_human.ainvoke(
            {"reason": reason},
            config={"configurable": cfg},
        )

    return [search_products_bound, policy_search_bound, handoff_to_human_bound]



