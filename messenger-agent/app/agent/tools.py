"""Agent tools. YOU implement the bodies. (Step 6 of the path)

Design (from the architecture review):
  exact facts   → product_lookup → SQL          (never embeddings)
  fuzzy known.  → policy_search  → text/pgvector
  escape hatch  → handoff_to_human

The factory pattern: tools need the psid and a DB session, but the LLM only
fills the (query/reason) arguments. A closure per request carries the context.

Definition of done: pytest tests/test_agent.py
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import httpx
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.nlp.arabizi import strip_darija_stopwords, build_search_query,looks_like_arabizi

from app.db import repo

try:
    from pinecone import AsyncPinecone
except ImportError:
    AsyncPinecone = None

logger = logging.getLogger(__name__)


def make_tools(
    psid: str,
    page_id: str,
    session_factory: async_sessionmaker,
    channel: str = "messenger",
    account_id: int | None = None,
    conversation_id: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[BaseTool]:
    """Build the per-conversation toolset."""

    @tool
    async def search_products(
        keywords: str | None = None,
        max_price: int | None = None,
        min_price: int | None = None,
        in_stock_only: bool = False,
        sort: str = "relevance",
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
        # Guardrail: a bare "3andkom?" with no filters would otherwise dump
        # the catalog into the prompt.
        limit = 5 if (keywords or max_price or min_price) else 8

        async with session_factory() as session:
            rows = await repo.find_products(
                session,
                page_id=page_id,
                keywords=keywords,
                max_price=max_price,
                min_price=min_price,
                in_stock_only=in_stock_only,
                sort=sort,
                limit=limit,
            )
            total = await repo.count_products(
                session,
                page_id=page_id,
                keywords=keywords,
                max_price=max_price,
                min_price=min_price,
                in_stock_only=in_stock_only,
            )

        if not rows:
            # Say WHICH filters found nothing. "No product found" makes the
            # model guess whether the name or the budget was the problem, and
            # a guess here becomes "we don't sell that" for something you do.
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
            # Without this the model presents 5 rows as the whole answer.
            lines.append(f"({len(rows)} of {total} matching products shown.)")
        return "\n".join(lines)


    @tool
    async def policy_search(question: str) -> str:
        """Search company policies (returns, delivery, warranty, payment)."""
        if AsyncPinecone is None:
            return "Policy search is temporarily unavailable."

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            return "Policy search API key is not configured."

        try:
            async with AsyncPinecone(api_key=pinecone_api_key) as pc:
                index = await pc.Index("facebook")

                query_embed_res = await pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[build_search_query(question) if looks_like_arabizi(question) else question],
                    parameters={
                        "input_type": "query",
                        "dimension": 384,
                    },
                )
                query_vector = query_embed_res.data[0].values

                results = await index.query(
                    vector=query_vector,
                    top_k=3,
                    namespace="faq",
                    include_metadata=True,
                )

                if not results.matches:
                    return "No matching store policies found."

                context_chunks = []
                for i, match in enumerate(results.matches, 1):
                    text = match.metadata.get("text", "No text content available")
                    section = match.metadata.get("section", "General")
                    context_chunks.append(f"--- Result {i} (Section: {section}) ---\n{text}")

                return "\n\n".join(context_chunks)
        except Exception as err:
            logger.exception("Error in policy search: %s", err)
            return "Could not retrieve store policies at this time."

    @tool
    async def handoff_to_human(reason: str) -> str:
        """Escalate this conversation to a human teammate. Use when the
        customer is angry, stuck, or explicitly asks for a person."""

        async with session_factory() as session:
            await repo.set_handoff(session, page_id, psid, True)

        if channel == "chatwoot" and account_id and conversation_id and http_client:
            from app import chatwoot
            await chatwoot.escalate_to_human(
                http_client,
                account_id,
                conversation_id,
                note=f"AI escalated conversation to human. Reason: {reason}",
            )
        return "Handoff activated."

    return [search_products, policy_search, handoff_to_human]
