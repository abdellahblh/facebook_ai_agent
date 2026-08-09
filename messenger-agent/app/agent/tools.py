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

from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import async_sessionmaker
from app.db import repo

def make_tools(psid: str, session_factory: async_sessionmaker) -> list[BaseTool]:
    """Build the per-conversation toolset."""

    @tool
    async def product_lookup(query: str) -> str:
        """Look up a product by name to get its EXACT price and stock.
        Use for any question about price, availability, or delivery of a product."""
        # TODO(you) — steps:
        #   1. async with session_factory() as session:
        #   2.     products = await repo.find_products(session, query)
        #   3. If empty → return "No product found matching '<query>'."
        #      (The prompt's RULE 1 turns that into an honest answer + handoff.)
        #   4. Format one line per product: "Name — price — in stock: N".
        #      Return the joined string. Tools return STRINGS to the model.
        async with session_factory() as session:
            products = await repo.find_products(session, query)
        if not products:
            return f"No product found matching '{query}'."
        return "\n".join(f"{p.name} — {p.price} — in stock: {p.stock}" for p in products)

    @tool
    async def policy_search(question: str) -> str:
        """Search company policies (returns, delivery, warranty, payment)."""
        # TODO(you) — v1 steps:
        #   1. Store policies as rows (title, body) in a `policies` table you
        #      add to models.py — or even a dict in config for day one.
        #   2. Naive search: ILIKE on title/body with the question's keywords,
        #      return the best chunk's text.
        #   v2 (later): pgvector — embed chunks, cosine search, same Postgres.
        #   The tool's INTERFACE doesn't change when you upgrade — that's the
        #   point of hiding retrieval behind a tool.

    @tool
    async def handoff_to_human(reason: str) -> str:
        """Escalate this conversation to a human teammate. Use when the
        customer is angry, stuck, or explicitly asks for a person."""
        # TODO(you) — steps:
        #   1. async with session_factory() as session:
        #          await repo.set_handoff(session, psid, True)
        #   2. (later) notify the owner: email / their own Messenger / dashboard.
        #   3. Return "Handoff activated." — the model needs a string back.
        async with session_factory() as session:            
            await repo.set_handoff(session, psid, True)
            return "Handoff activated."

    return [product_lookup, policy_search, handoff_to_human]
