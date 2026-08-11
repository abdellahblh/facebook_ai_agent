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
import os
from pinecone import AsyncPinecone

def make_tools(psid: str, page_id: str, session_factory: async_sessionmaker) -> list[BaseTool]:
    """Build the per-conversation toolset."""

    @tool
    async def product_lookup(query: str) -> str:
        """Look up a product by name to get its EXACT price and stock.
        Use for any question about price, availability, or delivery of a product."""

        async with session_factory() as session:
            products = await repo.find_products(session, query)
        if not products:
            return f"No product found matching '{query}'."
        return "\n".join(f"{p.name} — {p.price} — in stock: {p.stock}" for p in products)

    @tool
    async def policy_search(question: str) -> str:
        """Search company policies (returns, delivery, warranty, payment)."""

        async with AsyncPinecone(api_key=os.getenv("PINECONE_API_KEY")) as pc:
            # Await the Index object
            index = await pc.Index("facebook")

            # Generate query embedding asynchronously (384 dimensions)
            query_embed_res = await pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[question],
                parameters={
                    "input_type": "query",  # Use "query" for search queries
                    "dimension": 384,
                },
            )
            query_vector = query_embed_res.data[0].values

            # Search Pinecone asynchronously
            results = await index.query(vector=query_vector, top_k=3, namespace="faq", include_metadata=True)

            # If no results found
            if not results.matches:
                return "No matching store policies found."

            # Format retrieved matches into a clean string for the LLM
            context_chunks = []
            for i, match in enumerate(results.matches, 1):
                text = match.metadata.get("text", "No text content available")
                section = match.metadata.get("section", "General")
                context_chunks.append(f"--- Result {i} (Section: {section}) ---\n{text}")

            return "\n\n".join(context_chunks)

    @tool
    async def handoff_to_human(reason: str) -> str:
        """Escalate this conversation to a human teammate. Use when the
        customer is angry, stuck, or explicitly asks for a person."""

        async with session_factory() as session:
            await repo.set_handoff(session, page_id, psid, True)
            return "Handoff activated."

    return [product_lookup, policy_search, handoff_to_human]
