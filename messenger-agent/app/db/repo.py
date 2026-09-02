"""Queries — the only file that writes SQL. YOU implement all of it.
(Step 5 of the path)

Definition of done: pytest tests/test_repo.py
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Sequence
from sqlalchemy import Select, asc, desc, or_, func
from app.db.models import (  
    Customer,
    DeadLetter,
    MessageLog,
    Product,
)


async def save_message(
    session: AsyncSession, page_id: str, psid: str, role: str, text: str | None, meta: dict | None = None
) -> None:
    """Append one message to the log (role = "user" | "assistant" | "system").
    """
    new_user = MessageLog(page_id=page_id, psid=psid, role=role, text=text, meta=meta)
    session.add(new_user)
    await session.commit()


async def get_recent_messages(session: AsyncSession, psid: str, limit: int = 50) -> list[MessageLog]:
    """Latest messages for one customer, OLDEST FIRST (for display).
    """
    stmt = select(MessageLog).where(psid==MessageLog.psid).order_by(MessageLog.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(reversed(result.scalars().all()))



async def save_dead_letter(session: AsyncSession, raw: str, error: str) -> None:
    """Store a payload we couldn't parse — your future schema-extension c
    """
    new_error = DeadLetter(raw=raw, error=error)
    session.add(new_error)
    await session.commit()


async def get_or_create_customer(session: AsyncSession, page_id: str, psid: str) -> Customer:

    if customer := await session.get(Customer, psid):
        return customer
    new_user = Customer(page_id=page_id, psid=psid)
    session.add(new_user)
    await session.commit()
    return new_user


async def set_handoff(session: AsyncSession, page_id: str, psid: str, active: bool) -> None:
    """Flip the human-handoff flag. 
    While True, the worker must NOT answer
    """
    customer = await get_or_create_customer(session, page_id, psid)
    customer.handoff_active = active
    await session.commit()




async def find_products(
    session: AsyncSession,
    *,
    page_id: str,
    keywords: Optional[str] = None,
    max_price: Optional[int] = None,
    min_price: Optional[int] = None,
    in_stock_only: bool = False,
    sort: str = "relevance",
    limit: int = 5,
) -> Sequence[Product]:
    """Filtered catalog search. Every argument is optional except the tenant.

    page_id is keyword-only and REQUIRED. Making it positional invites the
    call that forgets it, which is how the current find_products ends up
    searching every store's catalog.
    """
    stmt: Select = select(Product).where(Product.page_id == page_id)

    if keywords:
        # Per-word AND across name AND description.
        #
        # WHY NOT ONE ILIKE ON THE WHOLE STRING: '%sac beige%' misses "Sac
        # Toile Beige" because a word sits between the two search terms.
        #
        # WHY DESCRIPTION TOO: colour and size are not in the name for most
        # catalogs. "red t-shirt" has to reach "T-Shirt Coton" whose
        # description says rouge. This is what makes a colour filter work
        # without a colour column.
        for word in (w for w in keywords.split() if len(w) > 1):
            pattern = f"%{word}%"
            stmt = stmt.where(
                or_(
                    Product.name.ilike(pattern),
                    Product.description.ilike(pattern),
                )
            )

    # price_da is an INTEGER column. Comparing the display string does not
    # work: '10000 DA' < '900 DA' is True, so a 10000 DA coat answers
    # "anything under 900".
    if max_price is not None:
        stmt = stmt.where(Product.price <= max_price)
    if min_price is not None:
        stmt = stmt.where(Product.price >= min_price)

    if in_stock_only:
        stmt = stmt.where(Product.stock > 0)

    if sort == "price_asc":
        stmt = stmt.order_by(asc(Product.price))
    elif sort == "price_desc":
        stmt = stmt.order_by(desc(Product.price))
    else:
        # "relevance" = in-stock first, then cheapest. There is no ranking
        # signal in ILIKE, so ordering by what the customer can actually buy
        # beats an arbitrary order — an out-of-stock top result reads as
        # a bot that does not know its own shop.
        stmt = stmt.order_by(desc(Product.stock > 0), asc(Product.price))

    stmt = stmt.limit(min(max(limit, 1), MAX_LIMIT))
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def count_products(session: AsyncSession, *, page_id: str, **filters) -> int:
    """How many rows MATCH, before the limit.

    Needed so the agent can say "3andna 14 haja taht 1000 DA, hadu l'arkhas
    5" instead of implying five is everything. Without it, a truncated list
    reads as the whole catalog.
    """
    from sqlalchemy import func

    stmt = select(func.count()).select_from(Product).where(Product.page_id == page_id)
    if filters.get("keywords"):
        for word in (w for w in filters["keywords"].split() if len(w) > 1):
            pattern = f"%{word}%"
            stmt = stmt.where(
                or_(Product.name.ilike(pattern), Product.description.ilike(pattern))
            )
    if filters.get("max_price") is not None:
        stmt = stmt.where(Product.price <= filters["max_price"])
    if filters.get("min_price") is not None:
        stmt = stmt.where(Product.price >= filters["min_price"])
    if filters.get("in_stock_only"):
        stmt = stmt.where(Product.stock > 0)
    return int((await session.execute(stmt)).scalar_one())
