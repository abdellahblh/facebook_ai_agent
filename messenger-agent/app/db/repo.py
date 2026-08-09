"""Queries — the only file that writes SQL. YOU implement all of it.
(Step 5 of the path)

Definition of done: pytest tests/test_repo.py
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import (  # noqa: F401  (all used by your implementations)
    Customer,
    DeadLetter,
    MessageLog,
    Product,
)


async def save_message(
    session: AsyncSession, page_id: str, psid: str, role: str, text: str | None, meta: dict | None = None
) -> None:
    """Append one message to the log (role = "user" | "assistant" | "system").

    TODO(you): build a MessageLog(...), session.add(it), await session.commit().
    """
    new_user = MessageLog(page_id=page_id, psid=psid, role=role, text=text, meta=meta)
    session.add(new_user)
    await session.commit()


async def get_recent_messages(session: AsyncSession, psid: str, limit: int = 50) -> list[MessageLog]:
    """Latest messages for one customer, OLDEST FIRST (for display).

    TODO(you) — steps:
      1. select(MessageLog).where(...psid match...)
         .order_by(MessageLog.created_at.desc()).limit(limit)
         → newest-first page of the right SIZE...
      2. ...then reverse it in Python before returning, so the caller gets
         chronological order. (Classic trick: you can't ORDER BY asc + LIMIT
         and still get the LATEST N.)
      3. result = await session.execute(stmt); return list(reversed(result.scalars().all()))
    """
    stmt = select(MessageLog).where(psid==MessageLog.psid).order_by(MessageLog.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(reversed(result.scalars().all()))



async def save_dead_letter(session: AsyncSession, raw: str, error: str) -> None:
    """Store a payload we couldn't parse — your future schema-extension corpus.

    TODO(you): same shape as save_message.
    """
    new_error = DeadLetter(raw=raw, error=error)
    session.add(new_error)
    await session.commit()


async def get_or_create_customer(session: AsyncSession, page_id: str, psid: str) -> Customer:
    """TODO(you) — steps:
    1. customer = await session.get(Customer, psid)
    2. If None → create, add, commit.
    3. Return it.
    """
    if customer := await session.get(Customer, psid):
        return customer
    new_user = Customer(page_id=page_id, psid=psid)
    session.add(new_user)
    await session.commit()
    return new_user


async def set_handoff(session: AsyncSession, page_id: str, psid: str, active: bool) -> None:
    """Flip the human-handoff flag. While True, the worker must NOT answer.

    TODO(you): get_or_create, set the flag, commit.
    """
    customer = await get_or_create_customer(session, page_id, psid)
    customer.handoff_active = active
    await session.commit()




async def find_products(session: AsyncSession, query: str, limit: int = 5) -> list[Product]:
    """EXACT-FACTS lookup for the product_lookup tool. SQL, not embeddings.

    TODO(you) — steps:
      1. select(Product).where(Product.name.ilike(f"%{query}%")).limit(limit)
         (ilike = case-insensitive; the parameter binding protects you from
         SQL injection — never f-string a value INTO the SQL text itself)
      2. Execute, return list(result.scalars().all()).
    """
    stmt = select(Product).where(Product.name.ilike(f"%{query}%")).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())