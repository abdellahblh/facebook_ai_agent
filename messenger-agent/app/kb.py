"""Knowledge-base admin API — backs the Chatwoot Knowledge Base settings pages.

CONTRACT (must match dashboard/composables/useKnowledgeBase.js exactly)
    GET    /kb/products?page_id=&q=&token=
    POST   /kb/products?token=            body: name price stock description page_id
    PUT    /kb/products/{id}?token=       body: same
    DELETE /kb/products/{id}?token=       body: none  → page_id via query
    GET    /kb/policies?page_id=&token=
    POST   /kb/policies?token=            body: title body published page_id
    DELETE /kb/policies/{id}?token=       body: none  → page_id via query
    GET    /kb/panel                      the iframe page (no token: it is a shell)

TWO THINGS THE PREVIOUS VERSION GOT WRONG
    1. NO AUTH. Every endpoint was open to the internet. Anyone who found the
       host could rewrite the prices the assistant quotes to customers.
    2. NO TENANT SCOPING on products. `page_id` was absent from ProductIn, so
       Pydantic silently DROPPED the value the UI sends, every product landed
       under the column default, and store A's catalog answered store B's
       customers. Silent because nothing errors — the data is just wrong.

WHY page_id IS A QUERY PARAM ON DELETE
    A DELETE has no body in this client. Without page_id we cannot verify the
    row belongs to the caller's tenant, so any id would be deletable by anyone
    who knows it.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, or_, select

from app.config import get_settings
from app.db import engine as db_engine
from app.db.models import Policy, Product

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kb", tags=["knowledge-base"])

PANEL_HTML = Path(__file__).parent / "static" / "kb_panel.html"


# ── auth ─────────────────────────────────────────────────────────────────────
def require_token(token: str = Query(default="")) -> None:
    """Gate every data endpoint on a shared secret.

    FAILS CLOSED. If kb_admin_token is unset the answer is 503, never "allow".
    A misconfigured deployment must be unusable, not unprotected — the UI has
    a distinct message for this case so it is visible rather than silent.

    compare_digest, not `==`: string equality returns as soon as it finds a
    differing byte, so response time leaks how many leading characters were
    correct. That turns brute force from 62^32 into 32*62.
    """
    import secrets

    expected = get_settings().kb_admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="KB_ADMIN_TOKEN not configured")
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="Invalid token")


Auth = Depends(require_token)


# ── payloads ─────────────────────────────────────────────────────────────────
class ProductIn(BaseModel):
    # page_id is REQUIRED. The UI always sends it; making it optional is how
    # the tenant leak happened, because a missing value became the default.
    page_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    price: str = Field(min_length=1, max_length=64)
    stock: int = Field(default=0, ge=0)
    description: str | None = None


class PolicyIn(BaseModel):
    page_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1)
    published: bool = True


def _as_uuid(raw: str) -> uuid.UUID:
    """The id column is Uuid, not String. Passing the raw str makes SQLAlchemy
    call .hex on it and blow up mid-statement — a 500 where a 400 belongs."""
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="Malformed id") from None


# ── the panel ────────────────────────────────────────────────────────────────



# ── products ─────────────────────────────────────────────────────────────────
@router.get("/products", dependencies=[Auth])
async def list_products(page_id: str, q: str | None = None) -> list[dict]:
    """This tenant's products, optionally filtered.

    Filtering server-side: a 500-item catalog should not be shipped to the
    panel on every keystroke.
    """
    async with db_engine.SessionFactory() as session:
        stmt = select(Product).where(Product.page_id == page_id)
        if q and q.strip():
            # Both columns: owners search by model name AND by attributes they
            # typed into the description ("leather", "size XL").
            pattern = f"%{q.strip()}%"
            stmt = stmt.where(
                or_(Product.name.ilike(pattern), Product.description.ilike(pattern))
            )
        rows = (await session.execute(stmt.order_by(Product.name))).scalars().all()
    return [
        {
            "id": str(p.id),
            "name": p.name,
            "price": p.price,
            "stock": p.stock,
            "description": p.description,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        }
        for p in rows
    ]


@router.post("/products", dependencies=[Auth])
async def upsert_product(payload: ProductIn) -> dict:
    async with db_engine.SessionFactory() as session:
        existing = (
            await session.execute(
                select(Product).where(
                    Product.page_id == payload.page_id,
                    Product.name == payload.name,
                )
            )
        ).scalar_one_or_none()
        if existing:
            # Upsert on (page_id, name): re-entering a product updates it
            # instead of creating a duplicate the assistant might quote.
            existing.price = payload.price
            existing.stock = payload.stock
            existing.description = payload.description
            product = existing
        else:
            product = Product(**payload.model_dump())
            session.add(product)
        await session.commit()
        return {"id": str(product.id), "status": "saved"}


@router.put("/products/{product_id}", dependencies=[Auth])
async def update_product(product_id: str, payload: ProductIn) -> dict:
    """Update BY ID, so renaming works.

    POST upserts on (page_id, name) — editing a name there creates a second
    product and leaves the old row behind for the assistant to quote.
    """
    pid = _as_uuid(product_id)
    async with db_engine.SessionFactory() as session:
        product = await session.get(Product, pid)
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        if product.page_id != payload.page_id:
            # 403, not 404: the row exists, the caller just does not own it.
            raise HTTPException(status_code=403, detail="Tenant mismatch")

        product.name = payload.name
        product.price = payload.price
        product.stock = payload.stock
        product.description = payload.description
        await session.commit()
        return {"id": str(product.id), "status": "updated"}


@router.delete("/products/{product_id}", dependencies=[Auth])
async def delete_product(product_id: str, page_id: str) -> dict:
    """page_id is a required query param, not optional.

    Without it this deletes by id alone, so any tenant could remove any other
    tenant's product given the id — and ids appear in the UI's own HTML.
    """
    # Validate the id BEFORE opening a session. Inside the `async with`, a
    # malformed id raises mid-transaction — wasted connection, and the 400
    # travels back up through a context manager that is rolling back.
    pid = _as_uuid(product_id)
    async with db_engine.SessionFactory() as session:
        result = await session.execute(
            delete(Product).where(
                Product.id == pid,
                Product.page_id == page_id,
            )
        )
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"status": "deleted"}


# ── policies ─────────────────────────────────────────────────────────────────
@router.get("/policies", dependencies=[Auth])
async def list_policies(page_id: str) -> list[dict]:
    async with db_engine.SessionFactory() as session:
        rows = (
            await session.execute(
                select(Policy).where(Policy.page_id == page_id).order_by(Policy.title)
            )
        ).scalars().all()
    return [
        {
            "id": str(p.id),
            "title": p.title,
            "body": p.body,
            "published": p.published,
        }
        for p in rows
    ]


@router.post("/policies", dependencies=[Auth])
async def upsert_policy(payload: PolicyIn) -> dict:
    async with db_engine.SessionFactory() as session:
        existing = (
            await session.execute(
                select(Policy).where(
                    Policy.page_id == payload.page_id,
                    Policy.title == payload.title,
                )
            )
        ).scalar_one_or_none()
        if existing:
            existing.body = payload.body
            existing.published = payload.published
            policy = existing
        else:
            policy = Policy(**payload.model_dump())
            session.add(policy)
        await session.commit()
        return {"id": str(policy.id), "status": "saved"}


@router.delete("/policies/{policy_id}", dependencies=[Auth])
async def delete_policy(policy_id: str, page_id: str) -> dict:
    pid = _as_uuid(policy_id)
    async with db_engine.SessionFactory() as session:
        result = await session.execute(
            delete(Policy).where(
                Policy.id == pid,
                Policy.page_id == page_id,
            )
        )
        await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Policy not found")
    return {"status": "deleted"}
