"""Knowledge-base panel API. In-memory SQLite, no network, no Chatwoot."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import engine as db_engine
from app.db.models import Base

TOKEN = "test-token"


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", TOKEN)
    monkeypatch.setenv("CHATWOOT_WEBHOOK_SECRET", "x")
    from app.config import get_settings

    get_settings.cache_clear()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(
        db_engine, "SessionFactory", async_sessionmaker(engine, expire_on_commit=False)
    )
    import app.main

    yield TestClient(app.main.app)
    await engine.dispose()
    get_settings.cache_clear()


def q(path, **params):
    parts = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{path}?token={TOKEN}" + (f"&{parts}" if parts else "")


# ── auth: the only gate Chatwoot gives us ─────────────────────────────────────

async def test_wrong_token_rejected(client):
    assert client.get("/kb/products?page_id=2&token=nope").status_code == 403


async def test_missing_token_rejected(client):
    assert client.get("/kb/products?page_id=2").status_code == 422


async def test_no_token_configured_refuses_service(client, monkeypatch):
    monkeypatch.setenv("KB_ADMIN_TOKEN", "")
    from app.config import get_settings

    get_settings.cache_clear()
    # Fails CLOSED: an unset token must not mean "let everyone in".
    assert client.get(q("/kb/products", page_id="2")).status_code == 503


# ── products ──────────────────────────────────────────────────────────────────

async def test_create_and_list_product(client):
    client.post(
        q("/kb/products"),
        json={"page_id": "2", "name": "Veste Rouge", "price": "3500 DA", "stock": 4},
    )
    items = client.get(q("/kb/products", page_id="2")).json()
    assert len(items) == 1
    assert items[0]["name"] == "Veste Rouge"
    assert items[0]["stock"] == 4


async def test_resaving_same_name_updates_not_duplicates(client):
    for stock in (4, 1):
        client.post(
            q("/kb/products"),
            json={"page_id": "2", "name": "Veste Rouge", "price": "3500 DA", "stock": stock},
        )
    items = client.get(q("/kb/products", page_id="2")).json()
    assert len(items) == 1, "re-entering a product must update it, not duplicate it"
    assert items[0]["stock"] == 1


async def test_products_are_tenant_scoped(client):
    """The bug this prevents: store A's catalog answering store B's customer."""
    client.post(q("/kb/products"), json={"page_id": "2", "name": "A jacket", "price": "1 DA"})
    client.post(q("/kb/products"), json={"page_id": "9", "name": "B sneaker", "price": "2 DA"})
    assert [p["name"] for p in client.get(q("/kb/products", page_id="2")).json()] == ["A jacket"]
    assert [p["name"] for p in client.get(q("/kb/products", page_id="9")).json()] == ["B sneaker"]


async def test_delete_product(client):
    client.post(q("/kb/products"), json={"page_id": "2", "name": "Gone", "price": "1 DA"})
    pid = client.get(q("/kb/products", page_id="2")).json()[0]["id"]
    client.delete(q(f"/kb/products/{pid}"))
    assert client.get(q("/kb/products", page_id="2")).json() == []


async def test_negative_stock_rejected(client):
    r = client.post(
        q("/kb/products"), json={"page_id": "2", "name": "X", "price": "1 DA", "stock": -5}
    )
    assert r.status_code == 422


# ── policies ──────────────────────────────────────────────────────────────────

async def test_create_and_list_policy(client):
    client.post(
        q("/kb/policies"),
        json={"page_id": "2", "title": "Returns", "body": "14 days, unworn."},
    )
    items = client.get(q("/kb/policies", page_id="2")).json()
    assert items[0]["title"] == "Returns"
    assert items[0]["published"] is True


async def test_policies_are_tenant_scoped(client):
    client.post(q("/kb/policies"), json={"page_id": "2", "title": "A", "body": "free shipping"})
    client.post(q("/kb/policies"), json={"page_id": "9", "title": "B", "body": "no free shipping"})
    assert len(client.get(q("/kb/policies", page_id="2")).json()) == 1


async def test_empty_body_rejected(client):
    r = client.post(q("/kb/policies"), json={"page_id": "2", "title": "T", "body": ""})
    assert r.status_code == 422


# ── the panel ─────────────────────────────────────────────────────────────────

async def test_panel_requires_token(client):
    assert client.get("/kb/panel?token=nope").status_code == 403


async def test_panel_html_speaks_chatwoot(client):
    html = client.get(q("/kb/panel")).text
    # The handshake Chatwoot documents: the iframe must ASK, because the
    # appContext event can fire before the listener is registered.
    assert "chatwoot-dashboard-app:fetch-info" in html
    assert "appContext" in html
    # Dark mode: a white panel inside Chatwoot's dark theme is the giveaway.
    assert "prefers-color-scheme: dark" in html


async def test_delete_policy(client):
    client.post(q("/kb/policies"), json={"page_id": "2", "title": "Gone", "body": "text"})
    cid = client.get(q("/kb/policies", page_id="2")).json()[0]["id"]
    client.delete(q(f"/kb/policies/{cid}"))
    assert client.get(q("/kb/policies", page_id="2")).json() == []


async def test_malformed_id_is_400_not_500(client):
    """A Uuid column given a raw string raises mid-statement. 400, not 500."""
    assert client.delete(q("/kb/products/not-a-uuid")).status_code == 400


# ── the stock view: search + edit ─────────────────────────────────────────────

async def test_search_filters_by_name(client):
    for name in ("Veste Rouge", "Botte Cuir", "Sac Toile"):
        client.post(q("/kb/products"), json={"page_id": "2", "name": name, "price": "1 DA"})
    found = client.get(q("/kb/products", page_id="2", q="cuir")).json()
    assert [p["name"] for p in found] == ["Botte Cuir"]


async def test_search_is_case_insensitive(client):
    client.post(q("/kb/products"), json={"page_id": "2", "name": "Veste Rouge", "price": "1 DA"})
    assert len(client.get(q("/kb/products", page_id="2", q="VESTE")).json()) == 1


async def test_search_also_matches_description(client):
    """Owners search by attributes they typed, not just the model name."""
    client.post(q("/kb/products"), json={
        "page_id": "2", "name": "Botte Cuir", "price": "1 DA",
        "description": "Full-grain leather, hand-finished"})
    assert len(client.get(q("/kb/products", page_id="2", q="leather")).json()) == 1


async def test_search_stays_inside_the_tenant(client):
    client.post(q("/kb/products"), json={"page_id": "2", "name": "Veste A", "price": "1 DA"})
    client.post(q("/kb/products"), json={"page_id": "9", "name": "Veste B", "price": "1 DA"})
    found = client.get(q("/kb/products", page_id="2", q="Veste")).json()
    assert [p["name"] for p in found] == ["Veste A"]


async def test_edit_renames_instead_of_duplicating(client):
    """The bug PUT exists to prevent: POST upserts on (tenant, name), so a
    rename through POST leaves the OLD product for the bot to quote."""
    client.post(q("/kb/products"), json={"page_id": "2", "name": "Old name", "price": "1 DA"})
    pid = client.get(q("/kb/products", page_id="2")).json()[0]["id"]
    client.put(q(f"/kb/products/{pid}"),
               json={"page_id": "2", "name": "New name", "price": "2 DA", "stock": 7})
    items = client.get(q("/kb/products", page_id="2")).json()
    assert len(items) == 1, "rename must not leave a duplicate behind"
    assert items[0]["name"] == "New name"
    assert items[0]["stock"] == 7


async def test_edit_cannot_move_a_product_between_tenants(client):
    client.post(q("/kb/products"), json={"page_id": "2", "name": "Mine", "price": "1 DA"})
    pid = client.get(q("/kb/products", page_id="2")).json()[0]["id"]
    r = client.put(q(f"/kb/products/{pid}"),
                   json={"page_id": "9", "name": "Stolen", "price": "1 DA"})
    assert r.status_code == 403


async def test_edit_missing_product_is_404(client):
    import uuid as _uuid
    r = client.put(q(f"/kb/products/{_uuid.uuid4()}"),
                   json={"page_id": "2", "name": "Ghost", "price": "1 DA"})
    assert r.status_code == 404


async def test_stock_is_returned_for_the_overview(client):
    for name, stock in (("A", 0), ("B", 2), ("C", 40)):
        client.post(q("/kb/products"),
                    json={"page_id": "2", "name": name, "price": "1 DA", "stock": stock})
    items = client.get(q("/kb/products", page_id="2")).json()
    assert {p["name"]: p["stock"] for p in items} == {"A": 0, "B": 2, "C": 40}
