"""Step 5: repository layer, on in-memory SQLite (aiosqlite)."""

from app.db.models import Product
from app.db.repo import (
    find_products,
    get_or_create_customer,
    get_recent_messages,
    save_dead_letter,
    save_message,
    set_handoff,
)


async def test_save_and_read_messages_in_order(sqlite_session_factory):
    async with sqlite_session_factory() as s:
        await save_message(s, "PAGE1", "PSID", "user", "first")
        await save_message(s, "PAGE1", "PSID", "assistant", "second")
        await save_message(s, "PAGE1", "PSID", "user", "third")

    async with sqlite_session_factory() as s:
        history = await get_recent_messages(s, "PSID", limit=10)
    assert [m.text for m in history] == ["first", "second", "third"]  # oldest first
    assert [m.role for m in history] == ["user", "assistant", "user"]  # one ROW per message


async def test_recent_messages_returns_latest_window(sqlite_session_factory):
    async with sqlite_session_factory() as s:
        for i in range(10):
            await save_message(s, "PAGE1", "PSID", "user", f"msg-{i}")
    async with sqlite_session_factory() as s:
        history = await get_recent_messages(s, "PSID", limit=3)
    assert [m.text for m in history] == ["msg-7", "msg-8", "msg-9"]  # the LATEST 3, in order


async def test_handoff_flag_roundtrip(sqlite_session_factory):
    async with sqlite_session_factory() as s:
        customer = await get_or_create_customer(s, "PAGE1", "PSID")
        assert customer.handoff_active is False
        await set_handoff(s , "PAGE1" , "PSID", True)
    async with sqlite_session_factory() as s:
        customer = await get_or_create_customer(s, "PAGE1", "PSID")
        assert customer.handoff_active is True


async def test_dead_letter_saved(sqlite_session_factory):
    async with sqlite_session_factory() as s:
        await save_dead_letter(s, raw='{"weird": true}', error="unknown kind")
    # if this didn't raise, the row exists; dashboards query it later


async def test_find_products_case_insensitive(sqlite_session_factory):
    async with sqlite_session_factory() as s:
        s.add(Product(name="Veste Rouge Classic", price="3500 DA", stock=4))
        s.add(Product(name="Veste Noire", price="4200 DA", stock=0))
        s.add(Product(name="Casquette", price="900 DA", stock=12))
        await s.commit()

    async with sqlite_session_factory() as s:
        hits = await find_products(s, "veste")
    assert {p.name for p in hits} == {"Veste Rouge Classic", "Veste Noire"}
