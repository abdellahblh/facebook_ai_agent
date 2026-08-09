"""Database models — provided complete (declarations, not logic).

Design decisions these tables encode (from the architecture review):

1. ONE ROW PER MESSAGE with a `role` column — NOT paired (user_message,
   agent_response) rows. Reality your v1 shape couldn't store: a customer
   sending 3 messages before any reply (debounce!), the bot staying silent
   (handoff), attachments, system events.

2. This is the LOG — the owner dashboard, the audit trail, the
   "what did the bot say" record. It is NOT the agent's memory.
   Agent memory = the LangGraph checkpointer (its own tables, created by
   AsyncPostgresSaver.setup()). Two layers, two jobs.

3. dead_letters: payloads we couldn't parse. Your future corpus of real
   Meta payloads — you extend schemas.py FROM EVIDENCE, not from docs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Index, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    # datetime.utcnow() is deprecated AND naive. Always timezone-aware.
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class MessageLog(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    page_id: Mapped[str] = mapped_column(String(64), index=True)  # tenant key
    psid: Mapped[str] = mapped_column(String(64))  # the customer
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | system
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)  # attachments, tool calls, ...
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # The composite index serves "conversation history for one customer, in order"
    # — the single most frequent query this table will ever answer.
    __table_args__ = (Index("ix_messages_psid_created", "psid", "created_at"),)


class Customer(Base):
    __tablename__ = "customers"

    psid: Mapped[str] = mapped_column(String(64), primary_key=True)
    page_id: Mapped[str] = mapped_column(String(64), index=True)
    handoff_active: Mapped[bool] = mapped_column(Boolean, default=False)
    facts: Mapped[dict] = mapped_column(JSON, default=dict)  # long-term memory layer (phase 2)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DeadLetter(Base):
    __tablename__ = "dead_letters"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    raw: Mapped[str] = mapped_column(Text)  # the exact bytes we couldn't parse
    error: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Product(Base):
    """Mirror of the client's catalog. Exact facts live in SQL, never in
    embeddings — vector similarity quoting the wrong product's price is the
    single most expensive failure this bot can have."""

    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), index=True)
    price: Mapped[str] = mapped_column(String(64))  # string: currency formatting is presentation
    stock: Mapped[int] = mapped_column(default=0)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
