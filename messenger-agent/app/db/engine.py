"""Engine and session factory. YOU implement init_engine. (Part of step 5)

Lesson encoded here: your v1's database.py imported itself and defined
everything twice. One module = one job: THIS file owns the engine;
models.py owns the tables; repo.py owns the queries. Nothing circular.
"""

from __future__ import annotations
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from app.db.models import Base


# Module-level holders, populated once by init_engine() at startup.
engine: AsyncEngine | None = None
SessionFactory: async_sessionmaker[AsyncSession] | None = None


async def init_engine(async_database_url: str) -> None:
    """Create the engine + session factory and create tables.

    TODO(you) — steps:
      1. global engine, SessionFactory
      2. engine = create_async_engine(async_database_url, pool_size=5,
         max_overflow=10, echo=False)
      3. SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
         (async_sessionmaker is the SQLAlchemy 2.0 way — not
         sessionmaker(class_=AsyncSession))
      4. Create tables:
             from app.db.models import Base
             async with engine.begin() as conn:
                 await conn.run_sync(Base.metadata.create_all)
         (Fine for learning. Production upgrade later: Alembic migrations.)

    """
    engine = create_async_engine(
        async_database_url,
        echo=True,  # Set to True to see SQL
        pool_size=5,
        max_overflow=10
    )

    # Create async session factory
    AsyncSessionLocal = async_sessionmaker(
        engine,
        expire_on_commit=False
                )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def dispose_engine() -> None:
    """TODO(you): if engine is not None → await engine.dispose(). Called at shutdown."""
    if engine is not None:
        await engine.dispose()
