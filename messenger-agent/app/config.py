"""Configuration — provided complete (boilerplate teaches nothing).

Everything comes from .env. No secrets in code, ever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    facebook_verify_token: str = os.getenv("FACEBOOK_VERIFY_TOKEN", "")
    facebook_app_secret: str = os.getenv("FACEBOOK_APP_SECRET", "")
    page_access_token: str = os.getenv("PAGE_ACCESS_TOKEN", "")

    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    debounce_seconds: int = int(os.getenv("DEBOUNCE_SECONDS", "10"))
    history_max_messages: int = int(os.getenv("HISTORY_MAX_MESSAGES", "30"))

    graph_api_version: str = "v23.0"  # check the changelog when upgrading

    @property
    def async_database_url(self) -> str:
        """SQLAlchemy async needs the +asyncpg driver marker in the URL."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
