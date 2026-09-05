"""Configuration — provided complete (boilerplate teaches nothing).

Everything comes from .env. No secrets in code, ever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    facebook_verify_token: str = field(default_factory=lambda: os.getenv("FACEBOOK_VERIFY_TOKEN", ""))
    facebook_app_secret: str = field(default_factory=lambda: os.getenv("FACEBOOK_APP_SECRET", ""))
    page_access_token: str = field(default_factory=lambda: os.getenv("PAGE_ACCESS_TOKEN", ""))

    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX", "facebook"))

    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    debounce_seconds: int = field(default_factory=lambda: int(os.getenv("DEBOUNCE_SECONDS", "10")))
    history_max_messages: int = field(default_factory=lambda: int(os.getenv("HISTORY_MAX_MESSAGES", "30")))
    business_name: str = field(default_factory=lambda: os.getenv("business_name", "dzbot"))
    chatwoot_base_url: str = field(
        default_factory=lambda: os.getenv("CHATWOOT_BASE_URL", "")
    )
    chatwoot_api_token: str = field(
        default_factory=lambda: (
            os.getenv("CHATWOOT_API_TOKEN")
            or os.getenv("CHATWOOT_API_KEY")
            or os.getenv("chatwoot_acces_token")
            or os.getenv("chatwoot_access_token")
            or ""
        )
    )
    # Long random string; it lives in the webhook URL path since Chatwoot
    # does not sign its webhooks.
    chatwoot_webhook_secret: str = field(
        default_factory=lambda: (
            os.getenv("CHATWOOT_WEBHOOK_SECRET")
            or os.getenv("chatwoot_webhook_secret")
            or ""
        )
    )
    # If true, only answer conversations marked `pending`. Default false so
    # the bot also handles newly created `open` unassigned conversations.
    chatwoot_bot_only_when_pending: bool = field(
        default_factory=lambda: os.getenv("CHATWOOT_BOT_ONLY_WHEN_PENDING", "false").lower()
        == "true"
    )

    # Who receives escalated conversations. ONE of these must be set or the
    # handoff does nothing. Find the ids in Chatwoot: Settings -> Teams, or
    # Settings -> Agents (the id is in the URL).
    # Fallback tenant when a payload omits account_id. Find it in your
    # Chatwoot dashboard URL: /app/accounts/<ID>/dashboard
    # Protects the Dashboard App panel + its API. Chatwoot cannot authenticate
    # the iframe, so this token is the only gate. Admin-only; rotate like a
    # password. python3 -c "import secrets; print(secrets.token_urlsafe(32))"
    kb_admin_token: str = field(
        default_factory=lambda: os.getenv("KB_ADMIN_TOKEN", "")
    )

    chatwoot_account_id: int = field(
        default_factory=lambda: int(os.getenv("CHATWOOT_ACCOUNT_ID", "0"))
    )

    chatwoot_handoff_team_id: int = field(
        default_factory=lambda: int(os.getenv("CHATWOOT_HANDOFF_TEAM_ID", "0"))
    )
    chatwoot_handoff_agent_id: int = field(
        default_factory=lambda: int(os.getenv("CHATWOOT_HANDOFF_AGENT_ID", "0"))
    )
    graph_api_version: str = "v23.0"  # check the changelog when upgrading
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    @property
    def async_database_url(self) -> str:
        """SQLAlchemy async needs the +asyncpg driver marker in the URL."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
