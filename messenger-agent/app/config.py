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
    fallback_llm_model: str = field(
        default_factory=lambda: os.getenv("FALLBACK_LLM_MODEL", "llama-3.3-70b-versatile")
    )
    llm_max_calls: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_CALLS", "15")))
    llm_cooldown_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_COOLDOWN_SECONDS", "60"))
    )


    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_tracing: bool = field(default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower() == "true")
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "facebook_ai_agent"))
    embedder_endpoint: str = field(
        default_factory=lambda: os.getenv("EMBEDDER_ENDPOINT")
        or os.getenv("VOYAGE_ENDPOINT", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    )

    debounce_seconds: int = field(default_factory=lambda: int(os.getenv("DEBOUNCE_SECONDS", "10")))
    agent_response_cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RESPONSE_CACHE_TTL_SECONDS", "300"))
    )
    cache_l0_max_size: int = field(
        default_factory=lambda: int(os.getenv("CACHE_L0_MAX_SIZE", "512"))
    )
    cache_lock_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CACHE_LOCK_TTL_SECONDS", "120"))
    )
    cache_lock_wait_seconds: float = field(
        default_factory=lambda: float(os.getenv("CACHE_LOCK_WAIT_SECONDS", "15"))
    )
    embedder_api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDER_API_KEY")
        or os.getenv("VOYAGE_API_KEY", "")
    )
    embedder_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDER_MODEL")
        or os.getenv("VOYAGE_EMBEDDING_MODEL", "qwen3.7-text-embedding")
    )
    embedder_dimensions: int = field(
        default_factory=lambda: int(
            os.getenv("EMBEDDER_DIMENSIONS")
            or os.getenv("VOYAGE_EMBEDDING_DIMENSIONS", "1024")
        )
    )
    guardrail_index: str = field(
        default_factory=lambda: os.getenv("REDIS_GUARDRAIL_INDEX", "customer-support-guardrails")
    )
    faiss_semantic_threshold: float = field(
        default_factory=lambda: float(os.getenv("FAISS_SEMANTIC_THRESHOLD", "0.90"))
    )
    guardrail_threshold: float = field(
        default_factory=lambda: float(os.getenv("GUARDRAIL_THRESHOLD", "0.50"))
    )
    faiss_cache_max_size: int = field(
        default_factory=lambda: int(os.getenv("FAISS_CACHE_MAX_SIZE", "512"))
    )
    guardrail_dataset_path: str = field(
        default_factory=lambda: os.getenv("GUARDRAIL_DATASET_PATH", "")
    )
    # Deprecated Redis-vector configuration retained so existing deployments
    # can boot while moving to the process-local FAISS cache.
    cache_embedding_model: str = field(default_factory=lambda: os.getenv("CACHE_EMBEDDING_MODEL", ""))
    cache_embedding_dimensions: int = field(default_factory=lambda: int(os.getenv("CACHE_EMBEDDING_DIMENSIONS", "1024")))
    cache_semantic_threshold: float = field(
        default_factory=lambda: float(os.getenv("CACHE_SEMANTIC_THRESHOLD", "0.10"))
    )
    cache_semantic_index: str = field(
        default_factory=lambda: os.getenv("CACHE_SEMANTIC_INDEX", "cache-semantic-index")
    )
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
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    guard_model: str = field(
        default_factory=lambda: os.getenv("GUARD_MODEL", "openai/gpt-oss-safeguard-20b")
    )
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )

    # Webhook inbox queue (Redis Streams). The Chatwoot webhook enqueues, an
    # in-process consumer drains it and runs worker.process_event. Set
    # QUEUE_ENABLED=false as the ops escape hatch to revert to inline tasks.
    redis_stream: str = field(
        default_factory=lambda: os.getenv("REDIS_STREAM", "webhook:inbound")
    )
    redis_stream_group: str = field(
        default_factory=lambda: os.getenv("REDIS_STREAM_GROUP", "webhook-consumers")
    )
    queue_enabled: bool = field(
        default_factory=lambda: os.getenv("QUEUE_ENABLED", "true").lower() == "true"
    )
    queue_reclaim_min_idle_s: int = field(
        # How long an unacked entry must sit idle before XAUTOCLAIM presumes its
        # consumer died and reclaims it. MUST exceed worst-case turn duration so
        # a slow-but-alive consumer is never tripped. Default 180s.
        default_factory=lambda: int(os.getenv("QUEUE_RECLAIM_MIN_IDLE_S", "180"))
    )
    queue_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("QUEUE_MAX_ATTEMPTS", "3"))
    )
    queue_maxlen: int = field(
        default_factory=lambda: int(os.getenv("QUEUE_MAXLEN", "10000"))
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    langfuse_api_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_base_url: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    )
    @property
    def async_database_url(self) -> str:
        """SQLAlchemy async needs the +asyncpg driver marker in the URL."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
