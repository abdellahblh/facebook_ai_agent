"""Agent Security Middleware & NeMo Guardrails Integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from typing import Any, Optional
from app.config import get_settings
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

# ── LangChain prebuilt PIIMiddleware ─────────────────────────────────────────
try:
    from langchain.agents.middleware import PIIMiddleware
    # Redact credit cards from user input before the LLM sees them
    pii_input_middleware = PIIMiddleware("credit_card", strategy="mask", apply_to_input=True)
    # Mask any credit cards that accidentally appear in the model's reply
    pii_output_middleware = PIIMiddleware("credit_card", strategy="mask", apply_to_output=True)
    HAS_PII_MIDDLEWARE = True
except ImportError:
    # Fallback: lightweight regex masker if the package version is older
    import re

    class _FallbackPII:
        def mask(self, text: str) -> str:
            return re.sub(r"\b(?:\d[ -]*?){13,16}\b", "<CREDIT_CARD_MASKED>", text)

    pii_input_middleware = _FallbackPII()
    pii_output_middleware = _FallbackPII()
    HAS_PII_MIDDLEWARE = False


def mask_input_pii(text: str) -> str:
    """Mask PII in one user message when using this custom LangGraph graph.

    ``PIIMiddleware`` is designed for LangChain's agent lifecycle, rather
    than as a callable ``str -> str`` object.  This adapter runs its public
    ``before_model`` hook and returns the resulting message content.
    """
    if not HAS_PII_MIDDLEWARE:
        return pii_input_middleware.mask(text)

    update = pii_input_middleware.before_model(
        {"messages": [HumanMessage(content=text)]}, runtime=None
    )
    if not update:
        return text
    return str(update["messages"][-1].content)


def mask_output_pii(text: str) -> str:
    """Mask PII in one model reply using ``PIIMiddleware``'s output hook."""
    if not HAS_PII_MIDDLEWARE:
        return pii_output_middleware.mask(text)

    update = pii_output_middleware.after_model(
        {"messages": [AIMessage(content=text)]}, runtime=None
    )
    if not update:
        return text
    return str(update["messages"][-1].content)

# ── NeMo Guardrails Safe Imports ─────────────────────────────────────────────
try:
    from nemoguardrails import LLMRails, RailsConfig
    from nemoguardrails.rails.llm.options import RailStatus, RailType
    HAS_NEMO_GUARDRAILS = True
except ImportError:
    LLMRails = None
    RailsConfig = None
    RailStatus = None
    RailType = None
    HAS_NEMO_GUARDRAILS = False


class NeMoGuardrailsWrapper:
    """Wrapper around NeMo Guardrails (LLMRails)."""

    def __init__(self, config_dir: Optional[str] = None):
        self.rails: Optional[Any] = None
        self.config_dir = config_dir or str(
            Path(__file__).resolve().parent
            / "nemo_guardrails_product_agent"
            / "nemo_guardrails_product_agent"
        )
        self._initialized = False

    def initialize(self) -> None:
        """Load the rail configuration once during the FastAPI lifespan."""
        if self._initialized:
            return
        self._initialized = True

        if not HAS_NEMO_GUARDRAILS:
            logger.info("nemoguardrails package not installed. Skipping NeMo Rails initialization.")
            return

        if os.path.exists(self.config_dir):
            try:
                # NeMo Guardrails OpenAI LLM engine expects OPENAI_API_KEY in environment variables
                settings = get_settings()
                OPENAI_API_KEY = get_settings.OPENAI_API_KEY
                if OPENAI_API_KEY:
                    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                config = RailsConfig.from_path(self.config_dir)
                self.rails = LLMRails(config)
                logger.info("Successfully loaded NeMo Guardrails from %s", self.config_dir)
            except Exception as exc:
                logger.warning("Failed to initialize NeMo Guardrails: %s", exc)
        else:
            logger.warning("NeMo Guardrails directory not found at: %s", self.config_dir)

    async def check_input(self, user_text: str) -> bool:
        """Runs NeMo INPUT rails — called before the graph sees the message."""
        if not self.rails:
            return True
        try:            
            result = await self.rails.check_async(
                [{"role": "user", "content": user_text}],
                rail_types=[RailType.INPUT],
            )
            if result.status == RailStatus.BLOCKED:
                logger.warning("NeMo input rail BLOCKED message.")
                return False
        except Exception as err:
            logger.error("Error during NeMo input rail check: %s", err)
        return True

    async def check_output(self, assistant_text: str, user_text: str = "") -> bool:
        """Runs NeMo OUTPUT rails — called after the graph produces a reply."""
        if not self.rails:
            return True
        try:
            messages = []
            if user_text:
                messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

            result = await self.rails.check_async(
                messages,
                rail_types=[RailType.OUTPUT],
            )
            if result.status == RailStatus.BLOCKED:
                logger.warning("NeMo output rail BLOCKED reply.")
                return False
        except Exception as err:
            logger.error("Error during NeMo output rail check: %s", err)
        return True
    def close(self):
        if self.rails:
            if hasattr(self.rails, "close") and callable(getattr(self.rails, "close")):
                try:
                    self.rails.close()
                except Exception as exc:
                    logger.warning("Error closing NeMo Guardrails: %s", exc)
            self.rails = None
            logger.info("Closed NeMo Guardrails.")
    


# The singleton is intentionally *constructed* at import time but initialized
# in FastAPI's lifespan. This prevents expensive configuration/model work from
# running in every worker import and makes its lifecycle observable on app.state.
nemo_guardrails = NeMoGuardrailsWrapper()
