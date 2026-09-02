"""Agent Security Middleware & NeMo Guardrails Integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

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

# Try importing NeMo Guardrails if installed
try:
    from nemoguardrails import LLMRails, RailsConfig
    HAS_NEMO_GUARDRAILS = True
except ImportError:
    LLMRails = None
    RailsConfig = None
    HAS_NEMO_GUARDRAILS = False




class NeMoGuardrailsWrapper:
    """Wrapper around NeMo Guardrails (LLMRails)."""

    def __init__(self, config_dir: Optional[str] = None):
        self.rails: Optional[Any] = None
        if not HAS_NEMO_GUARDRAILS:
            logger.info("nemoguardrails package not installed. Skipping NeMo Rails initialization.")
            return

        if not config_dir:
            base_path = Path(__file__).parent / "nemo_guardrails_product_agent" / "nemo_guardrails_product_agent"
            config_dir = str(base_path)

        if os.path.exists(config_dir):
            try:
                config = RailsConfig.from_path(config_dir)
                self.rails = LLMRails(config)
                logger.info("Successfully loaded NeMo Guardrails from %s", config_dir)
            except Exception as exc:
                logger.warning("Failed to initialize NeMo Guardrails: %s", exc)

    async def check_input(self, user_text: str) -> bool:
        """Runs NeMo INPUT rails — called before the graph sees the message."""
        if not self.rails:
            return True
        try:
            from nemoguardrails.rails.llm.options import RailStatus, RailType
            result = await self.rails.check_async(
                [{"role": "user", "content": user_text}],
                rail_types=[RailType.INPUT],
            )
            if result.status == RailStatus.BLOCKED:
                logger.warning("NeMo input rail BLOCKED message. Rail: %s", result.rail)
                return False
        except Exception as err:
            logger.error("Error during NeMo input rail check: %s", err)
        return True

    async def check_output(self, assistant_text: str, user_text: str = "") -> bool:
        """Runs NeMo OUTPUT rails — called after the graph produces a reply."""
        if not self.rails:
            return True
        try:
            from nemoguardrails.rails.llm.options import RailStatus, RailType
            messages = []
            if user_text:
                messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
            result = await self.rails.check_async(
                messages,
                rail_types=[RailType.OUTPUT],
            )
            if result.status == RailStatus.BLOCKED:
                logger.warning("NeMo output rail BLOCKED reply. Rail: %s", result.rail)
                return False
        except Exception as err:
            logger.error("Error during NeMo output rail check: %s", err)
        return True


# Global Singleton for NeMo Guardrails
nemo_guardrails = NeMoGuardrailsWrapper()