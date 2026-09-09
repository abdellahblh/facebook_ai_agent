"""Agent Security Middleware & NeMo Guardrails Integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Dict

from app.config import get_settings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig  

logger = logging.getLogger(__name__)

# ── LangChain prebuilt PIIMiddleware ─────────────────────────────────────────
try:
    from langchain.agents.middleware import PIIMiddleware
    pii_input_middleware = PIIMiddleware("credit_card", strategy="mask", apply_to_input=True)
    pii_output_middleware = PIIMiddleware("credit_card", strategy="mask", apply_to_output=True)
    HAS_PII_MIDDLEWARE = True
except ImportError:
    import re

    class _FallbackPII:
        def mask(self, text: str) -> str:
            return re.sub(r"\b(?:\d[ -]*?){13,16}\b", "<CREDIT_CARD_MASKED>", text)

    pii_input_middleware = _FallbackPII()
    pii_output_middleware = _FallbackPII()
    HAS_PII_MIDDLEWARE = False


def mask_input_pii(text: str) -> str:
    if not HAS_PII_MIDDLEWARE:
        return pii_input_middleware.mask(text)

    update = pii_input_middleware.before_model(
        {"messages": [HumanMessage(content=text)]}, runtime=None
    )
    if not update:
        return text
    return str(update["messages"][-1].content)


def mask_output_pii(text: str) -> str:
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
    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails  
    HAS_NEMO_GUARDRAILS = True
except ImportError:
    LLMRails = None
    RailsConfig = None
    RailStatus = None
    RailType = None
    RunnableRails = None
    HAS_NEMO_GUARDRAILS = False


class NeMoGuardrailsWrapper:
    """Wrapper around NeMo Guardrails (LLMRails & RunnableRails)."""

    def __init__(self, config_dir: Optional[str] = None):
        self.rails: Optional[Any] = None
        self.runnable_rails: Optional[Any] = None  
        self.config_dir = config_dir or str(
            Path(__file__).resolve().parent
            / "guardrails"
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
                settings = get_settings()

                config = RailsConfig.from_path(self.config_dir)
                self.rails = LLMRails(config)
                # Create RunnableRails to support LangChain/Langfuse callbacks
                self.runnable_rails = RunnableRails(config=config)
                
                logger.info("Successfully loaded NeMo Guardrails from %s", self.config_dir)
            except Exception as exc:
                logger.warning("Failed to initialize NeMo Guardrails: %s", exc)
        else:
            logger.warning("NeMo Guardrails directory not found at: %s", self.config_dir)

    async def check_input(self, user_text: str, config: Optional[RunnableConfig] = None) -> bool:
        """Runs NeMo INPUT rails with Langfuse trace propagation via config."""
        if not self.runnable_rails and not self.rails:
            return True
        try:
            # If config (callbacks) is present, use RunnableRails to propagate trace
            if self.runnable_rails and config:
                res = await self.runnable_rails.ainvoke(
                    {"input": user_text},
                    config=config  # <-- Pass RunnableConfig containing Langfuse CallbackHandler
                )
                # Check if response was modified or blocked by rails
                if isinstance(res, dict) and res.get("output") == "I'm sorry, I can't assist with that.":
                    return False
                return True
            
            # Fallback to direct check_async if no config is supplied
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

    async def check_output(self, assistant_text: str, user_text: str = "", config: Optional[RunnableConfig] = None) -> bool:
        """Runs NeMo OUTPUT rails with Langfuse trace propagation via config."""
        if not self.runnable_rails and not self.rails:
            return True
        try:
            if self.runnable_rails and config:
                res = await self.runnable_rails.ainvoke(
                    {"input": user_text, "output": assistant_text},
                    config=config  # <-- Pass RunnableConfig containing Langfuse CallbackHandler
                )
                return True

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
            self.runnable_rails = None
            logger.info("Closed NeMo Guardrails.")


nemo_guardrails = NeMoGuardrailsWrapper()