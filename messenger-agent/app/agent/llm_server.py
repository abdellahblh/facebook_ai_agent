"""Shared LLM gateway for production and evaluation calls."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class _CallBudget:
    def __init__(self, max_calls: int, cooldown_seconds: float) -> None:
        self.max_calls = max_calls
        self.cooldown_seconds = cooldown_seconds
        self.calls = 0
        self.lock = asyncio.Lock()


class LLMServer:
    """Rate-limit an LLM runnable and provide a non-crashing fallback reply."""

    def __init__(
        self,
        primary: Any,
        *,
        fallback: Any | None = None,
        fallback_text: str = "I'm sorry, I couldn't process your request right now.",
        max_calls: int = 15,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.primary = primary
        self.fallback = fallback
        self.fallback_text = fallback_text
        self._budget = _CallBudget(max_calls, cooldown_seconds)

    def bind_tools(self, *args: Any, **kwargs: Any) -> "LLMServer":
        bound = type(self)(
            self.primary.bind_tools(*args, **kwargs),
            fallback=self.fallback.bind_tools(*args, **kwargs) if self.fallback else None,
            fallback_text=self.fallback_text,
        )
        bound._budget = self._budget
        return bound

    async def _before_call(self) -> None:
        async with self._budget.lock:
            if self._budget.calls >= self._budget.max_calls:
                logger.info(
                    "LLM call limit reached (%s); cooling down for %.1fs.",
                    self._budget.max_calls,
                    self._budget.cooldown_seconds,
                )
                self._budget.calls = 0
                await asyncio.sleep(self._budget.cooldown_seconds)
            self._budget.calls += 1

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        await self._before_call()
        try:
            return await self.primary.ainvoke(*args, **kwargs)
        except Exception:
            logger.exception("Primary LLM call failed; trying fallback LLM.")
            if self.fallback is not None:
                try:
                    await self._before_call()
                    return await self.fallback.ainvoke(*args, **kwargs)
                except Exception:
                    logger.exception("Fallback LLM call failed; returning fallback response.")
            return AIMessage(content=self.fallback_text)


__all__ = ["LLMServer"]
