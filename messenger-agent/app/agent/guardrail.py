"""Embedding-based prompt-injection guardrail backed by an in-memory FAISS index."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict,Protocol
import httpx
from pydantic import BaseModel, Field
import faiss
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage

from openai import AsyncOpenAI
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.query import Query
from app.config import get_settings
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
endpoint = get_settings().embedder_endpoint 
class EmbeddingProvider(Protocol):
    async def embed_query(self, text: str) -> Sequence[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[Sequence[float]]: ...


def _normalise(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype="float32").reshape(1, -1)
    if array.size == 0:
        raise ValueError("Embedding vector cannot be empty")
    faiss.normalize_L2(array)
    return array


def _pack_vector(vector: Sequence[float]) -> bytes:
    import struct

    return struct.pack(f"{len(vector)}f", *vector)


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    embedding: tuple[float, ...]
    blocked: bool
    similarity: float | None = None
    example_id: str | None = None


class EmbedderProvider:
    """Async adapter for the embedding provider API using AsyncOpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int = 1024,
        embedder_endpoint: str = endpoint,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.url = embedder_endpoint
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.url,
        )

    async def embed_query(self, text: str) -> Sequence[float]:
        vectors = await self._embed([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
        return await self._embed(texts)

    async def _embed(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("EMBEDDER_API_KEY must be set before creating embeddings")
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimensions,
            )
            batch_vectors = [item.embedding for item in response.data]
            all_vectors.extend(batch_vectors)

        if len(all_vectors) != len(texts):
            raise RuntimeError(f"Embedder returned {len(all_vectors)} embeddings for {len(texts)} inputs")
        if any(len(vector) != self.dimensions for vector in all_vectors):
            raise RuntimeError(f"Embedder embeddings must have {self.dimensions} dimensions")
        return all_vectors


class RedisTopicControlGuard:
    """Topic-control guard that searches allowed support topics in Redis.

    Blocks queries if the closest in-topic match does not reach the similarity threshold.
    """

    def __init__(
        self,
        redis_client: AsyncRedis,
        index_name: str,
        threshold: float = 0.50,
        embedder: EmbeddingProvider = None,
    ) -> None:
        self.redis_client = redis_client
        self.index_name = index_name
        self.threshold = threshold
        self.embedder = embedder

    async def inspect(self, text: str) -> GuardrailDecision:
        embedding = tuple(float(value) for value in await self.embedder.embed_query(text))
        query = (
            Query("(*)=>[KNN 1 @embedding $query_vector AS distance]")
            .sort_by("distance")
            .return_fields("id", "distance")
            .paging(0, 1)
            .dialect(2)
        )
        result = await self.redis_client.ft(self.index_name).search(
            query,
            query_params={"query_vector": _pack_vector(embedding)},
        )
        if not result.docs:
            return GuardrailDecision(embedding=embedding, blocked=True, similarity=0.0)

        match = result.docs[0]
        similarity = 1.0 - float(match.distance)
        return GuardrailDecision(
            embedding=embedding,
            blocked=similarity < self.threshold,
            similarity=similarity,
            example_id=getattr(match, "id", None),
        )


class FaissTopicControlGuard:
    """In-memory FAISS guard that checks inbound messages against on-topic support examples.

    Blocks messages if the similarity score is below the threshold (default 0.50).
    """

    def __init__(
        self,
        examples: list[dict],
        vectors: list[Sequence[float]],
        threshold: float = 0.50,
        embedder: EmbeddingProvider = None,
    ) -> None:
        if not examples or len(examples) != len(vectors):
            raise ValueError("Topic-control examples and vectors must be non-empty and aligned")
        matrix = np.vstack([_normalise(vector) for vector in vectors]).astype("float32")
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.examples = examples
        self.threshold = threshold
        self.embedder = embedder

    @classmethod
    async def from_jsonl(
        cls, path: Path, embedder: EmbeddingProvider, threshold: float = 0.50
    ) -> "FaissTopicControlGuard":
        examples: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("text"):
                examples.append(record)
        if not examples:
            raise ValueError(f"No topic-control records found in {path}")
        vectors = await embedder.embed_documents([record["text"] for record in examples])
        return cls(examples, vectors, threshold, embedder)

    async def inspect(self, text: str) -> GuardrailDecision:
        # This is the only query-embedding call. Its result is forwarded to
        # FaissSemanticCache by the worker instead of embedding the text again.
        embedding = tuple(float(value) for value in await self.embedder.embed_query(text))
        return self.inspect_embedding(embedding)

    def inspect_embedding(self, embedding: Sequence[float]) -> GuardrailDecision:
        vector = _normalise(embedding)
        scores, ids = self.index.search(vector, 1)
        score, position = float(scores[0][0]), int(ids[0][0])
        example_id = self.examples[position].get("id") if position >= 0 else None
        return GuardrailDecision(
            embedding=tuple(float(value) for value in embedding),
            blocked=position < 0 or score < self.threshold,
            similarity=score if position >= 0 else None,
            example_id=example_id,
        )

class GuardDecision(BaseModel):
    violation: int = Field(default=0, description="1 if input violates policy, 0 if safe")
    category: Optional[str] = None
    rationale: Optional[str] = None
    rating: float = Field(default=None, ge=0.0, le=1.0)

    @property
    def is_safe(self) -> bool:
        if self.violation == 1:
            return False
        if self.rating is not None and self.rating > 0.5:
            return False
        return True


class LLM_GUARD_OPENAI_API:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str = "",
        temp: Optional[float] = 0.1,
        max_tokens: Optional[int] = 250,
    ) -> None:
        self.api_key = api_key
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temp = temp if temp is not None else 0.1
        self.max_tokens = max_tokens if max_tokens is not None else 250
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def check(self, user_message: str) -> GuardDecision:
        user_content = (
            f"Content to classify: {user_message}\nAnswer (JSON only):"
        )
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temp,
            response_format=GuardDecision,
            max_tokens=self.max_tokens,
        )
        results = response.choices[0].message.parse
        try:
            return results
        except Exception:
            logger.warning("Failed to parse guardmodel JSON response: %s", raw_text)
            # If "violation": 1 appears anywhere in raw_text, mark as violation
            if '"violation": 1' in raw_text or '"violation":1' in raw_text:
                return GuardDecision(violation=1, category="Unsafe", rationale=raw_text)
            return GuardDecision(violation=0, category=None, rationale=raw_text)

# Backward-compatible aliases
RedisPromptInjectionGuard = RedisTopicControlGuard
FaissPromptInjectionGuard = FaissTopicControlGuard
LLMGuardModel = LLM_GUARD_OPENAI_API
