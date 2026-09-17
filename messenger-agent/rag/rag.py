"""Embed store policies and guardrail examples with Jina and store them in Redis Stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from redis import Redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

load_dotenv()

DOCUMENT_PATH = Path(__file__).with_name("politiques_boutique.md")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
INDEX_NAME = os.getenv("REDIS_VECTOR_INDEX", "store-policies-jina")
GUARDRAIL_DATASET_PATH = Path(
    os.getenv(
        "GUARDRAIL_DATASET_PATH",
        Path(__file__).parents[1]
        / "data"
        / "vectorstore_guardrails"
        / "new_costumer_support_exam.jsonl",
    )
)
GUARDRAIL_INDEX_NAME = os.getenv(
    "REDIS_GUARDRAIL_INDEX", "customer-support-guardrails"
)
EMBEDDER_API_URL = os.getenv("EMBEDDER_ENDPOINT") or os.getenv("VOYAGE_ENDPOINT", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL") or os.getenv("VOYAGE_EMBEDDING_MODEL", "qwen3.7-text-embedding")
VECTOR_DIMENSIONS = int(os.getenv("EMBEDDER_DIMENSIONS") or os.getenv("VOYAGE_EMBEDDING_DIMENSIONS", "1024"))
TENANT_ID = os.getenv("TENANT_ID", "")


async def create_embeddings(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """Create normalized embeddings for retrieval documents or queries in batches."""
    api_key = os.getenv("EMBEDDER_API_KEY") or os.getenv("VOYAGE_API_KEY", "")
    if not api_key:
        raise RuntimeError("EMBEDDER_API_KEY must be set before creating embeddings")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=EMBEDDER_API_URL,
    )
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(
            input=batch,
            model=EMBEDDER_MODEL,
            dimensions=VECTOR_DIMENSIONS,
        )
        batch_vectors = [item.embedding for item in response.data]
        all_vectors.extend(batch_vectors)

    if len(all_vectors) != len(texts):
        raise RuntimeError(
            f"Embedder returned {len(all_vectors)} embeddings for {len(texts)} inputs"
        )
    if any(len(vector) != VECTOR_DIMENSIONS for vector in all_vectors):
        raise RuntimeError(f"Embedder embeddings must have {VECTOR_DIMENSIONS} dimensions")
    return all_vectors


def create_index(redis_client: Redis, index_name: str, key_prefix: str) -> None:
    """Create the Redis vector index once, ignoring the already-exists error."""
    schema_GUARDRAILS = (
        TextField("$.text", as_name="text"),
        TagField("$.id", as_name="id"),
        TagField("$.lang", as_name="lang"),
        TagField("$.class", as_name="class"),
        TagField("$.topic", as_name="topic"),

        VectorField(
            "$.embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSIONS,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="embedding",
        )
    )
    schema_policy = (
        TagField("$.source", as_name="source"),
        TagField("$.content_type", as_name="content_type"),
        TagField("$.tenant", as_name="tenant"),
        TextField("$.section_title", as_name="section_title"),
        TextField("$.text", as_name="text"),
        VectorField(
            "$.embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSIONS,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="embedding",
        ),
    )
    try:
        redis_client.ft(index_name).create_index(
            schema_GUARDRAILS if index_name == GUARDRAIL_INDEX_NAME else schema_policy,
            definition=IndexDefinition(prefix=[key_prefix], index_type=IndexType.JSON),
        )
    except Exception as error:
        if "Index already exists" not in str(error):
            raise


async def ingest_docs(tenant_id: str = "") -> int:
    """Split, embed, and store the policy document in Redis."""
    markdown = DOCUMENT_PATH.read_text(encoding="utf-8")
    header_splits = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "document_title"), ("##", "section_title")],
        strip_headers=False,
    ).split_text(markdown)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    ).split_documents(header_splits)

    for chunk in chunks:
        chunk.metadata.update({"source": DOCUMENT_PATH.name, "content_type": "store_policy"})

    redis_client = Redis.from_url(REDIS_URL, decode_responses=False, socket_timeout=60.0)
    redis_client.ping()
    key_prefix = f"{INDEX_NAME}:"
    create_index(redis_client, INDEX_NAME, key_prefix)
    vectors = await create_embeddings(
        [chunk.page_content for chunk in chunks])

    with redis_client.pipeline() as pipeline:
        for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            key = f"{key_prefix}{position}"
            document = {
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "content_type": chunk.metadata["content_type"],
                "section_title": chunk.metadata.get("section_title", "General"),
                "tenant": tenant_id,
                "embedding": vector,
            }
            pipeline.json().set(key, "$", document)
        pipeline.execute()
    return len(chunks)


async def ingest_guardrail_examples() -> int:
    """Embed guardrail examples and store every JSON key as Redis metadata."""
    records = [
        json.loads(line)
        for line in GUARDRAIL_DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No guardrail records found in {GUARDRAIL_DATASET_PATH}")
    if any(not record.get("text") for record in records):
        raise ValueError("Every guardrail record must contain a non-empty 'text' field")

    vectors = await create_embeddings(
        [record["text"] for record in records])
    redis_client = Redis.from_url(REDIS_URL, decode_responses=False, socket_timeout=60.0)
    redis_client.ping()
    key_prefix = f"{GUARDRAIL_INDEX_NAME}:"
    create_index(redis_client, GUARDRAIL_INDEX_NAME, key_prefix)

    with redis_client.pipeline() as pipeline:
        for position, record in enumerate(records):
            record_id = str(record.get("id", len(record)))
            metadata = {key: value for key, value in record.items() if key != "text"}
            document = {
                "text": record["text"],
                **metadata,
                "source": GUARDRAIL_DATASET_PATH.name,
                "content_type": "guardrail_example",
                "embedding": vectors[position],
            }
            pipeline.json().set(f"{key_prefix}{record_id}", "$", document)
            if (position + 1) % 50 == 0:
                pipeline.execute()
        pipeline.execute()
    return len(records)


async def search_policies(question: str, limit: int = 3, tenant: str = "") -> list[dict]:
    """Search stored policy vectors using query embedding, scoped to a tenant."""
    query_vectors = await create_embeddings([question])
    query_vector = query_vectors[0]
    redis_client = Redis.from_url(REDIS_URL, decode_responses=False, socket_timeout=60.0)
    tenant_filter = f"(@tenant:{{{tenant or 'default'}}})" if tenant else ""
    query = (
        Query(f"({tenant_filter})=>[KNN $limit @embedding $query_vector AS distance]")
        .sort_by("distance")
        .return_fields("text", "source", "section_title", "distance")
        .paging(0, limit)
        .dialect(2)
    )
    result = redis_client.ft(INDEX_NAME).search(
        query,
        query_params={"query_vector": _pack_vector(query_vector), "limit": limit},
    )
    return [dict(document.__dict__) for document in result.docs]


def _pack_vector(vector: list[float]) -> bytes:
    """Convert Gemini's float vector into Redis' FLOAT32 binary format."""
    import struct

    return struct.pack(f"{len(vector)}f", *vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guardrails",
        action="store_true",
        help="ingest customer-support guardrail examples instead of store policies",
    )
    args = parser.parse_args()
    if args.guardrails:
        count = asyncio.run(ingest_guardrail_examples())
        print(f"Stored {count} guardrail examples in Redis index '{GUARDRAIL_INDEX_NAME}'.")
    else:
        count = asyncio.run(ingest_docs(TENANT_ID))
        print(f"Stored {count} policy chunks in Redis index '{INDEX_NAME}'.")
