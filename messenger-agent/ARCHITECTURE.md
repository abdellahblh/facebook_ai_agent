# Messenger AI Agent — System Architecture & Execution Flow

This document provides a detailed end-to-end explanation of the Messenger AI Agent codebase, covering every component, file, database model, caching strategy, and the step-by-step path a message takes from the moment Meta sends a webhook request to the final reply delivered to the user.

---

## 1. System Overview

The **Messenger AI Agent** is an enterprise-ready, asynchronous automated customer support system built with **FastAPI**, **LangGraph**, **Google Gemini**, **PostgreSQL**, **Redis**, and **Pinecone**.

### Key Architectural Highlights
* **Asynchronous Webhook Pipeline**: Validates, normalizes, deduplicates, debounces, and locks incoming messages before invoking the AI agent.
* **LangGraph Agentic Workflow**: Multi-turn stateful LLM loop with dynamic tool calling and persistent checkpointing in PostgreSQL (`AsyncPostgresSaver`).
* **Hybrid Knowledge Retrieval**:
  * **Exact Facts (SQL)**: Product prices and inventory queried directly from PostgreSQL to guarantee 100% accuracy without LLM hallucination.
  * **Fuzzy Knowledge (RAG)**: Policy, warranty, and FAQ lookups served via Pinecone vector embeddings (`llama-text-embed-v2`).
* **Human Handoff Guardrails**: Seamless escalation mechanism that disables AI responses when a human teammate takes over.
* **Robust Resilience**: Deduplication for Meta's duplicate webhooks, message burst debouncing, and per-user locking to eliminate race conditions.

---

## 2. Directory & File Reference Map

```
messenger-agent/
├── app/
│   ├── main.py             # FastAPI entrypoint, lifespan startup/shutdown, dependency wiring
│   ├── config.py           # Application settings loaded via pydantic-settings & .env
│   ├── security.py         # Meta HMAC-SHA256 signature verification
│   ├── schemas.py          # Pydantic data models for Meta payloads & normalized InboundMessage
│   ├── webhook.py          # Meta GET (verification) and POST (event delivery) endpoints
│   ├── worker.py           # Core pipeline connecting incoming events to database, agent, & Send API
│   ├── messenger_api.py    # Outbound Meta Graph API helper (typing indicators, message splitting, text sending)
│   ├── agent/
│   │   ├── graph.py        # LangGraph graph definition (StateGraph, nodes, memory checkpointer, run_turn)
│   │   ├── prompts.py      # System prompt for the Gemini AI customer support agent
│   │   ├── state.py        # LangGraph State model (AgentState containing message history)
│   │   └── tools.py        # Tool factory (product_lookup via SQL, policy_search via Pinecone, handoff_to_human)
│   ├── db/
│   │   ├── engine.py       # Async SQLAlchemy engine initialization and session management
│   │   ├── models.py       # SQLAlchemy ORM models (MessageLog, Customer, Product, DeadLetter)
│   │   └── repo.py         # Database access repository functions (CRUD operations for messages, customers, products)
│   └── cache/
│       └── redis_ops.py    # Redis operations (deduplication, burst buffering/debouncing, per-user distributed locks)
├── rag/
│   ├── politiques_boutique.md  # Raw store policy document (returns, delivery, warranty, payments)
│   └── rag.py                  # Script to split, embed, and ingest policy docs into Pinecone
├── .env                    # Environment variables (API keys, DB credentials, tokens)
├── pyproject.toml / requirements.txt  # Dependencies
└── docker-compose.yml      # Local development service stack (PostgreSQL, Redis)
```

---

## 3. End-to-End Execution Flow (Message Received → Response Sent)

Below is the step-by-step trace of what happens when a customer sends a message on Facebook Messenger.

```
[ Customer sends message ]
           │
           ▼
[ 1. Meta Webhook POST /webhook ] ──► [ Security Check: HMAC-SHA256 ] (app/security.py)
           │
           ▼
[ 2. Payload Parsing & Normalization ] (app/schemas.py & app/webhook.py)
           │
           ▼
[ 3. Worker Pipeline Ingestion ] (app/worker.py -> process_event)
           │
           ├──► Step 3.1: Deduplication (app/cache/redis_ops.py -> is_duplicate)
           ├──► Step 3.2: Human Handoff Check (app/db/repo.py -> get_or_create_customer)
           ├──► Step 3.3: Special Content & Handoff Postback Handling
           ├──► Step 3.4: Message Burst Debouncing (app/cache/redis_ops.py -> buffer_and_wait)
           └──► Step 3.5: Per-User Distributed Lock (app/cache/redis_ops.py -> acquire_user_lock)
           │
           ▼
[ 4. Log User Message & Show Typing Indicator ] (app/db/repo.py & app/messenger_api.py)
           │
           ▼
[ 5. LangGraph Agent Turn ] (app/agent/graph.py & app/agent/tools.py)
           │
           ├── Tool Node: product_lookup  ──► SQL Query (app/db/repo.py -> find_products)
           ├── Tool Node: policy_search   ──► Vector Search (Pinecone Async Index)
           └── Tool Node: handoff_to_human ──► Flag Update (app/db/repo.py -> set_handoff)
           │
           ▼
[ 6. Send Response via Meta Graph API ] (app/messenger_api.py -> send_text)
           │
           ▼
[ 7. Log Assistant Message & Release Redis Lock ] (app/db/repo.py & app/cache/redis_ops.py)
```

---

### Phase 1: Webhook Ingestion & Security Verification
* **Files involved**: [`app/main.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/main.py), [`app/webhook.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/webhook.py), [`app/security.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/security.py)

1. Meta sends an HTTP `POST /webhook` request containing a JSON body and an `X-Hub-Signature-256` header.
2. `webhook.receive()` reads raw body bytes and invokes `verify_signature()` in [`app/security.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/security.py).
3. `verify_signature()` computes an HMAC-SHA256 digest over the raw request bytes using `FACEBOOK_APP_SECRET` and compares it against Meta's signature with `hmac.compare_digest()`. If signature verification fails, HTTP `403 Forbidden` is returned immediately.

---

### Phase 2: Schema Parsing & Event Normalization
* **Files involved**: [`app/schemas.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/schemas.py), [`app/webhook.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/webhook.py)

1. `WebhookPayload.model_validate_json(raw)` parses the JSON envelope. Models extend `Tolerant` (which sets `extra="ignore"`) so unexpected Meta fields will not break the endpoint.
2. `webhook.normalize()` extracts events:
   * Echo messages (`is_echo = True`) and delivery/read receipts are ignored.
   * Relevant message/postback attributes are packed into a platform-agnostic `InboundMessage` dataclass (`page_id`, `psid`, `kind`, `text`, `attachment_urls`, `postback_payload`, `mid`).
3. `process_event(normalized)` is invoked in [`app/worker.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/worker.py).

---

### Phase 3: Middleware Pipeline & Safety Guards
* **Files involved**: [`app/worker.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/worker.py), [`app/cache/redis_ops.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/cache/redis_ops.py), [`app/db/repo.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/db/repo.py)

`process_event()` executes a 10-step sequence:

1. **Deduplication (`redis_ops.is_duplicate`)**: Checks Redis key `seen:{message_id}` using `SET NX EX 3600`. If key exists, Meta's retried request is skipped.
2. **Handoff Guard (`repo.get_or_create_customer`)**: Retrieves customer state from PostgreSQL. If `customer.handoff_active == True`, the incoming user message is saved to the database for human agent context, and the AI agent execution aborts.
3. **Special Actions / Postbacks**:
   * If `inbound.postback_payload == "HANDOFF"`, sets `handoff_active = True` in DB and notifies the user via Messenger API that a human agent will take over.
   * If user sends attachment without text, sends standard acknowledgement.
4. **Message Debouncing (`redis_ops.buffer_and_wait`)**:
   * If a user sends multiple rapid messages (e.g. "Hi", "I have a question", "about my order"), each message appends to `buf:{psid}` list in Redis and updates `last:{psid}` timestamp token.
   * Waits `debounce_seconds = 2`. The turn that owns the latest timestamp reads all buffered strings, deletes the buffer, and merges them into a single string (e.g. "Hi I have a question about my order"). Earlier turns yield `None` and exit.
5. **User Lock (`redis_ops.acquire_user_lock`)**: Sets `lock:{psid}` in Redis (`EX 120`) to prevent concurrent pipeline execution for the same customer.

---

### Phase 4: Database Logging & Typing Indicator
* **Files involved**: [`app/db/repo.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/db/repo.py), [`app/messenger_api.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/messenger_api.py)

1. `repo.save_message()` logs the merged user input into PostgreSQL `messages` table (`role="user"`).
2. `messenger_api.send_typing()` sends a POST request to Meta Graph API endpoint `/me/messages` with body `{"recipient": {"id": psid}, "sender_action": "typing_on"}`.

---

### Phase 5: LangGraph Execution & Agent Tools
* **Files involved**: [`app/agent/graph.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/agent/graph.py), [`app/agent/tools.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/agent/tools.py), [`app/agent/prompts.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/agent/prompts.py)

1. `make_tools(psid, page_id, session_factory)` initializes request-scoped tools:
   * **`product_lookup(query)`**: Calls `repo.find_products()` via case-insensitive SQL ILIKE search on `products` table. Returns exact name, price, and stock count.
   * **`policy_search(question)`**: Connects asynchronously to Pinecone index `facebook`, embeds query with `llama-text-embed-v2` (384 dimensions), searches `faq` namespace, and formats matching policy text chunks.
   * **`handoff_to_human(reason)`**: Calls `repo.set_handoff(..., True)` to set `handoff_active = True`.
2. `build_graph()` constructs a LangGraph state machine:
   * `START → agent → (tools_condition?) → tools → agent → ... → END`
   * Checkpointed using PostgreSQL `AsyncPostgresSaver` with `thread_id = psid` so conversation state persists across interactions.
3. `run_turn()` invokes the compiled graph with `HumanMessage(merged_text)` and returns the LLM's final generated string response.

---

### Phase 6: Response Delivery & Splitting
* **Files involved**: [`app/messenger_api.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/messenger_api.py)

1. `messenger_api.send_text()` accepts the generated response text.
2. Meta Messenger enforces a strict 2,000 character limit per message payload. `split_message()` splits long agent responses cleanly along line breaks or sentence boundaries (`[.!?]`).
3. Each chunk is POSTed to `https://graph.facebook.com/{v}/me/messages` using `httpx.AsyncClient` with the page access token.

---

### Phase 7: Logging & Cleanup
* **Files involved**: [`app/db/repo.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/db/repo.py), [`app/cache/redis_ops.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/cache/redis_ops.py)

1. `repo.save_message()` logs the AI response into PostgreSQL `messages` table (`role="assistant"`).
2. The `finally` block in `process_event()` calls `redis_ops.release_user_lock(redis, psid)` to remove key `lock:{psid}`, allowing subsequent messages from the customer to be processed.

---

## 4. Database Schema Details (`app/db/models.py`)

* **`messages` (`MessageLog`)**: Append-only conversation history. Includes `id`, `page_id`, `psid`, `role` (`user` | `assistant` | `system`), `text`, `meta` (JSON), and `created_at`. Indexed on `(psid, created_at)`.
* **`customers` (`Customer`)**: Customer tracking entity storing `psid`, `page_id`, `handoff_active` (boolean), `facts` (JSON for long-term memory), and `created_at`.
* **`products` (`Product`)**: E-commerce catalog table storing `id`, `name` (indexed), `price` (string), `stock` (integer), and `description`.
* **`dead_letters` (`DeadLetter`)**: Error store for unparseable raw request payloads to allow auditing and schema expansion.

---

## 5. Summary Table: File vs Function Matrix

| File Path | Main Classes / Functions | Primary Responsibility |
| :--- | :--- | :--- |
| [`app/main.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/main.py) | `lifespan()`, `FastAPI app` | Application startup, resource initialization (DB, Redis, LLM, Saver, HTTP) |
| [`app/webhook.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/webhook.py) | `verify()`, `receive()`, `normalize()` | Webhook verification endpoint, incoming webhook receiver, message normalizer |
| [`app/security.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/security.py) | `verify_signature()` | Validates Meta HMAC-SHA256 signature on raw request body |
| [`app/schemas.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/schemas.py) | `WebhookPayload`, `MessagingEvent`, `InboundMessage` | Tolerant Pydantic models for parsing Meta payloads and normalizing events |
| [`app/worker.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/worker.py) | `process_event()` | Core end-to-end pipeline orchestrator |
| [`app/cache/redis_ops.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/cache/redis_ops.py) | `is_duplicate()`, `buffer_and_wait()`, `acquire_user_lock()`, `release_user_lock()` | Deduplication, burst debouncing, and distributed locks via Redis |
| [`app/db/repo.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/db/repo.py) | `save_message()`, `get_or_create_customer()`, `set_handoff()`, `find_products()` | Database operations interface for PostgreSQL tables |
| [`app/agent/graph.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/agent/graph.py) | `build_graph()`, `run_turn()` | LangGraph state graph compilation and multi-turn turn runner |
| [`app/agent/tools.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/agent/tools.py) | `make_tools()`, `product_lookup`, `policy_search`, `handoff_to_human` | LangChain tools factory providing access to SQL catalog, Pinecone RAG, and handoff |
| [`app/messenger_api.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/app/messenger_api.py) | `send_typing()`, `send_text()`, `split_message()` | Outbound communication with Meta Graph API |
| [`rag/rag.py`](file:///c:/Users/PcTec/OneDrive/Documents/courses/cs50ai/messenger-agent-skeleton/messenger-agent/rag/rag.py) | `ingest_docs()` | Markdown splitting, vector embedding (`llama-text-embed-v2`), and Pinecone vector store upsert |
