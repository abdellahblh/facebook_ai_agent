# Messenger AI Agent — Learning Skeleton

A production-shaped Facebook Messenger customer-support agent.
**Every file is structured and documented; the function bodies are yours to write.**

The tests are your definition of done: they encode every bug from the first
code review (hub.mode aliases, the envelope shape, always-200, echo filtering,
sticker ints, ...). Implement until green.

```
FastAPI webhook ──► Redis (dedupe / debounce / locks) ──► LangGraph agent ──► Send API
                                    │                          │
                                    └──────── PostgreSQL ◄─────┘
                                    (message log + agent memory via checkpointer)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d          # Postgres 16 + Redis 7
cp .env.example .env          # then fill the values

pytest                        # see where you stand (red = your TODO list)
```

Run the server:  `uvicorn app.main:app --reload --port 8000`
Expose locally:  `ngrok http 8000` → paste the URL in Meta's webhook config.

## The learning path — implement in THIS order

| # | File | What you learn | Done when |
|---|------|----------------|-----------|
| 1 | `app/security.py` | HMAC signatures, raw body vs parsed body | `pytest tests/test_security.py` green |
| 2 | `app/webhook.py` GET | Meta verification, query param aliases | `pytest tests/test_webhook.py -k verify` green |
| 3 | `app/webhook.py` POST | envelope parsing, always-200, echo filter | `pytest tests/test_webhook.py -k receive` green |
| 4 | `app/cache/redis_ops.py` | dedupe, debounce, per-user locks | `pytest tests/test_redis_ops.py` green |
| 5 | `app/db/repo.py` | async SQLAlchemy, one-row-per-message | `pytest tests/test_repo.py` green |
| 6 | `app/agent/tools.py` + `graph.py` | tool-calling agent + checkpointer memory | `pytest tests/test_agent.py` green |
| 7 | `app/messenger_api.py` | Send API, typing indicator, msg splitting | manual test via ngrok |
| 8 | `app/worker.py` | the full pipeline, in order | `pytest` all green |

`tests/test_schemas.py` passes from day one — the schemas are provided complete
because they encode the tolerance lessons (read them, they're commented).

## The five bugs you already made once (don't repeat them)

1. Meta sends `hub.mode` with a DOT — FastAPI params need `Query(alias="hub.mode")`.
2. Meta POSTs the ENVELOPE `{object, entry[].messaging[]}` — never a bare message.
3. Non-200 responses to Meta = retries, then your webhook gets disabled. Always 200 after the signature check.
4. `is_echo` messages are YOUR OWN bot's replies echoed back — process them and you build an infinite loop.
5. `sticker_id` arrives as an int. Pydantic v2 does not coerce int → str.

## Production gate

```bash
./check.sh    # ruff + mypy + pip-audit + pytest, all must pass before deploy
```
