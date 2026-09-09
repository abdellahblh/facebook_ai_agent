"""Run the agent over the English dataset on Langfuse (v3 SDK) and score it.

MIRRORS run_turn() STEP FOR STEP — AND MUST KEEP DOING SO
    See evaluators.py and the original docstring for the full rationale.
    check_drift() compares run_turn's source against RUN_TURN_STEPS at
    startup and warns if production changed shape.

WHY dataset.run_experiment() AND NOT A HAND-ROLLED LOOP
    An earlier version of this script manually captured trace IDs and called
    langfuse.score()/item.link() itself. That capture happened BEFORE the
    @observe-wrapped target ran, so it read the trace ID of the wrong (empty)
    context — scores were likely never linked to the right trace at all.
    run_experiment() owns trace creation, linking, and scoring internally,
    which removes that entire failure mode.

BUDGET
    ~2.2 Gemini calls per turn, +1 per NeMo self-check rail per turn when
    the rails have an LLM. --dry-run prints the estimate and exits.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import pathlib
import sys
import time
import uuid
import warnings

import httpx
from google.api_core.exceptions import InternalServerError, ResourceExhausted, ServiceUnavailable
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from tests.evals.evaluators import ALL_EVALUATORS, TOOL_ALIASES
from app.config import get_settings
settings = get_settings()
langfuse = get_client()
langfuse_handler = CallbackHandler()

RETRYABLE_EXCEPTIONS = (
    ResourceExhausted,
    ServiceUnavailable,
    InternalServerError,
    OutputParserException,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
)
DATASET = "english-production-readiness-v1"
TURN_SEP = " || "
BLOCKED_REPLY = "Sma7lna, ma n9derch njawbek 3la had l'demande."
RUN_TURN_STEPS = (
    "pii_input_middleware", "check_input", "ainvoke", "check_output", "pii_output_middleware",
)
CALLS_PER_TURN = 2.2


class RateLimiter:
    """Evenly spaces calls to stay under `rpm` per rolling minute. Shared
    across EVERY Gemini-touching call — including the two guardrail checks,
    which is why call_with_retry now wraps those too, not just the graph."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self.min_interval
            wait = start - now
        if wait > 0:
            await asyncio.sleep(wait)


RATE_LIMITER = RateLimiter(rpm=15)


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=6, max=60),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=lambda rs: print(f"Rate limited, retrying in {rs.next_action.sleep:.1f}s..."),
)
async def call_with_retry(coro_factory):
    await RATE_LIMITER.acquire()
    return await coro_factory()


def check_drift(run_turn_fn) -> list[str]:
    try:
        src = inspect.getsource(run_turn_fn)
    except (OSError, TypeError):
        return []
    return [step for step in RUN_TURN_STEPS if step not in src]


def _reply_text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def make_target(graph, guardrails, pii_in, pii_out, *, page_id: str = "demo",
                 max_input_chars: int = 8_000, recursion_limit: int = 15):
    """Task function for run_experiment(). Called once per dataset item with
    the item's input; must return a plain dict (the evaluators' `output`)."""

    async def target(item_input: dict) -> dict:
        thread_id = f"eval-{uuid.uuid4()}"
        raw_text = item_input.get("text", "")
        if not raw_text:
            print(f"WARNING: Empty or missing 'text' in input: {item_input}")
        turns = [t.strip() for t in raw_text.split(TURN_SEP)]
        tool_outputs: list[str] = []
        tools_called: list[str] = []
        answer = "error"
        blocked_by: str | None = None
        seen = 0

        for turn in turns:
            blocked_by = None
            user_text = turn[:max_input_chars]
            if not user_text:
                continue

            if hasattr(pii_in, "mask"):
                user_text = pii_in.mask(user_text)

            # Guardrail checks now go through call_with_retry too, so they
            # count against RATE_LIMITER just like the graph call does.
            if not await call_with_retry(lambda: guardrails.check_input(user_text)):
                answer, blocked_by = BLOCKED_REPLY, "input_rail"
                continue

            result = await call_with_retry(lambda: graph.ainvoke(
                {"messages": [HumanMessage(content=user_text)]},
                config={
                    "configurable": {"thread_id": thread_id, "page_id": page_id},
                    "recursion_limit": recursion_limit,
                    "callbacks": [langfuse_handler],
                },
            ))
            msgs = result["messages"]
            for m in msgs[seen:]:
                if getattr(m, "type", "") == "tool":
                    tool_outputs.append(str(m.content))
                    name = getattr(m, "name", None) or "unknown"
                    tools_called.append(TOOL_ALIASES.get(name, name))
            seen = len(msgs)
            reply = _reply_text(msgs[-1].content)

            if not await call_with_retry(lambda: guardrails.check_output(reply, user_text=user_text)):
                answer, blocked_by = BLOCKED_REPLY, "output_rail"
                continue

            if hasattr(pii_out, "mask"):
                reply = pii_out.mask(reply)
            answer = reply

        return {
            "answer": answer,
            "tool_outputs": tool_outputs,
            "tools_called": tools_called,
            "blocked_by": blocked_by,
            "turns": len(turns),
        }

    return target


def _resolve_system_prompt(prompts_module, business_name: str) -> str:
    for name in ("build_system_prompt", "get_system_prompt", "make_system_prompt"):
        fn = getattr(prompts_module, name, None)
        if callable(fn):
            try:
                return fn(business_name)
            except TypeError:
                return fn()
    for name in ("SYSTEM_PROMPT", "SYSTEM", "PROMPT"):
        val = getattr(prompts_module, name, None)
        if isinstance(val, str):
            return val.format(business_name=business_name) if "{business_name}" in val else val
    raise RuntimeError(
        "could not find the system prompt in app.agent.prompts — expected "
        "build_system_prompt()/SYSTEM_PROMPT. Add the real name to "
        "_resolve_system_prompt()."
    )


async def main(args: argparse.Namespace) -> None:
    dataset = langfuse.get_dataset(DATASET)
    items = dataset.items
    if args.category:
        items = [i for i in items if isinstance(i.metadata, dict) and i.metadata.get("category") == args.category]
    if args.limit:
        items = items[: args.limit]
    n = len(items)

    if n == 0:
        print(f"no examples matched category={args.category!r} — check the dataset in the Langfuse UI")
        sys.exit(1)

    est = round(n * CALLS_PER_TURN)
    print(f"{n} examples selected  ->  ~{est} Gemini calls (+ up to {2 * n} if NeMo self-check rails are live)")
    if args.dry_run:
        print("dry run: nothing executed.")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.checkpoint.memory import MemorySaver

    from app.agent import prompts
    from app.agent.graph import MAX_INPUT_CHARS, RECURSION_LIMIT, build_graph, run_turn
    from app.agent.security import nemo_guardrails, pii_input_middleware, pii_output_middleware
    from app.agent.tools import make_tools
    from app.config import get_settings
    from app.db import engine as db_engine

    drift = check_drift(run_turn)
    if drift:
        warnings.warn(
            f"run_turn() no longer references {drift} — the eval target mirrors a "
            f"pipeline production no longer runs. Update make_target() first.",
            stacklevel=1,
        )

    settings = get_settings()
    if hasattr(db_engine, "init_engine"):
        await db_engine.init_engine(settings.async_database_url)

    print(f"Seeding database for page_id='{args.page_id}'...")
    try:
        nemo_guardrails.initialize()

        llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
        tools = make_tools("eval-psid", args.page_id, db_engine.SessionFactory)
        system_prompt = _resolve_system_prompt(prompts, args.business_name)
        graph = build_graph(llm, tools, system_prompt, checkpointer=MemorySaver())

        target = make_target(
            graph, nemo_guardrails, pii_input_middleware, pii_output_middleware,
            page_id=args.page_id, max_input_chars=MAX_INPUT_CHARS, recursion_limit=RECURSION_LIMIT,
        )

        experiment_name = f"{args.experiment}-{args.category or 'all'}"
        print(f"Starting evaluation run: {experiment_name}...")

        # run_experiment() owns concurrency, tracing, and score/trace linking —
        # this is what removes the trace-id bug from the hand-rolled version.
        # VERIFY against your installed langfuse version: exact param names for
        # the task function's input, and whether it's item.input or an `item=`
        # kwarg, can differ between SDK point releases.
        result = dataset.run_experiment(
            name=experiment_name,
            task=lambda item: target(item.input if isinstance(item.input, dict) else {"text": item.input}),
            evaluators=ALL_EVALUATORS,
            max_concurrency=2,
            metadata={
                "dataset": DATASET,
                "category": args.category or "all",
                "model": settings.gemini_model,
                "prompt_version": os.getenv("PROMPT_VERSION", "unversioned"),
                "guardrails_loaded": bool(getattr(nemo_guardrails, "rails", None)),
                "page_id": args.page_id,
            },
        )

        print(result.format())
        langfuse.flush()
        print(f"Evaluation complete. Results posted to Langfuse under run '{experiment_name}'.")
        pass
    finally:
        if hasattr(db_engine, "dispose_engine"):
            await db_engine.dispose_engine()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the agent on the English Langfuse dataset.")
    p.add_argument("--category", help="one category, e.g. rag_groundedness (metadata filter)")
    p.add_argument("--limit", type=int, help="cap the number of examples")
    p.add_argument("--experiment", default="en-baseline", help="experiment name prefix")
    p.add_argument("--page-id", default="demo", help="tenant; matches seed_products.sql")
    p.add_argument("--business-name", default="Nova Gadgets")
    p.add_argument("--dry-run", action="store_true", help="print the budget estimate and exit")
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))