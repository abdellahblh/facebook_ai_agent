"""Run the agent over the English dataset on LangSmith and score it.

MIRRORS run_turn() STEP FOR STEP — AND MUST KEEP DOING SO
    app.agent.graph.run_turn returns only the final string. Four evaluators
    need what happened in between: which tools fired, what they returned,
    whether a rail blocked. So make_target() re-implements run_turn's five
    steps in the same order with the same refusal string:

        PII mask in -> NeMo input rail -> graph -> NeMo output rail -> PII mask out

    and captures the middle. check_drift() compares run_turn's source
    against that step list at startup and warns if production changed shape.
    A silent divergence here means you are scoring a pipeline nobody ships.

MULTI-TURN ROWS
    8 rows carry several turns in one cell, joined by " || ". They are fed
    sequentially on ONE thread_id with an in-memory checkpointer, so turn two
    really has turn one in its history. Tool calls accumulate across turns;
    the answer is the LAST turn's. A fresh thread_id per example means state
    never leaks between rows or between runs.

WHY MemorySaver AND NOT THE PRODUCTION CHECKPOINTER
    Eval threads would otherwise land in the production Postgres checkpoint
    tables — 195 junk conversations per run, forever, competing with the
    500 MB free tier. In-memory is per-process and gone when this exits.

BUDGET
    ~2.2 Gemini calls per turn, +1 per NeMo self-check rail per turn when
    the rails have an LLM. Full set ≈ 430-900 requests against a ~1,000/day
    free tier. --dry-run prints the estimate and exits. Run one --category
    at a time on a SEPARATE eval API key.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
import uuid
import warnings

from langchain_core.messages import HumanMessage
from langsmith import Client, aevaluate

from tests.evals.evaluators import ALL_EVALUATORS, TOOL_ALIASES

DATASET = "english-production-readiness-v1"
TURN_SEP = " || "
# Verbatim from app/agent/graph.py — the refusal production returns when a
# rail blocks. If that string changes, change it here, or blocked turns will
# be scored as normal answers.
BLOCKED_REPLY = "Sma7lna, ma n9derch njawbek 3la had l'demande."
RUN_TURN_STEPS = (
    "pii_input_middleware", "check_input", "ainvoke", "check_output", "pii_output_middleware",
)
CALLS_PER_TURN = 2.2


def check_drift(run_turn_fn) -> list[str]:
    """Names that run_turn's source no longer contains. Empty = in sync."""
    try:
        src = inspect.getsource(run_turn_fn)
    except (OSError, TypeError):
        return []
    return [step for step in RUN_TURN_STEPS if step not in src]


def _reply_text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def make_target(graph, guardrails, pii_in, pii_out, *,
                page_id: str = "demo", max_input_chars: int = 8_000, recursion_limit: int = 15):
    """Build the LangSmith target. Every dependency is injected so the
    smoke test can pass fakes and never import app.* or call Gemini."""

    async def target(inputs: dict) -> dict:
        thread_id = f"eval-{uuid.uuid4()}"
        turns = [t.strip() for t in (inputs.get("text") or "").split(TURN_SEP)]
        tool_outputs: list[str] = []
        tools_called: list[str] = []
        answer = ""
        blocked_by: str | None = None
        seen = 0  # messages already scanned; the checkpointer returns full history

        for turn in turns:
            blocked_by = None
            user_text = turn[:max_input_chars]

            # 1. PII mask on input
            if hasattr(pii_in, "mask"):
                user_text = pii_in.mask(user_text)

            # 2. NeMo input rail — production returns the refusal and never
            #    runs the graph for this turn. Later turns still run.
            if not await guardrails.check_input(user_text):
                answer, blocked_by = BLOCKED_REPLY, "input_rail"
                continue

            # 3. Graph
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_text)]},
                config={"configurable": {"thread_id": thread_id, "page_id": page_id},
                        "recursion_limit": recursion_limit},
            )
            msgs = result["messages"]
            for m in msgs[seen:]:
                if getattr(m, "type", "") == "tool":
                    tool_outputs.append(str(m.content))
                    name = getattr(m, "name", None) or "unknown"
                    tools_called.append(TOOL_ALIASES.get(name, name))
            seen = len(msgs)
            reply = _reply_text(msgs[-1].content)

            # 4. NeMo output rail
            if not await guardrails.check_output(reply, user_text=user_text):
                answer, blocked_by = BLOCKED_REPLY, "output_rail"
                continue

            # 5. PII mask on output
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
    """prompts.py has changed shape across this project. Handle the three
    shapes seen so far and fail LOUD on anything else — a wrong prompt makes
    every score meaningless, so guessing is worse than stopping."""
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


def _select_examples(client: Client, category: str | None, limit: int | None):
    """Whole dataset by name, or a metadata-filtered slice."""
    if not category and not limit:
        return DATASET, None
    kwargs = {"dataset_name": DATASET}
    if category:
        kwargs["metadata"] = {"category": category}
    if limit:
        kwargs["limit"] = limit
    examples = list(client.list_examples(**kwargs))
    return examples, len(examples)


async def main(args: argparse.Namespace) -> None:
    client = Client()
    data, n = _select_examples(client, args.category, args.limit)
    if n is None:
        n = client.read_dataset(dataset_name=DATASET).example_count or 195
    if n == 0:
        print(f"no examples matched category={args.category!r} — check the spelling against the CSV")
        sys.exit(1)

    est = round(n * CALLS_PER_TURN)
    print(f"{n} examples selected  ->  ~{est} Gemini calls (+ up to {2 * n} if NeMo self-check rails are live)")
    if args.dry_run:
        print("dry run: nothing executed.")
        return

    # Real imports only here, so --dry-run and the smoke test never need app.*
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

    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
    tools = make_tools("eval-psid", args.page_id, db_engine.SessionFactory)
    system_prompt = _resolve_system_prompt(prompts, args.business_name)
    graph = build_graph(llm, tools, system_prompt, checkpointer=MemorySaver())

    target = make_target(
        graph, nemo_guardrails, pii_input_middleware, pii_output_middleware,
        page_id=args.page_id, max_input_chars=MAX_INPUT_CHARS, recursion_limit=RECURSION_LIMIT,
    )

    results = await aevaluate(
        target,
        data=data,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=args.experiment,
        # 2, not 10: RPM is 15 and each example makes ~2.2 calls. Ten in
        # flight is ~22 concurrent requests -> 429s that land as LOW SCORES,
        # not errors. You would be measuring your rate limit.
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
    print(results)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the agent on the English LangSmith dataset.")
    p.add_argument("--category", help="one category, e.g. rag_groundedness (metadata filter)")
    p.add_argument("--limit", type=int, help="cap the number of examples")
    p.add_argument("--experiment", default="en-baseline", help="experiment name prefix")
    p.add_argument("--page-id", default="demo", help="tenant; matches seed_products.sql")
    p.add_argument("--business-name", default="Nova Gadgets")
    p.add_argument("--dry-run", action="store_true", help="print the budget estimate and exit")
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
