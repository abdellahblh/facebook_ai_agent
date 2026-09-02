"""Measure the darija eval set. Baseline first, variants after.

Usage:
    python evals/run_darija.py                 # inspect the set, no API calls
    python evals/run_darija.py --normalize     # show normalisation per message

Deliberately does NOT call an LLM yet. Phase 2 of the plan is to measure the
baseline, and that needs the real agent wired in. This script exists now so
the set is inspectable and the scoring shape is fixed before any numbers
exist — it is much harder to stay honest about a metric invented after seeing
the results.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.nlp.arabizi import (
    build_search_query,
    looks_like_arabizi,
    normalize_arabizi,
)

EVAL_FILE = Path(__file__).parent / "darija_messages.yaml"


def load() -> list[dict]:
    return yaml.safe_load(EVAL_FILE.read_text(encoding="utf-8"))["messages"]


def summarise(messages: list[dict]) -> None:
    print(f"\n{len(messages)} messages\n")
    for field in ("script", "intent", "confidence"):
        counts = collections.Counter(m[field] for m in messages)
        print(f"  {field:11}", dict(counts.most_common()))

    needs_policy = sum(1 for m in messages if m.get("policy"))
    needs_product = sum(1 for m in messages if m.get("product"))
    print(f"\n  {needs_policy} need a policy retrieved, {needs_product} name a product")

    low = [m for m in messages if m["confidence"] != "high"]
    if low:
        print(f"\n  ⚠️  {len(low)} rows are medium/low confidence — a native")
        print("      speaker should check these before any number is trusted:")
        for m in low:
            print(f"        {m['confidence']:6} {m['id']:12} {m['text'][:58]}")


def show_normalisation(messages: list[dict]) -> None:
    """What the retrieval query would become. Read this before trusting it:
    if normalisation mangles a product name here, it will mangle it live."""
    print("\nNormalisation preview (retrieval query only)\n")
    changed = 0
    for m in messages:
        raw = m["text"]
        norm = normalize_arabizi(raw)
        if norm == raw:
            continue
        changed += 1
        print(f"  {m['id']}")
        print(f"    in   {raw}")
        print(f"    out  {norm}")
        print(f"    query {build_search_query(raw)[:110]}")
        # The check that matters: did a product name survive?
        if m.get("product"):
            print(f"    product expected: {m['product']}")
        print()
    detected = sum(1 for m in messages if looks_like_arabizi(m["text"]))
    print(f"  {changed}/{len(messages)} messages changed; {detected} flagged as darija")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true", help="preview normalisation")
    args = parser.parse_args()

    messages = load()
    summarise(messages)
    if args.normalize:
        show_normalisation(messages)

    print("\nNot yet measured: intent accuracy and recall@3 need the agent wired in.")
    print("That is Phase 2. This script fixes the scoring shape first, on purpose.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
