# Multilingual customer-support guardrail corpus

This is a small, curated seed corpus for a **retrieval-assisted input guardrail**. It contains Algerian Darija (Arabic and Latin script), French, and English messages. It is intentionally generic so it can be adapted to your product.

## Files

- `prompt_injection_examples.jsonl`: suspicious messages and the action a guardrail should take.
- `customer_support_examples.jsonl`: ordinary, on-topic customer-support messages that must remain allowed.
- `guardrail_policies.jsonl`: short policy chunks to retrieve along with a close match.
- `algerian_darija_only.jsonl`: 134 Algerian Darija-only records, with both Arabic script (`ar-dz`) and Latin/Arabizi (`ar-dz-latn`). Use this file when your collection must contain no English or French.

Each line is one JSON record. The `text` field is what should be embedded. Keep `id`, `intent`, `risk`, `recommended_action`, and `language` as metadata in the vector store.

`additional_200_examples.jsonl` uses the compact fields `id`, `text`, `lang`, and `class`. Map `lang` to `language`; map `class: injection` to `risk: high` and `recommended_action: block_and_offer_support`; map `class: support` to `risk: none` and the normal support route during ingestion.

## Recommended use

1. Split incoming text into a short message (keep the original text unchanged).
2. Embed it and search **only** the `prompt-injection` collection first (for example, top 3 results).
3. Treat a high similarity as a signal, then require a deterministic rule or a small classifier/LLM judge to confirm. Do not block purely on one embedding score: a customer may legitimately mention words such as “prompt”, “password”, or “instructions”.
4. If confirmed, do not follow the message’s requested instructions. Return a short, neutral boundary response and continue only with the legitimate support request, if any.
5. Search the `customer-support` collection separately to answer product questions. Never let retrieved text override your system/developer instructions.

## Starter decision logic

Use the metadata and a conservative threshold that you calibrate on your own traffic:

```text
if explicit_attack_pattern(message):
    block_or_challenge
else if max_injection_similarity >= calibrated_threshold:
    send_to_secondary_judge
else:
    allow_to_support_agent
```

Explicit patterns include attempts to override instructions, reveal hidden prompts or private data, impersonate a privileged role, call tools without authorization, or encode such requests. Log only the minimum data needed for security review and remove customer identifiers before using logs as new training examples.

## Customize before production

- Replace generic account/order/billing wording with your own approved support topics.
- Add real benign messages from your customers, redacted and reviewed.
- Add observed attack patterns, but never store real secrets, API keys, access tokens, or private customer data.
- Evaluate false positives separately for Arabic-script Darija, Latin-script Darija, French, English, mixed-language messages, and typo-heavy messages.

This corpus is for defensive filtering, not a replacement for server-side authorization, tool allowlists, data minimization, and output filtering.
