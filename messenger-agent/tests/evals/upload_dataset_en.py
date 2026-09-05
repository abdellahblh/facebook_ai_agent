"""Upload english_test_dataset.csv to LangSmith. Run once.

IDEMPOTENT ON NAME, NEVER DESTRUCTIVE. If the dataset already exists this
refuses and exits 1 — it does not delete, overwrite, or create a duplicate.
A revised dataset gets a new --name (…-v2) so every past experiment stays
attached to the exact rows it was scored against.

WHY upload_csv AND NOT create_examples FROM THE YAML
    The CSV is the artifact you can open, filter and hand-edit in a
    spreadsheet before uploading; the YAML is the source of truth the CSV is
    generated from. Uploading the CSV means what you reviewed is what ran.

    The price of that choice: every reference_outputs value arrives as a
    STRING — "0", "1", "" — not as Python bools/None. evaluators.py handles
    this with a _flag() coercion helper. If you write a new evaluator, use
    it; bool("0") is True.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from langsmith import Client

HERE = pathlib.Path(__file__).parent
DEFAULT_CSV = HERE / "english_test_dataset.csv"
DEFAULT_NAME = "english-production-readiness-v1"

INPUT_KEYS = ["text"]
OUTPUT_KEYS = [
    "expect_tool",
    "expect_handoff",
    "expect_not_found",
    "expect_fact",
    "expect_fact_source_fr",
]
# id, category, trap, confidence are NOT listed above on purpose: LangSmith
# keeps unmapped CSV columns as example metadata, which is what lets
# run_experiment_en.py slice by --category and what makes a failing row
# show its own `trap` explanation in the UI.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--name", default=DEFAULT_NAME)
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print(f"no such file: {csv_path}")
        sys.exit(1)

    client = Client()

    if any(True for _ in client.list_datasets(dataset_name=args.name)):
        print(
            f"dataset {args.name!r} already exists — refusing to duplicate.\n"
            f"Pass --name {args.name.rsplit('-v', 1)[0]}-v2 for a revised set."
        )
        sys.exit(1)

    dataset = client.upload_csv(
        csv_file=str(csv_path),
        input_keys=INPUT_KEYS,
        output_keys=OUTPUT_KEYS,
        name=args.name,
        description=(
            "English production-readiness eval for the Messenger/Chatwoot "
            "support agent: 195 rows, 11 categories. Generated from "
            "english_test_dataset.yaml; regenerate the CSV, do not hand-edit."
        ),
        data_type="kv",
    )
    uploaded = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
    print(f"created {dataset.name}  id={dataset.id}")
    print(f"{uploaded} examples uploaded (expected 195)")
    if uploaded != 195:
        print("COUNT MISMATCH — inspect the dataset in the UI before running anything.")
        sys.exit(1)


if __name__ == "__main__":
    main()
