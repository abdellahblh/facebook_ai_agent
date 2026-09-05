# Bugs this dataset will surface immediately — fix before running

Found reading `messenger-agent/app/db/repo.py` on `algerian_version`
(commit `ba67e59`) while grounding the SQL-filter rows. Confirmed live in a
sandbox, not just read-and-guessed:

## 1. `NameError: name 'select' is not defined` — every product query crashes

```python
from sqlalchemy import Select, asc, desc, or_, func   # capital Select = the TYPE
...
stmt: Select = select(Product).where(...)             # lowercase select() = the FUNCTION, never imported
```

Reproduced:
```
CONFIRMED NameError at runtime: name 'select' is not defined
```

`find_products`, `count_products`, and `get_recent_messages` all call
`select(...)` and none of them will run. Fix: `from sqlalchemy import select`
(lowercase), alongside the existing import line.

## 2. `NameError: name 'MAX_LIMIT' is not defined`

```python
stmt = stmt.limit(min(max(limit, 1), MAX_LIMIT))
```

`MAX_LIMIT` is referenced but never defined or imported anywhere in the
file. Even after fixing bug 1, every call still crashes here. Fix: add
`MAX_LIMIT = 12` (or whatever ceiling you want) near the top of the file.

## 3. Price filtering compares a STRING column with integer bounds

```python
# models.py
price: Mapped[str] = mapped_column(String(64))   # "4500 DA" -- a string

# repo.py
if max_price is not None:
    stmt = stmt.where(Product.price <= max_price)   # comparing text to an int
```

The docstring in `find_products` even says *"price_da is an INTEGER
column... comparing the display string does not work"* — but the code
still filters on `Product.price`, the string column, not a `price_da`
integer column. There isn't one in `models.py`.

This one won't just misbehave, it will likely raise a database-level type
error in Postgres (`operator does not exist: character varying <= integer`),
or at minimum silently do a lexicographic string comparison where
`'12000 DA' <= 5000` evaluates in a way that has nothing to do with the
actual prices — the exact bug already diagnosed earlier in this project
(`'10000 DA' < '900 DA'` is `True`).

Fix: add an integer `price_da` column (migration + backfill, same pattern
used earlier in this project for the Algerian version), populate it from
`price` on write, and filter/sort on `price_da` — keep `price` as the
display string.

## Why this matters for THIS dataset specifically

`sql_filter_specific` (20 rows) and 8 of the `correctness_no_hallucination`
rows exist to exercise exactly this code path. Until bugs 1 and 2 are
fixed, every one of those 28 rows will fail identically with the same
`NameError` — which looks like "28 failures" in a report but is actually
one bug, twice. Fix bugs 1 and 2 first, THEN run the dataset, or the
failure count will be meaningless.
