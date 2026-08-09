#!/usr/bin/env bash
# The pre-deploy gate. All green or it doesn't ship.
set -uo pipefail
failed=0
run() { echo ""; echo "─── $1 ───"; shift; "$@" && echo "  ok" || { echo "  FAILED"; failed=1; }; }

run "lint"            ruff check .
run "format"          ruff format --check .
run "types"           mypy app
run "dependency CVEs" pip-audit
run "tests"           pytest -q

echo ""
[ "$failed" -ne 0 ] && { echo "SOME CHECKS FAILED - do not deploy"; exit 1; }
echo "ALL CHECKS PASSED - safe to deploy"
