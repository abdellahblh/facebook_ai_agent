#!/usr/bin/env bash
# The pre-deploy gate. All green or it doesn't ship.
set -uo pipefail
failed=0

run() { 
    echo ""
    echo "─── $1 ───"
    shift
    "$@" && echo "  ok" || { echo "  FAILED"; failed=1; }
}

# 1. Code Quality & Formatting
run "linter"             ruff check .
run "formatting check"   ruff format .

# 2. Type Safety
run "types"              mypy app

# 3. Security Audits
run "code security"      bandit -q -r app
run "dependency CVEs"    pip-audit --environment .venv

# 4. Automated Tests
run "tests"              pytest -q

echo ""
if [ "$failed" -ne 0 ]; then
    echo "❌ SOME CHECKS FAILED - do not deploy"
    exit 1
fi

echo "✅ ALL CHECKS PASSED - safe to deploy"