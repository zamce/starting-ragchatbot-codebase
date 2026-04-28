#!/usr/bin/env bash
# Frontend code quality checks: formatting (Prettier) and linting (ESLint)
set -e

FRONTEND_DIR="$(cd "$(dirname "$0")/../frontend" && pwd)"

cd "$FRONTEND_DIR"

echo "=== Frontend Quality Checks ==="
echo ""

echo "--- Prettier (formatting check) ---"
npx prettier --check .
echo "Formatting: OK"
echo ""

echo "--- ESLint (linting) ---"
npx eslint script.js eslint.config.js
echo "Linting: OK"
echo ""

echo "All checks passed."
