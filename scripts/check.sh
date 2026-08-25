#!/bin/bash
# Verify code quality without modifying files: formatting, import order, and lint.
# Intended for CI / pre-commit use — exits non-zero on any violation.
set -e

cd "$(dirname "$0")/.."

echo "Checking formatting with black..."
uv run black --check backend main.py

echo "Checking import order with isort..."
uv run isort --check-only backend main.py

echo "Linting with ruff..."
uv run ruff check backend main.py

echo "All quality checks passed."
