#!/bin/bash
# Lint the Python codebase with ruff.
set -e

cd "$(dirname "$0")/.."

echo "Linting with ruff..."
uv run ruff check backend main.py
