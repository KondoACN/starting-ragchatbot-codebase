#!/bin/bash
# Auto-format the Python codebase with isort (import order) and black (style).
set -e

cd "$(dirname "$0")/.."

echo "Sorting imports with isort..."
uv run isort backend main.py

echo "Formatting with black..."
uv run black backend main.py
