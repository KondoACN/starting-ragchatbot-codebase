# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for Python dependency management (Python 3.13 required).

**Always use `uv` to run the server and any Python commands, and to manage all dependencies (`uv add`/`uv remove`/`uv sync`) — never call `pip` or `python` directly, and never edit `pyproject.toml`/`uv.lock` by hand.**

```bash
# Install dependencies
uv sync

# Run the app (from repo root) — starts uvicorn with --reload on port 8000
./run.sh

# Equivalent manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code quality

Formatting (`black`, `isort`) and linting (`ruff`) for the Python code in `backend/` and `main.py` are dev dependencies, driven by scripts in `scripts/`:

```bash
./scripts/format.sh   # auto-fix import order and style
./scripts/lint.sh     # ruff lint, no changes made
./scripts/check.sh    # non-mutating: format + lint verification, for CI/pre-commit
```

Config for all three tools lives in `pyproject.toml` under `[tool.black]`, `[tool.isort]`, and `[tool.ruff]`.

Requires a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`).

- Web UI: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs

There is no test suite, linter, or build step in this repo.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. It's a FastAPI backend (`backend/`) serving a static vanilla-JS frontend (`frontend/`), backed by ChromaDB for vector search and Anthropic's Claude for generation.

### Request flow

`frontend/script.js` → `POST /api/query` (`backend/app.py`) → `RAGSystem.query()` (`backend/rag_system.py`) → `AIGenerator.generate_response()` (`backend/ai_generator.py`).

The key architectural detail is that **Claude decides for itself whether to search**, via Anthropic tool calling:

1. `AIGenerator` sends the query to Claude with the `search_course_content` tool definition and `tool_choice: auto`.
2. If Claude's `stop_reason` is `tool_use`, `AIGenerator._handle_tool_execution()` runs the tool (`CourseSearchTool.execute()` in `backend/search_tools.py`), feeds the result back as a `tool_result` message, and makes a **second** Claude call to produce the final answer.
3. If Claude answers directly (general knowledge, no `tool_use`), the first call's response is used as-is — no second call.

`RAGSystem.query()` then pulls source citations off the tool via `ToolManager.get_last_sources()` (populated as a side effect of `CourseSearchTool.execute()`, reset after each query) and saves the exchange to `SessionManager` for conversation history. The system prompt enforces "one search per query maximum" and no meta-commentary about searching.

### Component responsibilities (`backend/`)

- `app.py` — FastAPI app. On startup, auto-loads all documents from `../docs` into the vector store (skips courses already indexed by title). Serves the frontend as static files.
- `rag_system.py` — orchestrator wiring all components together; the only class other modules should need to call into from the API layer.
- `ai_generator.py` — all Claude API interaction, including the two-call tool-use loop described above. System prompt lives here.
- `search_tools.py` — the `Tool` abstraction (`ABC`) and `ToolManager` registry, plus the concrete `CourseSearchTool`. New tools should implement `Tool` (`get_tool_definition()` + `execute()`) and register via `ToolManager.register_tool()`.
- `vector_store.py` — wraps ChromaDB. Maintains **two collections**: `course_catalog` (one entry per course, used to semantically resolve fuzzy course names like `"MCP"` → the real title via `_resolve_course_name`) and `course_content` (the actual chunked text, filterable by `course_title`/`lesson_number`). Course titles are used as ChromaDB IDs, so they're the de facto unique key for a course.
- `document_processor.py` — parses raw course documents into `Course`/`Lesson`/`CourseChunk` objects and sentence-aware chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP` from config). Expects a specific input format (see below).
- `session_manager.py` — in-memory (non-persistent) conversation history, capped at `MAX_HISTORY` exchanges per session.
- `models.py` — the Pydantic domain models (`Course`, `Lesson`, `CourseChunk`) shared across the above.
- `config.py` — single `Config` dataclass loaded from `.env`; change chunk size, result limits, history length, model names, etc. here.

### Expected document format (`docs/*.txt`)

`document_processor.py` parses course files with this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [lesson title]
[lesson content...]
```
The course title is the unique identifier used across both ChromaDB collections — adding a document whose title already exists in `course_catalog` is treated as a skip (see `RAGSystem.add_course_folder`), not an update.
