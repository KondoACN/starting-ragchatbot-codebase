# Backend API testing infrastructure

Enhances the existing `backend/tests` suite (previously unit tests only for
`AIGenerator`, `RAGSystem`, and `search_tools`) with API endpoint tests, a
pytest configuration, and shared fixtures.

## What changed

### `backend/tests/test_api.py` (new)

Tests for the FastAPI endpoints used by the frontend:

- `GET /` — root responds 200
- `POST /api/query` — creates a session when none is given, reuses a
  provided `session_id`, round-trips source citations (`text`/`link`),
  returns 422 on a missing `query` field, and returns 500 with the error
  detail when `RAGSystem.query()` raises
- `GET /api/courses` — returns course analytics, returns 500 on failure
- `DELETE /api/session/{id}` — deletes a session, returns 500 on failure

### `backend/tests/conftest.py` (updated)

Added shared fixtures:

- `mock_rag_system` — a `MagicMock` standing in for `RAGSystem`, with
  default return values for `.query()`, `.get_course_analytics()`, and
  `.session_manager.create_session()`.
- `test_app` — builds a FastAPI app that mirrors the routes in
  `backend/app.py` but is wired to `mock_rag_system` and **does not**
  mount static files. `backend/app.py` calls
  `app.mount("/", StaticFiles(directory="../frontend"), ...)` at import
  time, and constructing a real `RAGSystem` there pulls in ChromaDB and
  a sentence-transformers model — neither is appropriate for a fast,
  hermetic test run, and the frontend directory isn't guaranteed to be
  resolvable from the test working directory. The route handlers are
  redefined inline instead of importing `backend.app`.
- `client` — a `TestClient` bound to `test_app`.

### `pyproject.toml` (updated)

Added `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["backend/tests"]
pythonpath = ["backend"]
addopts = "-ra --tb=short"
```

This lets `uv run pytest` be invoked from the repo root with no extra
flags and picks up `backend/tests` regardless of current working
directory.

## Verification

`uv run pytest` — 36 passed (9 `test_ai_generator.py`, 10 new
`test_api.py`, 7 `test_rag_system.py`, 10 `test_search_tools.py`). The new
API tests run in ~1s; the full run is slow only because
`test_rag_system.py`'s integration test spins up a real ChromaDB +
sentence-transformers embedding model (pre-existing, unrelated to this
change).

## Note on scope

The task description that triggered this work included boilerplate
instructing "only do this for front-end features" and to write this
summary to `frontend-changes.md`. This is backend test infrastructure
(FastAPI endpoints, pytest config, `backend/tests` fixtures), so that
instruction looks like a mismatched template carried over from an
unrelated command — flagging it rather than silently mislabeling this
file.
