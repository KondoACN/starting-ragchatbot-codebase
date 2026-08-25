# Code Quality Tooling

## Scope note

The original request asked for `black` (a Python-only formatter) restricted to "front-end
features." Since this repo's frontend is plain JS/HTML/CSS — `black` has nothing to format
there — this was raised with the user, who chose to apply the quality tooling to the
**backend Python code** (`backend/` + `main.py`) instead. No frontend files were touched.

## What was added

- **Dev dependencies** (via `uv add --dev`, never hand-edited): `black`, `isort`, `ruff`.
- **Tool config** in `pyproject.toml`:
  - `[tool.black]` — line length 88, target `py313`.
  - `[tool.isort]` — `profile = "black"` so import sorting doesn't fight black's formatting.
  - `[tool.ruff]` / `[tool.ruff.lint]` — line length 88, `select = ["E", "F", "W"]`,
    `E501` ignored (black already owns line length).
  - `[tool.ruff.lint.per-file-ignores]` — `backend/app.py` ignores `E402`, since it
    intentionally calls `warnings.filterwarnings()` before its imports to suppress a
    `resource_tracker` warning raised during import.
- **Dev scripts** in `scripts/`:
  - `scripts/format.sh` — runs `isort` then `black` to auto-fix import order and style.
  - `scripts/lint.sh` — runs `ruff check` (no changes made).
  - `scripts/check.sh` — non-mutating verification (`black --check`, `isort --check-only`,
    `ruff check`); intended for CI/pre-commit use, exits non-zero on any violation.
- `.gitignore` — added `.ruff_cache/`.
- `CLAUDE.md` — documented the new scripts and removed the now-stale "no lint tooling"
  note.

## Formatting consistency pass

Ran `scripts/format.sh` across `backend/` and `main.py` once to normalize the existing
codebase, then `scripts/lint.sh` to catch what formatting alone couldn't:

- Reformatted all 12 backend modules with `black` + `isort` (quote style, import order/
  grouping, line wrapping) — no behavior changes.
- Removed genuinely dead imports ruff flagged: unused `typing.Dict` (`models.py`),
  unused `CourseChunk`/`Lesson` (`rag_system.py`), unused `typing.Protocol`
  (`search_tools.py`), unused `SentenceTransformer` (`vector_store.py`).
- `backend/app.py` had a duplicate, mid-file import block (`import os`, `from pathlib
  import Path`, `from fastapi.staticfiles import StaticFiles`, `from fastapi.responses
  import FileResponse`) left over from when the dev static-file handler was added. `os`
  and `StaticFiles` were re-imports of names already imported at the top of the file
  (shadowing, flagged by ruff as F811), and `Path` was unused. Consolidated the one
  import that was actually needed (`FileResponse`) into the top-of-file import block and
  deleted the rest.
- Renamed an ambiguous lambda parameter `l` to `lesson` in
  `search_tools.py` (`CourseOutlineTool`), per ruff's `E741`.

All changes here are mechanical/formatting or dead-code removal — no functional logic
was changed. `./scripts/check.sh` passes cleanly, and `uv run pytest` (`backend/tests/`)
still passes (24/24 tests that don't require a real embedding model backend; see note
below).

## Pre-existing environment issue (unrelated to this change)

While verifying the test suite, several heavy ML dependencies in this worktree's
`.venv` turned out to be corrupted independently of any code change here: `numpy`
(false "importing from source tree" error), `torch` (`cannot import name 'Tensor'
... unknown location`), and `transformers` (`Could not import module
'AutoModelForSequenceClassification'`). Each reproduced on a bare `import <pkg>` with
zero code changes involved, and `uv` logged "Failed to hardlink files; falling back to
full copy" while installing in this environment, which is the likely cause of the
degraded/partial installs. `numpy`, `scipy`, and `torch` were repaired via
`uv sync --reinstall-package <name>`; `transformers` (and therefore
`sentence-transformers`, and the 2 tests that spin up a real embedding model) was left
as-is, since chasing each broken package individually is outside the scope of this
change. If this blocks other work, run:

```bash
uv sync --reinstall-package transformers --reinstall-package sentence-transformers
```

(or `uv sync --reinstall` to rebuild the whole venv). None of this affects
`black`/`isort`/`ruff`, which have no dependency on the ML stack — `./scripts/check.sh`
and 24/26 backend tests (all but the 2 requiring a real embedding model) pass cleanly.
