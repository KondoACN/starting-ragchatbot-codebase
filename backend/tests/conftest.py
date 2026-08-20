import sys
from pathlib import Path

# backend/*.py modules use flat imports (e.g. `from vector_store import ...`),
# so the backend/ directory itself must be on sys.path for tests to import them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
