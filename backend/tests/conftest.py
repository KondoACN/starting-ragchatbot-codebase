"""
Shared pytest fixtures for the backend test suite.

`test_app` mirrors the API routes defined in backend/app.py (query, courses,
session delete, root) but wires them to a mocked RAGSystem and skips the
`app.mount("/", StaticFiles(directory="../frontend"), ...)` call. That mount
points at a relative frontend/ directory that doesn't exist in the test
environment (and would raise at import time if we imported backend.app
directly), and it would also drag in a real RAGSystem -- with a real
ChromaDB client and sentence-transformers model -- as an import-time side
effect. Redefining the routes here keeps API tests fast, hermetic, and
independent of the frontend build.
"""
import sys
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

# backend/*.py modules use flat imports (e.g. `from vector_store import ...`),
# so the backend/ directory itself must be on sys.path for tests to import them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Request/response models, mirroring backend/app.py
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    """A single source citation, with an optional link to the lesson/course"""
    text: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem, with sane default returns."""
    mock = MagicMock()
    mock.query.return_value = ("This is a test answer.", [])
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    mock.session_manager.create_session.return_value = "test-session-id"
    return mock


@pytest.fixture
def test_app(mock_rag_system):
    """
    A FastAPI app exposing the same /api/* routes as backend/app.py, wired to
    `mock_rag_system` instead of a real RAGSystem, and without the static
    file mount.
    """
    app = FastAPI(title="Course Materials RAG System (test)")

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        try:
            mock_rag_system.session_manager.delete_session(session_id)
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """A TestClient bound to `test_app`, for exercising the API endpoints."""
    return TestClient(test_app)
