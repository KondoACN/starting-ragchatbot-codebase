"""
Tests for the FastAPI endpoints (/, /api/query, /api/courses,
/api/session/{id}) that back the RAG chat frontend.

These hit `client` (see conftest.py), a TestClient wrapping a route-for-route
copy of backend/app.py's API with the real RAGSystem swapped for a
MagicMock. That avoids the app.mount("/", StaticFiles(...)) call in
backend/app.py, which points at a frontend/ directory that isn't present in
the test environment and would fail at import time.
"""


class TestRoot:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200


class TestQueryEndpoint:
    def test_query_without_session_id_creates_one(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-id"
        assert data["answer"] == "This is a test answer."
        assert data["sources"] == []
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "test-session-id")

    def test_query_with_session_id_reuses_it(self, client, mock_rag_system):
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "existing-session"},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "existing-session"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "existing-session")

    def test_query_returns_sources_with_links(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "MCP stands for Model Context Protocol.",
            [
                {"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"},
                {"text": "MCP Course - Lesson 2", "link": None},
            ],
        )

        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        sources = response.json()["sources"]
        assert sources == [
            {"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"},
            {"text": "MCP Course - Lesson 2", "link": None},
        ]

    def test_query_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={"session_id": "abc"})
        assert response.status_code == 422

    def test_query_rag_system_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("vector store unavailable")

        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 500
        assert response.json()["detail"] == "vector store unavailable"


class TestCoursesEndpoint:
    def test_get_courses_returns_analytics(self, client, mock_rag_system):
        response = client.get("/api/courses")

        assert response.status_code == 200
        assert response.json() == {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }

    def test_get_courses_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("chromadb down")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert response.json()["detail"] == "chromadb down"


class TestSessionDeleteEndpoint:
    def test_delete_session_succeeds(self, client, mock_rag_system):
        response = client.delete("/api/session/test-session-id")

        assert response.status_code == 200
        assert response.json() == {"success": True}
        mock_rag_system.session_manager.delete_session.assert_called_once_with(
            "test-session-id"
        )

    def test_delete_session_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.session_manager.delete_session.side_effect = RuntimeError(
            "session store unavailable"
        )

        response = client.delete("/api/session/test-session-id")

        assert response.status_code == 500
        assert response.json()["detail"] == "session store unavailable"
