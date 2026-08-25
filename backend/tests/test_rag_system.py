"""
Tests for how RAGSystem.query() (backend/rag_system.py) handles
content-related questions.

`TestQueryOrchestration` mocks every collaborator so it runs instantly and
verifies the wiring: prompt construction, tool/tool_manager pass-through,
source retrieval + reset ordering, session history updates, and that
exceptions from the AI layer are NOT swallowed (app.py's /api/query catches
them and turns them into the 500 the frontend reports as "Query failed").

`TestContentQueryAgainstRealVectorStore` is a slower integration test that
spins up a *real* ChromaDB instance in a throwaway temp directory -- never
the project's backend/chroma_db path used by the dev server. Opening a
second PersistentClient against that same path while the dev server holds
it open was observed to hang indefinitely rather than fail fast, so this
isolation is required, not just tidy. Only the Anthropic call is mocked, so
this exercises the real CourseSearchTool -> VectorStore -> ChromaDB path
used for real content questions.
"""

import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import rag_system as rag_system_module
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class FakeConfig:
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-sonnet-5"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2

    def __init__(self, chroma_path):
        self.CHROMA_PATH = chroma_path


@pytest.fixture
def mocked_rag_system(monkeypatch):
    """RAGSystem with every collaborator replaced by a Mock."""
    monkeypatch.setattr(rag_system_module, "DocumentProcessor", MagicMock())
    monkeypatch.setattr(rag_system_module, "VectorStore", MagicMock())
    monkeypatch.setattr(rag_system_module, "AIGenerator", MagicMock())
    monkeypatch.setattr(rag_system_module, "SessionManager", MagicMock())
    monkeypatch.setattr(rag_system_module, "CourseSearchTool", MagicMock())
    monkeypatch.setattr(rag_system_module, "CourseOutlineTool", MagicMock())
    monkeypatch.setattr(rag_system_module, "ToolManager", MagicMock())

    config = SimpleNamespace(
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHROMA_PATH="unused",
        EMBEDDING_MODEL="unused",
        MAX_RESULTS=5,
        ANTHROPIC_API_KEY="test",
        ANTHROPIC_MODEL="test",
        MAX_HISTORY=2,
    )
    return RAGSystem(config=config)


class TestQueryOrchestration:
    def test_content_question_is_sent_to_ai_generator_with_tools(
        self, mocked_rag_system
    ):
        rs = mocked_rag_system
        rs.ai_generator.generate_response.return_value = (
            "MCP servers expose tools, resources, and prompts."
        )
        rs.tool_manager.get_last_sources.return_value = [
            {"text": "MCP Course - Lesson 4", "link": "https://x"}
        ]

        answer, sources = rs.query("How do I create an MCP server?", session_id="s1")

        assert answer == "MCP servers expose tools, resources, and prompts."
        assert sources == [{"text": "MCP Course - Lesson 4", "link": "https://x"}]

        _, kwargs = rs.ai_generator.generate_response.call_args
        assert "How do I create an MCP server?" in kwargs["query"]
        assert kwargs["tools"] == rs.tool_manager.get_tool_definitions.return_value
        assert kwargs["tool_manager"] is rs.tool_manager

    def test_sources_are_reset_after_being_read(self, mocked_rag_system):
        rs = mocked_rag_system
        rs.ai_generator.generate_response.return_value = "answer"
        rs.tool_manager.get_last_sources.return_value = [{"text": "x", "link": None}]

        rs.query("some content question")

        rs.tool_manager.get_last_sources.assert_called_once()
        rs.tool_manager.reset_sources.assert_called_once()

    def test_session_history_is_updated_with_query_and_response(
        self, mocked_rag_system
    ):
        rs = mocked_rag_system
        rs.ai_generator.generate_response.return_value = "the answer"
        rs.tool_manager.get_last_sources.return_value = []

        rs.query("what does lesson 3 cover?", session_id="abc")

        rs.session_manager.add_exchange.assert_called_once_with(
            "abc", "what does lesson 3 cover?", "the answer"
        )

    def test_no_session_id_skips_history_lookup_and_update(self, mocked_rag_system):
        rs = mocked_rag_system
        rs.ai_generator.generate_response.return_value = "answer"
        rs.tool_manager.get_last_sources.return_value = []

        rs.query("a question", session_id=None)

        rs.session_manager.get_conversation_history.assert_not_called()
        rs.session_manager.add_exchange.assert_not_called()

    def test_ai_generator_exception_propagates_uncaught(self, mocked_rag_system):
        """RAGSystem.query has no try/except -- any exception here becomes
        the 500 that the frontend surfaces as 'Query failed'."""
        rs = mocked_rag_system
        rs.ai_generator.generate_response.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            rs.query("a content question")


@pytest.fixture(scope="module")
def real_vector_store_rag_system():
    chroma_path = tempfile.mkdtemp(prefix="rag_test_chroma_")
    config = FakeConfig(chroma_path)

    system = rag_system_module.RAGSystem(config)

    course = Course(
        title="Test MCP Course",
        course_link="https://example.com/course",
        instructor="Tester",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Intro to MCP",
                lesson_link="https://example.com/l1",
            )
        ],
    )
    chunks = [
        CourseChunk(
            course_title=course.title,
            lesson_number=1,
            chunk_index=0,
            content="MCP servers expose tools, resources, and prompts to AI applications.",
        )
    ]
    system.vector_store.add_course_metadata(course)
    system.vector_store.add_course_content(chunks)

    yield system

    shutil.rmtree(chroma_path, ignore_errors=True)


class TestContentQueryAgainstRealVectorStore:
    def test_real_search_tool_finds_seeded_content(self, real_vector_store_rag_system):
        rs = real_vector_store_rag_system
        result = rs.search_tool.execute(query="What does an MCP server expose?")

        assert "Test MCP Course" in result
        assert "tools, resources, and prompts" in result

    def test_full_query_pipeline_with_mocked_anthropic_only(
        self, real_vector_store_rag_system
    ):
        rs = real_vector_store_rag_system
        rs.ai_generator = MagicMock()
        rs.ai_generator.generate_response.side_effect = lambda **kwargs: (
            kwargs["tool_manager"].execute_tool(
                "search_course_content", query="MCP server capabilities"
            )
        )

        answer, sources = rs.query("What does an MCP server expose?")

        assert "tools, resources, and prompts" in answer
        assert sources and sources[0]["text"].startswith("Test MCP Course")
