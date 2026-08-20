"""
Unit tests for CourseSearchTool.execute() in backend/search_tools.py.

VectorStore is replaced with a Mock, so these run instantly and never touch
the real ChromaDB store (which the running dev server may already hold
open -- see test_rag_system.py's module docstring for why that matters).
"""
from unittest.mock import MagicMock

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_store():
    return MagicMock()


class TestCourseSearchToolExecute:
    def test_returns_formatted_results_with_lesson_context(self):
        store = make_store()
        store.search.return_value = SearchResults(
            documents=["Lesson content about MCP servers."],
            metadata=[{"course_title": "MCP Course", "lesson_number": 4}],
            distances=[0.1],
        )
        store.get_lesson_link.return_value = "https://example.com/lesson4"

        tool = CourseSearchTool(store)
        result = tool.execute(query="MCP servers", course_name="MCP", lesson_number=4)

        assert "[MCP Course - Lesson 4]" in result
        assert "Lesson content about MCP servers." in result
        store.get_lesson_link.assert_called_once_with("MCP Course", 4)

    def test_last_sources_populated_with_lesson_link(self):
        store = make_store()
        store.search.return_value = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.1],
        )
        store.get_lesson_link.return_value = "https://example.com/l2"

        tool = CourseSearchTool(store)
        tool.execute(query="q")

        assert tool.last_sources == [
            {"text": "MCP Course - Lesson 2", "link": "https://example.com/l2"}
        ]

    def test_course_level_result_without_lesson_number_uses_course_link(self):
        store = make_store()
        store.search.return_value = SearchResults(
            documents=["doc"],
            metadata=[{"course_title": "MCP Course", "lesson_number": None}],
            distances=[0.1],
        )
        store.get_course_link.return_value = "https://example.com/course"

        tool = CourseSearchTool(store)
        tool.execute(query="q")

        store.get_course_link.assert_called_once_with("MCP Course")
        assert tool.last_sources == [
            {"text": "MCP Course", "link": "https://example.com/course"}
        ]

    def test_search_error_is_returned_verbatim_and_no_sources_recorded(self):
        store = make_store()
        store.search.return_value = SearchResults.empty("No course found matching 'Nope'")

        tool = CourseSearchTool(store)
        result = tool.execute(query="q", course_name="Nope")

        assert result == "No course found matching 'Nope'"
        assert tool.last_sources == []

    def test_empty_results_message_includes_filters(self):
        store = make_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

        tool = CourseSearchTool(store)
        result = tool.execute(query="q", course_name="MCP", lesson_number=3)

        assert result == "No relevant content found in course 'MCP' in lesson 3."

    def test_empty_results_message_without_filters(self):
        store = make_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

        tool = CourseSearchTool(store)
        result = tool.execute(query="q")

        assert result == "No relevant content found."

    def test_execute_passes_query_course_name_and_lesson_number_to_store(self):
        store = make_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

        tool = CourseSearchTool(store)
        tool.execute(query="what is mcp", course_name="MCP", lesson_number=2)

        store.search.assert_called_once_with(
            query="what is mcp", course_name="MCP", lesson_number=2
        )

    def test_get_tool_definition_schema(self):
        tool = CourseSearchTool(make_store())
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert definition["input_schema"]["required"] == ["query"]
        assert set(definition["input_schema"]["properties"]) == {
            "query", "course_name", "lesson_number"
        }


class TestToolManagerWithSearchTool:
    def test_execute_tool_dispatches_to_registered_search_tool(self):
        store = make_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)

        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="q")
        assert result == "No relevant content found."

    def test_unknown_tool_name_returns_error_string_not_exception(self):
        manager = ToolManager()
        result = manager.execute_tool("not_a_real_tool", query="q")
        assert result == "Tool 'not_a_real_tool' not found"
