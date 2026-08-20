"""
Unit tests for AIGenerator's tool-calling behavior in backend/ai_generator.py.

The real anthropic.Anthropic client is replaced with a Mock so no network
call is made; these tests only check that AIGenerator drives the
Anthropic tool-use protocol (and the ToolManager it's handed) correctly.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def make_response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


@pytest.fixture
def generator():
    gen = AIGenerator(api_key="test-key", model="claude-sonnet-5")
    gen.client = MagicMock()
    return gen


class TestDirectResponseNoToolUse:
    def test_returns_text_without_calling_tools(self, generator):
        generator.client.messages.create.return_value = make_response(
            "end_turn", [text_block("Paris is the capital of France.")]
        )
        tool_manager = MagicMock()

        result = generator.generate_response(
            "What is the capital of France?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "Paris is the capital of France."
        tool_manager.execute_tool.assert_not_called()
        assert generator.client.messages.create.call_count == 1

    def test_tools_and_tool_choice_included_when_tools_provided(self, generator):
        generator.client.messages.create.return_value = make_response("end_turn", [text_block("ok")])
        tools = [{"name": "search_course_content"}]

        generator.generate_response("hi", tools=tools, tool_manager=MagicMock())

        _, kwargs = generator.client.messages.create.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self, generator):
        generator.client.messages.create.return_value = make_response("end_turn", [text_block("ok")])

        generator.generate_response("hi")

        _, kwargs = generator.client.messages.create.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs


class TestToolUseFlow:
    def test_calls_search_tool_and_returns_final_answer(self, generator):
        first = make_response(
            "tool_use",
            [tool_use_block("search_course_content", {"query": "MCP servers"}, id_="tu_1")],
        )
        second = make_response("end_turn", [text_block("Here is how to build an MCP server...")])
        generator.client.messages.create.side_effect = [first, second]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "[MCP Course - Lesson 4]\nHow to build a server"

        result = generator.generate_response(
            "How do I build an MCP server?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "Here is how to build an MCP server..."
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP servers"
        )
        assert generator.client.messages.create.call_count == 2

    def test_second_call_includes_tool_result_message_with_matching_id(self, generator):
        first = make_response(
            "tool_use",
            [tool_use_block("search_course_content", {"query": "q"}, id_="tu_42")],
        )
        second = make_response("end_turn", [text_block("answer")])
        generator.client.messages.create.side_effect = [first, second]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search result text"

        generator.generate_response(
            "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]
        assert messages[-1]["role"] == "user"
        tool_result = messages[-1]["content"][0]
        assert tool_result["tool_use_id"] == "tu_42"
        assert tool_result["content"] == "search result text"

    def test_final_call_does_not_re_expose_tools(self, generator):
        first = make_response(
            "tool_use", [tool_use_block("search_course_content", {"query": "q"})]
        )
        second = make_response("end_turn", [text_block("answer")])
        generator.client.messages.create.side_effect = [first, second]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        generator.generate_response(
            "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        second_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs

    def test_conversation_history_included_in_system_prompt(self, generator):
        generator.client.messages.create.return_value = make_response("end_turn", [text_block("ok")])

        generator.generate_response("hi", conversation_history="User: hi\nAssistant: hello")

        _, kwargs = generator.client.messages.create.call_args
        assert "Previous conversation" in kwargs["system"]
        assert "User: hi" in kwargs["system"]


class TestExtractText:
    def test_extracts_text_block(self):
        resp = make_response("end_turn", [text_block("hello")])
        assert AIGenerator._extract_text(resp) == "hello"

    def test_returns_empty_string_when_no_text_block_present(self):
        thinking_only = SimpleNamespace(type="thinking", thinking="...")
        resp = make_response("end_turn", [thinking_only])
        assert AIGenerator._extract_text(resp) == ""
