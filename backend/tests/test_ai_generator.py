"""
Unit tests for AIGenerator's sequential tool-calling behavior in
backend/ai_generator.py.

The real anthropic.Anthropic client is replaced with a Mock so no network
call is made; these tests only check that AIGenerator drives the
Anthropic tool-use protocol (and the ToolManager it's handed) correctly,
asserting on external behavior -- API calls made (count, kwargs), tools
executed (name, args, order), and the final returned string -- not on
internal round-counter state.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call

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
        generator.client.messages.create.return_value = make_response(
            "end_turn", [text_block("ok")]
        )
        tools = [{"name": "search_course_content"}]

        generator.generate_response("hi", tools=tools, tool_manager=MagicMock())

        _, kwargs = generator.client.messages.create.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self, generator):
        generator.client.messages.create.return_value = make_response(
            "end_turn", [text_block("ok")]
        )

        generator.generate_response("hi")

        _, kwargs = generator.client.messages.create.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs


class TestSingleRoundToolUse:
    """A single tool call followed by Claude answering directly (no cap hit)."""

    def test_calls_search_tool_and_returns_final_answer(self, generator):
        first = make_response(
            "tool_use",
            [
                tool_use_block(
                    "search_course_content", {"query": "MCP servers"}, id_="tu_1"
                )
            ],
        )
        second = make_response(
            "end_turn", [text_block("Here is how to build an MCP server...")]
        )
        generator.client.messages.create.side_effect = [first, second]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = (
            "[MCP Course - Lesson 4]\nHow to build a server"
        )

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

    def test_second_round_still_offers_tools(self, generator):
        """Core fix: round 2 must still be tool-enabled, unlike the old
        single-round behavior where the follow-up call never had tools."""
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
        assert second_call_kwargs["tools"] == [{"name": "search_course_content"}]
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    def test_conversation_history_included_in_system_prompt(self, generator):
        generator.client.messages.create.return_value = make_response(
            "end_turn", [text_block("ok")]
        )

        generator.generate_response(
            "hi", conversation_history="User: hi\nAssistant: hello"
        )

        _, kwargs = generator.client.messages.create.call_args
        assert "Previous conversation" in kwargs["system"]
        assert "User: hi" in kwargs["system"]


class TestMultiRoundToolUse:
    """Sequential tool calling across rounds, up to the MAX_TOOL_ROUNDS cap."""

    def test_two_tool_rounds_then_forced_synthesis_call(self, generator):
        round1 = make_response(
            "tool_use", [tool_use_block("get_course_outline", {"course_name": "X"}, id_="tu_1")]
        )
        round2 = make_response(
            "tool_use", [tool_use_block("search_course_content", {"query": "topic"}, id_="tu_2")]
        )
        round3 = make_response("end_turn", [text_block("Course Y covers the same topic.")])
        generator.client.messages.create.side_effect = [round1, round2, round3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["Lesson 4: Topic A", "Course Y - Lesson 2"]

        result = generator.generate_response(
            "Find a course covering the same topic as lesson 4 of course X",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "Course Y covers the same topic."
        assert generator.client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_args_list == [
            call("get_course_outline", course_name="X"),
            call("search_course_content", query="topic"),
        ]

    def test_tools_omitted_only_on_final_forced_call(self, generator):
        round1 = make_response("tool_use", [tool_use_block("get_course_outline", {"course_name": "X"})])
        round2 = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q"})])
        round3 = make_response("end_turn", [text_block("answer")])
        generator.client.messages.create.side_effect = [round1, round2, round3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline result", "search result"]

        generator.generate_response(
            "q",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        calls = generator.client.messages.create.call_args_list
        assert "tools" in calls[0].kwargs
        assert "tools" in calls[1].kwargs
        assert "tools" not in calls[2].kwargs
        assert "tool_choice" not in calls[2].kwargs

    def test_second_round_direct_answer_stops_at_two_calls(self, generator):
        """If Claude answers after round 2 instead of requesting a 3rd tool
        call, no forced synthesis call should happen."""
        round1 = make_response("tool_use", [tool_use_block("search_course_content", {"query": "a"})])
        round2 = make_response("end_turn", [text_block("final answer")])
        generator.client.messages.create.side_effect = [round1, round2]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        result = generator.generate_response(
            "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        assert result == "final answer"
        assert generator.client.messages.create.call_count == 2
        tool_manager.execute_tool.assert_called_once()

    def test_tool_use_ids_match_across_rounds(self, generator):
        round1 = make_response("tool_use", [tool_use_block("get_course_outline", {"course_name": "X"}, id_="tu_1")])
        round2 = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q"}, id_="tu_2")])
        round3 = make_response("end_turn", [text_block("answer")])
        generator.client.messages.create.side_effect = [round1, round2, round3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline result", "search result"]

        generator.generate_response(
            "q",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        calls = generator.client.messages.create.call_args_list
        # Round 2's messages include round 1's tool_result (tu_1) ...
        round1_result = calls[1].kwargs["messages"][2]["content"][0]
        assert round1_result["tool_use_id"] == "tu_1"
        assert round1_result["content"] == "outline result"
        # ... and round 3's messages additionally include round 2's tool_result (tu_2)
        round2_result = calls[2].kwargs["messages"][4]["content"][0]
        assert round2_result["tool_use_id"] == "tu_2"
        assert round2_result["content"] == "search result"

    def test_conversation_history_present_in_system_on_every_round(self, generator):
        round1 = make_response("tool_use", [tool_use_block("search_course_content", {"query": "a"})])
        round2 = make_response("tool_use", [tool_use_block("search_course_content", {"query": "b"})])
        round3 = make_response("end_turn", [text_block("answer")])
        generator.client.messages.create.side_effect = [round1, round2, round3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        generator.generate_response(
            "q",
            conversation_history="User: earlier question\nAssistant: earlier answer",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        for call in generator.client.messages.create.call_args_list:
            assert "earlier question" in call.kwargs["system"]

    def test_tool_manager_none_with_tool_use_response_falls_back_to_text(self, generator):
        generator.client.messages.create.return_value = make_response(
            "tool_use", [tool_use_block("search_course_content", {"query": "q"})]
        )

        result = generator.generate_response(
            "q", tools=[{"name": "search_course_content"}], tool_manager=None
        )

        assert result == ""
        assert generator.client.messages.create.call_count == 1


class TestToolExecutionErrors:
    def test_tool_failure_on_first_round_short_circuits_to_synthesis_call(self, generator):
        round1 = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q"}, id_="tu_1")])
        synthesis = make_response("end_turn", [text_block("Sorry, that lookup failed.")])
        generator.client.messages.create.side_effect = [round1, synthesis]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("boom")

        result = generator.generate_response(
            "q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        assert result == "Sorry, that lookup failed."
        assert generator.client.messages.create.call_count == 2
        tool_manager.execute_tool.assert_called_once()

        synthesis_call_kwargs = generator.client.messages.create.call_args_list[1].kwargs
        assert "tools" not in synthesis_call_kwargs
        tool_result = synthesis_call_kwargs["messages"][-1]["content"][0]
        assert tool_result["tool_use_id"] == "tu_1"
        assert "boom" in tool_result["content"]

    def test_tool_failure_on_second_round_still_returns_text_answer(self, generator):
        round1 = make_response("tool_use", [tool_use_block("get_course_outline", {"course_name": "X"})])
        round2 = make_response("tool_use", [tool_use_block("search_course_content", {"query": "q"})])
        synthesis = make_response("end_turn", [text_block("Here's what I found in the first search.")])
        generator.client.messages.create.side_effect = [round1, round2, synthesis]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["outline result", RuntimeError("boom")]

        result = generator.generate_response(
            "q",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "Here's what I found in the first search."
        assert generator.client.messages.create.call_count == 3
        assert tool_manager.execute_tool.call_count == 2

        synthesis_call_kwargs = generator.client.messages.create.call_args_list[2].kwargs
        assert "tools" not in synthesis_call_kwargs


class TestExtractText:
    def test_extracts_text_block(self):
        resp = make_response("end_turn", [text_block("hello")])
        assert AIGenerator._extract_text(resp) == "hello"

    def test_returns_empty_string_when_no_text_block_present(self):
        thinking_only = SimpleNamespace(type="thinking", thinking="...")
        resp = make_response("end_turn", [thinking_only])
        assert AIGenerator._extract_text(resp) == ""
