from typing import Any, Dict, List, Optional, Tuple

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool-enabled API rounds per query. Claude can
    # use the result of one tool call to decide on another (e.g. look up a
    # course outline, then search using a lesson title it just learned) before
    # a final round forces a text-only answer.
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: Use **only** for questions about specific course content or detailed educational materials (e.g. explanations, examples, or topics covered within lessons)
- **get_course_outline**: Use for questions about a course's structure, syllabus, or lesson list — e.g. "what lessons are in X", "show me the outline of X", "what does course X cover"
- **Up to 2 sequential tool calls per query**: you may call a tool, review its result, and then call another tool (the same one again with refined input, or a different one) if the first result tells you what you still need to look up — e.g. look up a course's outline to find a lesson title, then search for that title in another course
- Only make a second tool call if the first result didn't give you enough to answer; do not call a tool again just to double-check an already-sufficient result
- Synthesize all tool results gathered so far into one accurate, fact-based response
- If a tool yields no results, state this clearly without offering alternatives

Outline Responses:
- When using get_course_outline, always include in your answer: the course title, the course link, and every lesson listed with its number and title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use the appropriate tool(s) first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool output"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        # Extended thinking is disabled: left implicit, it occasionally makes the
        # model end its turn with an empty text block after a thinking block,
        # producing a blank answer even though sources were found.
        self.base_params = {
            "model": self.model,
            "max_tokens": 800,
            "thinking": {"type": "disabled"},
        }

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional sequential tool usage and
        conversation context. Claude may use tools across up to
        MAX_TOOL_ROUNDS API rounds, each informed by the previous round's
        tool results, before producing a final answer.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        return self._run_tool_loop(messages, system_content, tools, tool_manager)

    def _run_tool_loop(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        tools: Optional[List],
        tool_manager,
    ) -> str:
        """
        Drive up to MAX_TOOL_ROUNDS rounds of tool-enabled API calls.

        Each round: call Claude with tools attached, and if it responds with
        tool_use, execute the requested tool(s) and feed the results back for
        the next round. The loop ends as soon as Claude responds without a
        tool_use block (it's ready to answer), and always terminates after
        MAX_TOOL_ROUNDS rounds or immediately if a tool call fails - in
        either case with one final tools-omitted call so Claude is forced to
        produce a text answer from whatever was gathered so far.
        """
        for _ in range(self.MAX_TOOL_ROUNDS):
            api_params = self._build_api_params(messages, system, tools)
            response = self.client.messages.create(**api_params)

            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text(response)

            messages.append({"role": "assistant", "content": response.content})

            tool_results, tool_round_failed = self._execute_tool_calls(
                response, tool_manager
            )
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if tool_round_failed:
                break

        # Round cap reached, or a tool call failed: force a text-only answer
        # by omitting tools entirely so Claude cannot request another one.
        final_params = self._build_api_params(messages, system, tools=None)
        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response)

    def _build_api_params(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        tools: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Build Anthropic API call parameters for a single round, optionally with tools attached"""
        params = {**self.base_params, "messages": messages, "system": system}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}
        return params

    @staticmethod
    def _execute_tool_calls(
        response, tool_manager
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute every tool_use block in a response. A tool raising an
        exception is caught so it can never crash the request; a synthetic
        error result is fed back to Claude instead, same as a real result.

        Returns:
            (tool_results, failed) - tool_result content blocks to send back
            to Claude, and whether any tool call raised.
        """
        tool_results = []
        failed = False
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                except Exception as e:
                    result = f"Tool execution failed: {e}"
                    failed = True

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result,
                    }
                )
        return tool_results, failed

    @staticmethod
    def _extract_text(response) -> str:
        """Get the text content from a response, skipping any non-text blocks (e.g. extended thinking)"""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
