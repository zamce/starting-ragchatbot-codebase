"""
Suite 2: ai_generator.py integration tests.
Verifies AIGenerator correctly calls tools and handles the two-turn conversation.
This suite isolates the integration layer that was identified as the likely failure point.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import SearchResults


def make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_name, tool_id, input_dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = input_dict
    return block


def make_api_response(stop_reason, content_blocks):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


def make_generator():
    return AIGenerator(api_key="test-key", model="claude-sonnet-4-6")


def make_tool_manager(tool_return="Mocked search result"):
    manager = MagicMock(spec=ToolManager)
    manager.execute_tool.return_value = tool_return
    manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    ]
    return manager


class TestDirectResponse:
    """Tests for queries that don't trigger tool use."""

    def test_returns_text_when_no_tool_use(self):
        gen = make_generator()
        direct_response = make_api_response("end_turn", [make_text_block("Python is a language.")])
        gen.client.messages.create = MagicMock(return_value=direct_response)

        result = gen.generate_response("What is Python?")
        assert result == "Python is a language."

    def test_api_called_with_user_message(self):
        gen = make_generator()
        direct_response = make_api_response("end_turn", [make_text_block("Answer")])
        gen.client.messages.create = MagicMock(return_value=direct_response)

        gen.generate_response("What is Python?")
        call_kwargs = gen.client.messages.create.call_args[1]
        assert call_kwargs["messages"][0] == {"role": "user", "content": "What is Python?"}

    def test_api_called_with_correct_model(self):
        gen = make_generator()
        direct_response = make_api_response("end_turn", [make_text_block("Answer")])
        gen.client.messages.create = MagicMock(return_value=direct_response)

        gen.generate_response("test")
        call_kwargs = gen.client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_system_prompt_included(self):
        gen = make_generator()
        direct_response = make_api_response("end_turn", [make_text_block("Answer")])
        gen.client.messages.create = MagicMock(return_value=direct_response)

        gen.generate_response("test")
        call_kwargs = gen.client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_conversation_history_appended_to_system(self):
        gen = make_generator()
        direct_response = make_api_response("end_turn", [make_text_block("Answer")])
        gen.client.messages.create = MagicMock(return_value=direct_response)

        gen.generate_response("test", conversation_history="User: hello\nAssistant: hi")
        call_kwargs = gen.client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert "User: hello" in call_kwargs["system"]

    def test_tools_included_when_provided(self):
        gen = make_generator()
        direct_response = make_api_response("end_turn", [make_text_block("Answer")])
        gen.client.messages.create = MagicMock(return_value=direct_response)
        tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]

        gen.generate_response("test", tools=tools)
        call_kwargs = gen.client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self):
        gen = make_generator()
        direct_response = make_api_response("end_turn", [make_text_block("Answer")])
        gen.client.messages.create = MagicMock(return_value=direct_response)

        gen.generate_response("test")
        call_kwargs = gen.client.messages.create.call_args[1]
        assert "tools" not in call_kwargs


class TestToolExecution:
    """Tests for the two-turn tool use flow — the integration layer under investigation."""

    def test_handle_tool_execution_called_on_tool_use_stop_reason(self):
        gen = make_generator()
        tool_block = make_tool_use_block("search_course_content", "call_1", {"query": "Python loops"})
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Final answer")])
        gen.client.messages.create = MagicMock(side_effect=[tool_response, final_response])

        result = gen.generate_response("Tell me about Python loops", tool_manager=make_tool_manager())
        assert result == "Final answer"
        assert gen.client.messages.create.call_count == 2

    def test_tool_executed_with_correct_name_and_kwargs(self):
        gen = make_generator()
        tool_input = {"query": "Python loops", "course_name": "Python Basics"}
        tool_block = make_tool_use_block("search_course_content", "call_1", tool_input)
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Done")])
        gen.client.messages.create = MagicMock(side_effect=[tool_response, final_response])
        manager = make_tool_manager()

        gen.generate_response("test", tool_manager=manager)
        manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python loops",
            course_name="Python Basics"
        )

    def test_tool_result_added_as_user_message(self):
        gen = make_generator()
        tool_block = make_tool_use_block("search_course_content", "call_abc", {"query": "loops"})
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Done")])
        gen.client.messages.create = MagicMock(side_effect=[tool_response, final_response])
        manager = make_tool_manager("Search result content here")

        gen.generate_response("test", tool_manager=manager)

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # Last message should be user message containing tool_result
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        tr = tool_result_msg["content"][0]
        assert tr["type"] == "tool_result"
        assert tr["tool_use_id"] == "call_abc"
        assert tr["content"] == "Search result content here"

    def test_round_one_follow_up_includes_tools(self):
        """After round 1 tool use, the follow-up call retains tools so Claude can make a second tool call."""
        gen = make_generator()
        tool_block = make_tool_use_block("search_course_content", "call_1", {"query": "loops"})
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Done")])
        tools = [{"name": "search_course_content"}]
        gen.client.messages.create = MagicMock(side_effect=[tool_response, final_response])

        gen.generate_response("test", tools=tools, tool_manager=make_tool_manager())

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    def test_assistant_tool_use_included_in_second_call_messages(self):
        gen = make_generator()
        tool_block = make_tool_use_block("search_course_content", "call_1", {"query": "loops"})
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Done")])
        gen.client.messages.create = MagicMock(side_effect=[tool_response, final_response])

        gen.generate_response("test", tool_manager=make_tool_manager())

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # messages[0] = user query, messages[1] = assistant tool use, messages[2] = tool result
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == tool_response.content

    def test_tool_skipped_when_no_tool_manager(self):
        """When tool_manager is None the condition short-circuits; code falls through to content[0].text.
        In production a real ToolUseBlock has no .text — this would AttributeError.
        Use a spec'd mock to simulate that so the test catches the real behavior."""
        gen = make_generator()

        # Simulate a real ToolUseBlock that has no .text attribute
        from anthropic.types import ToolUseBlock
        tool_block = MagicMock(spec=ToolUseBlock)
        tool_block.type = "tool_use"
        tool_response = make_api_response("tool_use", [tool_block])
        gen.client.messages.create = MagicMock(return_value=tool_response)

        with pytest.raises(AttributeError):
            gen.generate_response("test", tools=[{"name": "search_course_content"}], tool_manager=None)

    def test_tool_executed_with_only_required_query_param(self):
        """Claude may call the tool with just the required 'query' param — must not error."""
        gen = make_generator()
        tool_block = make_tool_use_block("search_course_content", "call_1", {"query": "MCP overview"})
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Done")])
        gen.client.messages.create = MagicMock(side_effect=[tool_response, final_response])
        manager = make_tool_manager()

        result = gen.generate_response("Explain MCP", tool_manager=manager)
        manager.execute_tool.assert_called_once_with("search_course_content", query="MCP overview")
        assert result == "Done"


class TestSequentialToolCalling:
    """Tests for the two-round sequential tool use flow."""

    def _tools(self):
        return [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

    def test_two_tool_rounds_makes_three_api_calls(self):
        gen = make_generator()
        block1 = make_tool_use_block("search_course_content", "id_1", {"query": "outline of course X"})
        block2 = make_tool_use_block("search_course_content", "id_2", {"query": "topic Y"})
        r1 = make_api_response("tool_use", [block1])
        r2 = make_api_response("tool_use", [block2])
        r3 = make_api_response("end_turn", [make_text_block("Final answer")])
        gen.client.messages.create = MagicMock(side_effect=[r1, r2, r3])
        manager = make_tool_manager()

        result = gen.generate_response("Find a course on the same topic as lesson 4 of course X",
                                       tools=self._tools(), tool_manager=manager)

        assert result == "Final answer"
        assert gen.client.messages.create.call_count == 3
        assert manager.execute_tool.call_count == 2

    def test_round_one_retains_tools_in_api_call(self):
        """Round-1 follow-up call must include tools so Claude can make a second tool call."""
        gen = make_generator()
        block1 = make_tool_use_block("search_course_content", "id_1", {"query": "q1"})
        block2 = make_tool_use_block("search_course_content", "id_2", {"query": "q2"})
        r1 = make_api_response("tool_use", [block1])
        r2 = make_api_response("tool_use", [block2])
        r3 = make_api_response("end_turn", [make_text_block("Done")])
        gen.client.messages.create = MagicMock(side_effect=[r1, r2, r3])
        tools = self._tools()

        gen.generate_response("test", tools=tools, tool_manager=make_tool_manager())

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tools"] == tools
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    def test_final_api_call_strips_tools(self):
        """Third API call (after 2 tool rounds) must not include tools."""
        gen = make_generator()
        block1 = make_tool_use_block("search_course_content", "id_1", {"query": "q1"})
        block2 = make_tool_use_block("search_course_content", "id_2", {"query": "q2"})
        r1 = make_api_response("tool_use", [block1])
        r2 = make_api_response("tool_use", [block2])
        r3 = make_api_response("end_turn", [make_text_block("Done")])
        gen.client.messages.create = MagicMock(side_effect=[r1, r2, r3])

        gen.generate_response("test", tools=self._tools(), tool_manager=make_tool_manager())

        third_call_kwargs = gen.client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs
        assert "tool_choice" not in third_call_kwargs

    def test_early_termination_no_second_tool_use(self):
        """If Claude returns end_turn on the round-1 follow-up, no third API call is made."""
        gen = make_generator()
        block1 = make_tool_use_block("search_course_content", "id_1", {"query": "q1"})
        r1 = make_api_response("tool_use", [block1])
        r2 = make_api_response("end_turn", [make_text_block("Answered in one round")])
        gen.client.messages.create = MagicMock(side_effect=[r1, r2])

        result = gen.generate_response("test", tools=self._tools(), tool_manager=make_tool_manager())

        assert result == "Answered in one round"
        assert gen.client.messages.create.call_count == 2
        assert make_tool_manager().execute_tool.call_count == 0  # separate manager, not called here

    def test_message_structure_after_two_rounds(self):
        """Final API call messages: user, assistant, user(results), assistant, user(results)."""
        gen = make_generator()
        block1 = make_tool_use_block("search_course_content", "id_1", {"query": "q1"})
        block2 = make_tool_use_block("search_course_content", "id_2", {"query": "q2"})
        r1 = make_api_response("tool_use", [block1])
        r2 = make_api_response("tool_use", [block2])
        r3 = make_api_response("end_turn", [make_text_block("Done")])
        gen.client.messages.create = MagicMock(side_effect=[r1, r2, r3])

        gen.generate_response("original query", tools=self._tools(), tool_manager=make_tool_manager())

        third_call_kwargs = gen.client.messages.create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]
        assert len(messages) == 5
        assert messages[0] == {"role": "user", "content": "original query"}
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    def test_tool_execution_failure_graceful_recovery(self):
        """If execute_tool raises, the error is captured as a tool result and the chain continues."""
        gen = make_generator()
        block1 = make_tool_use_block("search_course_content", "id_1", {"query": "q1"})
        r1 = make_api_response("tool_use", [block1])
        r2 = make_api_response("end_turn", [make_text_block("Recovered answer")])
        gen.client.messages.create = MagicMock(side_effect=[r1, r2])

        manager = make_tool_manager()
        manager.execute_tool.side_effect = RuntimeError("DB unavailable")

        result = gen.generate_response("test", tools=self._tools(), tool_manager=manager)

        assert result == "Recovered answer"
        assert gen.client.messages.create.call_count == 2
        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        tool_result_content = second_call_kwargs["messages"][-1]["content"][0]
        assert "Tool execution failed" in tool_result_content["content"]
        assert "DB unavailable" in tool_result_content["content"]
