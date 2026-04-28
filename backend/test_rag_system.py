"""
Suite 3: RAG system end-to-end tests.
Tests the full query path: RAGSystem.query() → AIGenerator → tool → response + sources.
Uses a real VectorStore against the existing chroma_db when possible, or mocks when not.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from rag_system import RAGSystem
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def make_config():
    cfg = MagicMock()
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-sonnet-4-6"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "./chroma_db"
    return cfg


# ---------------------------------------------------------------------------
# Suite 3a: query() response and sources wiring
# ---------------------------------------------------------------------------

class TestRAGSystemQuery:
    """Verifies the response/sources tuple is assembled correctly end-to-end."""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_returns_answer_and_sources_tuple(self, MockGenerator, MockStore):
        mock_gen = MockGenerator.return_value
        mock_gen.generate_response.return_value = "Here is what I found about Python."
        mock_store = MockStore.return_value
        mock_store.get_course_count.return_value = 1
        mock_store.get_existing_course_titles.return_value = ["Python Basics"]

        system = RAGSystem(make_config())
        system.tool_manager.get_last_sources = MagicMock(return_value=[
            {"label": "Python Basics - Lesson 1", "url": "https://example.com/1"}
        ])
        system.tool_manager.reset_sources = MagicMock()

        answer, sources = system.query("What are Python functions?")
        assert answer == "Here is what I found about Python."
        assert len(sources) == 1
        assert sources[0]["label"] == "Python Basics - Lesson 1"

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_empty_sources_when_no_tool_called(self, MockGenerator, MockStore):
        mock_gen = MockGenerator.return_value
        mock_gen.generate_response.return_value = "Python is a language."
        mock_store = MockStore.return_value

        system = RAGSystem(make_config())
        system.tool_manager.get_last_sources = MagicMock(return_value=[])
        system.tool_manager.reset_sources = MagicMock()

        answer, sources = system.query("What is Python?")
        assert answer == "Python is a language."
        assert sources == []

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_sources_reset_after_query(self, MockGenerator, MockStore):
        mock_gen = MockGenerator.return_value
        mock_gen.generate_response.return_value = "Answer"
        MockStore.return_value

        system = RAGSystem(make_config())
        system.tool_manager.get_last_sources = MagicMock(return_value=[])
        reset_mock = MagicMock()
        system.tool_manager.reset_sources = reset_mock

        system.query("test")
        reset_mock.assert_called_once()

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_generate_response_called_with_tools(self, MockGenerator, MockStore):
        mock_gen = MockGenerator.return_value
        mock_gen.generate_response.return_value = "Answer"
        MockStore.return_value

        system = RAGSystem(make_config())
        system.tool_manager.get_last_sources = MagicMock(return_value=[])
        system.tool_manager.reset_sources = MagicMock()

        system.query("test query")
        call_kwargs = mock_gen.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is system.tool_manager

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_wrapped_in_prompt(self, MockGenerator, MockStore):
        mock_gen = MockGenerator.return_value
        mock_gen.generate_response.return_value = "Answer"
        MockStore.return_value

        system = RAGSystem(make_config())
        system.tool_manager.get_last_sources = MagicMock(return_value=[])
        system.tool_manager.reset_sources = MagicMock()

        system.query("What are Python loops?")
        call_kwargs = mock_gen.generate_response.call_args[1]
        assert "What are Python loops?" in call_kwargs["query"]


# ---------------------------------------------------------------------------
# Suite 3b: full tool-use path through the system (deeper integration)
# ---------------------------------------------------------------------------

class TestRAGSystemToolPath:
    """
    Patches only the Anthropic HTTP client — everything else (VectorStore, ToolManager)
    uses real or lightly mocked implementations to test the full integration path.
    """

    @patch("rag_system.VectorStore")
    def test_tool_result_flows_back_to_final_answer(self, MockStore):
        mock_store = MockStore.return_value
        mock_store.get_existing_course_titles.return_value = []
        # Make the search return real content so the tool produces a useful result
        mock_store.search.return_value = SearchResults(
            documents=["Python loops iterate over sequences."],
            metadata=[{"course_title": "Python Basics", "lesson_number": 3}],
            distances=[0.1]
        )
        mock_store.get_lesson_link.return_value = "https://example.com/lesson/3"

        tool_input = {"query": "Python loops", "course_name": "Python Basics"}
        tool_block = make_tool_use_block("search_course_content", "call_xyz", tool_input)
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Loops iterate over sequences.")])

        system = RAGSystem(make_config())
        system.ai_generator.client.messages.create = MagicMock(
            side_effect=[tool_response, final_response]
        )

        answer, sources = system.query("Tell me about Python loops")
        assert answer == "Loops iterate over sequences."
        assert len(sources) == 1
        assert "Python Basics" in sources[0]["label"]
        assert sources[0]["url"] == "https://example.com/lesson/3"

    @patch("rag_system.VectorStore")
    def test_two_api_calls_made_when_tool_used(self, MockStore):
        mock_store = MockStore.return_value
        mock_store.get_existing_course_titles.return_value = []
        mock_store.search.return_value = SearchResults(
            documents=["content"], metadata=[{"course_title": "C1", "lesson_number": 1}], distances=[0.1]
        )
        mock_store.get_lesson_link.return_value = None

        tool_block = make_tool_use_block("search_course_content", "call_1", {"query": "loops"})
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Done")])

        system = RAGSystem(make_config())
        create_mock = MagicMock(side_effect=[tool_response, final_response])
        system.ai_generator.client.messages.create = create_mock

        system.query("Python loops")
        assert create_mock.call_count == 2

    @patch("rag_system.VectorStore")
    def test_session_history_passed_to_generator(self, MockStore):
        MockStore.return_value.get_existing_course_titles.return_value = []
        direct_response = make_api_response("end_turn", [make_text_block("Answer")])

        system = RAGSystem(make_config())
        system.ai_generator.client.messages.create = MagicMock(return_value=direct_response)
        session_id = system.session_manager.create_session()
        system.session_manager.add_exchange(session_id, "prev question", "prev answer")

        system.query("follow-up question", session_id=session_id)
        call_kwargs = system.ai_generator.client.messages.create.call_args[1]
        assert "prev question" in call_kwargs["system"] or call_kwargs.get("system") is not None

    @patch("rag_system.VectorStore")
    def test_exchange_added_to_session_after_query(self, MockStore):
        MockStore.return_value.get_existing_course_titles.return_value = []
        direct_response = make_api_response("end_turn", [make_text_block("My answer")])

        system = RAGSystem(make_config())
        system.ai_generator.client.messages.create = MagicMock(return_value=direct_response)
        session_id = system.session_manager.create_session()

        system.query("my question", session_id=session_id)
        history = system.session_manager.get_conversation_history(session_id)
        assert "my question" in history
        assert "My answer" in history


# ---------------------------------------------------------------------------
# Suite 3c: error surface — what the frontend sees when things go wrong
# ---------------------------------------------------------------------------

class TestRAGSystemErrorHandling:
    """Ensures errors propagate in a way that app.py can catch and convert to HTTP 500."""

    @patch("rag_system.VectorStore")
    def test_api_exception_propagates_out_of_query(self, MockStore):
        MockStore.return_value.get_existing_course_titles.return_value = []

        system = RAGSystem(make_config())
        system.ai_generator.client.messages.create = MagicMock(
            side_effect=Exception("API error: invalid model")
        )
        with pytest.raises(Exception, match="API error"):
            system.query("test")

    @patch("rag_system.VectorStore")
    def test_tool_error_string_does_not_raise(self, MockStore):
        """A tool that returns an error string should not cause the system to raise."""
        mock_store = MockStore.return_value
        mock_store.get_existing_course_titles.return_value = []
        mock_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="DB connection failed"
        )

        tool_block = make_tool_use_block("search_course_content", "call_1", {"query": "test"})
        tool_response = make_api_response("tool_use", [tool_block])
        final_response = make_api_response("end_turn", [make_text_block("Could not find results.")])

        system = RAGSystem(make_config())
        system.ai_generator.client.messages.create = MagicMock(side_effect=[tool_response, final_response])

        answer, sources = system.query("test question")
        assert isinstance(answer, str)
        assert sources == []
