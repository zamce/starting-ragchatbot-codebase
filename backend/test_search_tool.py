"""
Suite 1: CourseSearchTool.execute() output tests.
Verifies the tool produces correct output formats for all input combinations.
"""
import pytest
from unittest.mock import MagicMock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


def make_results(docs, metas):
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


def make_store(results=None, error=None):
    store = MagicMock()
    if error:
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[], error=error)
    else:
        store.search.return_value = results or make_results(
            ["Lesson content about Python functions."],
            [{"course_title": "Intro to Python", "lesson_number": 2}]
        )
    store.get_lesson_link.return_value = "https://example.com/lesson/2"
    return store


class TestCourseSearchToolExecute:
    def test_basic_query_returns_formatted_string(self):
        tool = CourseSearchTool(make_store())
        result = tool.execute(query="Python functions")
        assert "Intro to Python" in result
        assert "Lesson 2" in result
        assert "Lesson content about Python functions." in result

    def test_result_includes_course_and_lesson_header(self):
        tool = CourseSearchTool(make_store())
        result = tool.execute(query="test")
        assert result.startswith("[Intro to Python - Lesson 2]")

    def test_query_with_course_name_passes_filter(self):
        store = make_store()
        tool = CourseSearchTool(store)
        tool.execute(query="functions", course_name="Python")
        store.search.assert_called_once_with(query="functions", course_name="Python", lesson_number=None)

    def test_query_with_lesson_number_passes_filter(self):
        store = make_store()
        tool = CourseSearchTool(store)
        tool.execute(query="functions", lesson_number=3)
        store.search.assert_called_once_with(query="functions", course_name=None, lesson_number=3)

    def test_all_filters_passed(self):
        store = make_store()
        tool = CourseSearchTool(store)
        tool.execute(query="closures", course_name="Python", lesson_number=5)
        store.search.assert_called_once_with(query="closures", course_name="Python", lesson_number=5)

    def test_no_results_returns_message(self):
        store = make_store(results=SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)
        result = tool.execute(query="unknown topic")
        assert "No relevant content found" in result

    def test_no_results_with_course_filter_mentions_course(self):
        store = make_store(results=SearchResults(documents=[], metadata=[], distances=[]))
        tool = CourseSearchTool(store)
        result = tool.execute(query="unknown", course_name="MCP Course")
        assert "MCP Course" in result

    def test_search_error_returns_error_message(self):
        tool = CourseSearchTool(make_store(error="Search error: collection is empty"))
        result = tool.execute(query="anything")
        assert "Search error" in result

    def test_sources_tracked_after_search(self):
        tool = CourseSearchTool(make_store())
        tool.execute(query="functions")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "Intro to Python - Lesson 2"

    def test_sources_include_lesson_link(self):
        tool = CourseSearchTool(make_store())
        tool.execute(query="functions")
        assert tool.last_sources[0]["url"] == "https://example.com/lesson/2"

    def test_multiple_results_formatted_with_separator(self):
        results = make_results(
            ["Content A", "Content B"],
            [
                {"course_title": "Course 1", "lesson_number": 1},
                {"course_title": "Course 1", "lesson_number": 2},
            ]
        )
        store = make_store(results=results)
        store.get_lesson_link.return_value = None
        tool = CourseSearchTool(store)
        result = tool.execute(query="test")
        assert "Content A" in result
        assert "Content B" in result
        assert result.count("[Course 1") == 2

    def test_result_without_lesson_number_omits_lesson(self):
        results = make_results(
            ["General course info."],
            [{"course_title": "General Course"}]
        )
        store = make_store(results=results)
        tool = CourseSearchTool(store)
        result = tool.execute(query="info")
        assert "[General Course]" in result
        assert "Lesson" not in result


class TestToolManager:
    def test_execute_unknown_tool_returns_error_string(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_get_last_sources_returns_empty_when_no_search(self):
        manager = ToolManager()
        tool = CourseSearchTool(make_store())
        manager.register_tool(tool)
        assert manager.get_last_sources() == []

    def test_get_last_sources_after_search(self):
        manager = ToolManager()
        tool = CourseSearchTool(make_store())
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources_clears_last_sources(self):
        manager = ToolManager()
        tool = CourseSearchTool(make_store())
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()
        assert manager.get_last_sources() == []
