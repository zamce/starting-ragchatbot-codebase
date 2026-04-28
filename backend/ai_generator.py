import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may use up to 2 sequential tool calls per query
- Use a second tool call only when the first result is insufficient or a different tool is needed to fully answer the question
- After tool calls, synthesize all results into a single final answer
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Outline Tool Usage:
- Use the outline tool for any question about a course outline, structure, syllabus, or lesson list
- You may use up to 2 sequential tool calls per query if the first call's result requires a follow-up tool call
- Return the course title, course link, and a numbered list of every lesson (lesson number and lesson title)
- Do not summarize or omit lessons — include the complete list

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Course outline questions**: Use the outline tool, then present the full lesson list
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


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
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls per query.
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        rounds_completed = 0
        while (
            response.stop_reason == "tool_use"
            and tool_manager is not None
            and rounds_completed < self.MAX_TOOL_ROUNDS
        ):
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = f"Tool execution failed: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            rounds_completed += 1

            next_params = {**self.base_params, "messages": messages, "system": system_content}
            if rounds_completed < self.MAX_TOOL_ROUNDS and tools:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**next_params)

        return response.content[0].text