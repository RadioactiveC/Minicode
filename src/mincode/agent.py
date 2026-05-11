"""Core agent loop: user input -> LLM -> tool calls -> loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class ChatModel(Protocol):
    """Any model adapter must implement this single method."""

    def complete(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return one assistant message dict (OpenAI format)."""


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


@dataclass
class AgentCallbacks:
    """Display hooks. The agent stays framework-agnostic."""

    on_status: Callable[[str], None] | None = None
    on_thinking: Callable[[str], None] | None = None
    on_tool_call: Callable[[str, str], None] | None = None
    on_tool_result: Callable[[str, str], None] | None = None


class Agent:
    """A tool-calling agent loop over Chat Completions."""

    def __init__(
        self,
        *,
        model: ChatModel,
        tools: "ToolRegistry",  # noqa: F821 — forward ref
        system_prompt: str,
        max_steps: int = 20,
        callbacks: AgentCallbacks | None = None,
    ) -> None:
        from mincode.tools import ToolRegistry  # avoid circular at import time

        self.model = model
        self.tools: ToolRegistry = tools
        self.max_steps = max_steps
        self.cb = callbacks or AgentCallbacks()
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def _fire(self, name: str, *args: Any) -> None:
        fn = getattr(self.cb, name, None)
        if fn is not None:
            fn(*args)

    def run(self, user_input: str) -> str:
        """Run one user turn to completion, returning the final text reply."""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_steps):
            self._fire("on_status", "Thinking...")
            assistant = self.model.complete(
                messages=self.messages,
                tools=self.tools.as_openai_tools(),
            )
            self._fire("on_status", "")

            # Show thinking if present (MiniMind supports <think> tags)
            reasoning = assistant.get("reasoning_content")
            if reasoning:
                self._fire("on_thinking", str(reasoning))

            tool_calls = assistant.get("tool_calls") or []

            # Build the assistant record to append to history
            record: dict[str, Any] = {
                "role": "assistant",
                "content": assistant.get("content"),
            }
            if assistant.get("reasoning_content"):
                record["reasoning_content"] = assistant["reasoning_content"]
            if tool_calls:
                record["tool_calls"] = tool_calls
            self.messages.append(record)

            # No tool calls → turn is done
            if not tool_calls:
                return _coerce_text(assistant.get("content"))

            # Execute each tool call
            for call in tool_calls:
                call_id = str(call.get("id", ""))
                function = call.get("function") or {}
                name = str(function.get("name", ""))
                arguments = str(function.get("arguments", "{}"))

                self._fire("on_tool_call", name, arguments)
                result = self.tools.execute(name, arguments)
                self._fire("on_tool_result", name, result)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })

        raise RuntimeError(f"max steps exceeded ({self.max_steps})")
