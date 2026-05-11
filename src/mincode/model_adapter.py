"""Model adapter for MiniMind — connects via its OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ModelError(RuntimeError):
    """Raised when model calls fail."""


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK message object to a plain dict."""
    content = message.content if hasattr(message, "content") else ""
    result: dict[str, Any] = {"role": "assistant", "content": content or ""}

    # MiniMind's API can return reasoning_content for <think> tags
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        result["reasoning_content"] = reasoning_content

    raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
    if raw_tool_calls:
        calls: list[dict[str, Any]] = []
        for item in raw_tool_calls:
            function = getattr(item, "function", None)
            calls.append({
                "id": str(getattr(item, "id", "")),
                "type": "function",
                "function": {
                    "name": str(getattr(function, "name", "")),
                    "arguments": str(getattr(function, "arguments", "{}")),
                },
            })
        result["tool_calls"] = calls

    return result


@dataclass
class MiniMindClient:
    """OpenAI-compatible client that talks to MiniMind's serve_openai_api.py.

    Usage:
        client = MiniMindClient(base_url="http://localhost:8998/v1")
        msg = client.complete(messages=[...], tools=[...])
    """

    base_url: str = "http://localhost:8998/v1"
    api_key: str = "minimind"  # MiniMind doesn't check keys, but OpenAI SDK requires one
    model: str = "minimind"
    timeout: int = 120
    temperature: float = 0.7
    # NOTE: MiniMind API uses max_tokens for BOTH prompt truncation ([-max_tokens:] on the
    # prompt string) AND generation limit (max_length = prompt_len + max_tokens). So this
    # value must be large enough to not truncate the prompt, but not so large that CPU
    # inference takes forever. 4096 is a good compromise.
    max_tokens: int = 4096
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelError("openai SDK is required: pip install openai") from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a completion request and return one assistant message dict."""
        from openai import APIConnectionError, APIError, APITimeoutError

        # Build request kwargs — stream=False is critical because MiniMind defaults to stream=True
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        # Only pass tools if we have them — MiniMind API accepts tools parameter
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(**kwargs)
        except APITimeoutError as exc:
            raise ModelError("MiniMind request timed out — is serve_openai_api.py running?") from exc
        except APIConnectionError as exc:
            raise ModelError(
                f"Cannot connect to MiniMind at {self.base_url} — "
                "start it with: cd ../minimind && python scripts/serve_openai_api.py"
            ) from exc
        except APIError as exc:
            raise ModelError(f"MiniMind API error: {exc}") from exc

        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError) as exc:
            raise ModelError(f"Unexpected response from MiniMind: {response}") from exc

        return _message_to_dict(message)
