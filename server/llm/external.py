import httpx
import json
import logging
from collections.abc import AsyncGenerator

from server.config import settings
from server.models.schemas import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

# Provider configs
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.1-8b-instant",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "default_model": "gemini-2.0-flash",
    },
}


class ExternalLLMClient:
    """Client for external LLM APIs (Groq, Google AI Studio) using OpenAI-compatible format."""

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.provider = provider or settings.external_llm_provider
        self.api_key = api_key or settings.external_llm_api_key
        self.model = model or settings.external_llm_model

        provider_config = PROVIDERS.get(self.provider, PROVIDERS["groq"])
        self.base_url = provider_config["base_url"]
        if not self.model:
            self.model = provider_config["default_model"]

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """Send a chat request and return the full response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.base_url,
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Send a chat request with tool definitions, return structured response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.base_url,
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        message = data["choices"][0]["message"]
        content = message.get("content")

        # Parse tool calls if present
        raw_tool_calls = message.get("tool_calls")
        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        arguments=args,
                    )
                )

        return LLMResponse(content=content, tool_calls=tool_calls)

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream the response token by token."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                self.base_url,
                json=payload,
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token

    async def chat_stream_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
        tool_choice: str = "auto",
    ) -> LLMResponse | AsyncGenerator[str, None]:
        """Chat with tools support.

        First makes a non-streaming request to check for tool calls.
        If the LLM returns tool calls, returns an LLMResponse.
        If the LLM returns text, re-requests with streaming and returns a generator.
        """
        # First check: does the LLM want to call tools?
        response = await self.chat_with_tools(
            messages, tools, temperature, tool_choice
        )

        if response.tool_calls:
            return response

        # No tool calls — stream the text response for better UX
        # Re-request without tools for a clean stream
        async def _stream():
            async for token in self.chat_stream(messages, temperature):
                yield token

        return _stream()

    async def is_available(self) -> bool:
        """Check if the external API key is configured."""
        return bool(self.api_key)


# Singleton
external_client = ExternalLLMClient()

