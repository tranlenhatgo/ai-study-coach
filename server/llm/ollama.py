import httpx
import json
import logging
from collections.abc import AsyncGenerator

from server.config import settings
from server.models.schemas import LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for locally-hosted Ollama LLM (Phi-3-mini)."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = settings.ollama_timeout

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """Send a chat request and return the full response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Send a chat request with tool definitions.

        Ollama supports tool calling for compatible models. If the model
        doesn't support tools, falls back to a regular chat call.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
            "tools": tools,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()

            message = data.get("message", {})
            content = message.get("content")

            # Parse tool calls if present
            raw_tool_calls = message.get("tool_calls")
            tool_calls = None
            if raw_tool_calls:
                tool_calls = []
                for tc in raw_tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", f"call_{len(tool_calls)}"),
                            name=func.get("name", ""),
                            arguments=args,
                        )
                    )

            return LLMResponse(content=content, tool_calls=tool_calls)

        except Exception as e:
            # Fallback: model may not support tools — use regular chat
            logger.warning(f"Ollama tool calling failed, falling back to regular chat: {e}")
            text = await self.chat(messages, temperature)
            return LLMResponse(content=text, tool_calls=None)

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Send a chat request and stream the response token by token."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break

    async def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


# Singleton
ollama_client = OllamaClient()

