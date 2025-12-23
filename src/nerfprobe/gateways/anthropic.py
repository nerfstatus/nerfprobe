"""Anthropic Claude gateway."""

from collections.abc import AsyncIterator

import httpx
from nerfprobe_core import LogprobResult, ModelTarget


class AnthropicGateway:
    """
    Gateway for Anthropic Claude models.
    Uses the Messages API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com"
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            if self.api_key:
                headers["x-api-key"] = self.api_key
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def generate(self, model: ModelTarget, prompt: str) -> str:
        """Simple completion."""
        client = await self._get_client()
        response = await client.post(
            "/v1/messages",
            json={
                "model": model.model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        # Extract text from content blocks
        content = data.get("content", [])
        text_parts = [block["text"] for block in content if block["type"] == "text"]
        return "".join(text_parts)

    async def generate_stream(self, model: ModelTarget, prompt: str) -> AsyncIterator[str]:
        """Streaming completion for timing analysis."""
        client = await self._get_client()
        async with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model.model_name,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        import json

                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def generate_with_logprobs(
        self,
        model: ModelTarget,
        prompt: str,
        top_logprobs: int = 5,
    ) -> LogprobResult:
        """Not supported by Anthropic API."""
        raise NotImplementedError("Anthropic API does not support logprobs")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
