"""Ollama gateway for local models."""

from collections.abc import AsyncIterator

import httpx
from nerfprobe_core import LogprobResult, ModelTarget


class OllamaGateway:
    """
    Gateway for Ollama local models.
    Default: http://localhost:11434
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        return self._client

    async def generate(self, model: ModelTarget, prompt: str) -> str:
        """Simple completion via Ollama generate API."""
        client = await self._get_client()
        response = await client.post(
            "/api/generate",
            json={
                "model": model.model_name,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def generate_stream(self, model: ModelTarget, prompt: str) -> AsyncIterator[str]:
        """Streaming completion for timing analysis."""
        client = await self._get_client()
        async with client.stream(
            "POST",
            "/api/generate",
            json={
                "model": model.model_name,
                "prompt": prompt,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        import json

                        data = json.loads(line)
                        text = data.get("response", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue

    async def generate_with_logprobs(
        self,
        model: ModelTarget,
        prompt: str,
        top_logprobs: int = 5,
    ) -> LogprobResult:
        """Not supported by Ollama."""
        raise NotImplementedError("Ollama does not support logprobs")

    async def list_models(self) -> list[str]:
        """List available local models."""
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
