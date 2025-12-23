"""Google Gemini gateway."""

from collections.abc import AsyncIterator

import httpx
from nerfprobe_core import LogprobResult, ModelTarget


class GoogleGateway:
    """
    Gateway for Google Gemini models.
    Uses the Generative Language API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com"
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
        """Simple completion."""
        client = await self._get_client()
        response = await client.post(
            f"/v1beta/models/{model.model_name}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
            },
        )
        response.raise_for_status()
        data = response.json()
        # Extract text from candidates
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            text_parts = [part["text"] for part in parts if "text" in part]
            return "".join(text_parts)
        return ""

    async def generate_stream(self, model: ModelTarget, prompt: str) -> AsyncIterator[str]:
        """Streaming completion for timing analysis."""
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"/v1beta/models/{model.model_name}:streamGenerateContent",
            params={"key": self.api_key, "alt": "sse"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        import json

                        data = json.loads(data_str)
                        candidates = data.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    yield part["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def generate_with_logprobs(
        self,
        model: ModelTarget,
        prompt: str,
        top_logprobs: int = 5,
    ) -> LogprobResult:
        """Not supported by Gemini API."""
        raise NotImplementedError("Gemini API does not support logprobs")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
