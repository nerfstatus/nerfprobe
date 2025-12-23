"""Zhipu AI gateway for GLM models."""

from collections.abc import AsyncIterator

import httpx
from nerfprobe_core import LogprobResult, ModelTarget


class ZhipuGateway:
    """
    Gateway for Zhipu AI (GLM-4, ChatGLM).
    API: https://open.bigmodel.cn/
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def generate(self, model: ModelTarget, prompt: str) -> str:
        """Simple completion via Zhipu."""
        client = await self._get_client()
        response = await client.post(
            "/chat/completions",
            json={
                "model": model.model_name,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_stream(self, model: ModelTarget, prompt: str) -> AsyncIterator[str]:
        """Streaming completion."""
        client = await self._get_client()
        async with client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": model.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        import json

                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def generate_with_logprobs(
        self,
        model: ModelTarget,
        prompt: str,
        top_logprobs: int = 5,
    ) -> LogprobResult:
        """Not supported by Zhipu."""
        raise NotImplementedError("Zhipu does not support logprobs")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
