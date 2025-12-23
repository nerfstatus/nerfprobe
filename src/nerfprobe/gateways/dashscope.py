"""Alibaba DashScope gateway for Qwen models."""

from collections.abc import AsyncIterator

import httpx
from nerfprobe_core import LogprobResult, ModelTarget


class DashScopeGateway:
    """
    Gateway for Alibaba DashScope (Qwen models).
    API: https://dashscope.aliyun.com/
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
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
        """Simple completion via DashScope."""
        client = await self._get_client()
        response = await client.post(
            "/services/aigc/text-generation/generation",
            json={
                "model": model.model_name,
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {},
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("output", {}).get("text", "")

    async def generate_stream(self, model: ModelTarget, prompt: str) -> AsyncIterator[str]:
        """Streaming completion."""
        client = await self._get_client()
        async with client.stream(
            "POST",
            "/services/aigc/text-generation/generation",
            json={
                "model": model.model_name,
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {"incremental_output": True},
            },
            headers={"X-DashScope-SSE": "enable"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    try:
                        import json

                        data = json.loads(line[5:])
                        text = data.get("output", {}).get("text", "")
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
        """Not supported by DashScope."""
        raise NotImplementedError("DashScope does not support logprobs")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
