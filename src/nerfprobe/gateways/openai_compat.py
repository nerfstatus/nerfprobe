"""OpenAI-compatible gateway for OpenAI, OpenRouter, vLLM, Ollama, etc."""

from typing import AsyncIterator
import httpx

from nerfprobe_core import ModelTarget, LogprobResult, LogprobToken


class OpenAIGateway:
    """
    Gateway for OpenAI-compatible endpoints.
    Works with: OpenAI, OpenRouter, vLLM, Ollama, Together, Fireworks, etc.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
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
        """Simple completion."""
        client = await self._get_client()
        response = await client.post(
            "/chat/completions",
            json={
                "model": model.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_stream(
        self, model: ModelTarget, prompt: str
    ) -> AsyncIterator[str]:
        """Streaming completion for timing analysis."""
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
        """Completion with token-level log probabilities."""
        client = await self._get_client()
        response = await client.post(
            "/chat/completions",
            json={
                "model": model.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "logprobs": True,
                "top_logprobs": top_logprobs,
            },
        )
        response.raise_for_status()
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        logprobs_data = data["choices"][0].get("logprobs", {}).get("content", [])

        tokens = []
        for item in logprobs_data:
            top_probs = {}
            for alt in item.get("top_logprobs", []):
                top_probs[alt["token"]] = alt["logprob"]
            tokens.append(
                LogprobToken(
                    token=item["token"],
                    logprob=item["logprob"],
                    top_logprobs=top_probs if top_probs else None,
                )
            )

        return LogprobResult(text=text, tokens=tokens)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
