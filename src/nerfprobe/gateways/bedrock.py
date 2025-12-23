"""AWS Bedrock gateway for Claude and other models."""

from typing import AsyncIterator
import json
import httpx

from nerfprobe_core import ModelTarget, LogprobResult


class BedrockGateway:
    """
    Gateway for AWS Bedrock.
    Supports Anthropic Claude, Meta Llama, Amazon Titan, etc.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        timeout: float = 60.0,
    ):
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

        # Import boto3 for AWS signing if available
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            self._boto_available = True
        except ImportError:
            self._boto_available = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_endpoint(self, model_id: str) -> str:
        """Get Bedrock endpoint for model."""
        return f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{model_id}/invoke"

    async def generate(self, model: ModelTarget, prompt: str) -> str:
        """Simple completion via Bedrock."""
        if not self._boto_available:
            raise ImportError("boto3 required for Bedrock gateway. pip install boto3")

        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import Credentials

        client = await self._get_client()
        endpoint = self._get_endpoint(model.model_name)

        # Build request body based on model type
        if "anthropic" in model.model_name.lower():
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            # Generic format for other models
            body = {
                "inputText": prompt,
                "textGenerationConfig": {"maxTokenCount": 4096},
            }

        body_json = json.dumps(body)

        # Sign request with AWS SigV4
        credentials = Credentials(
            self.access_key or "",
            self.secret_key or "",
            self.session_token,
        )
        request = AWSRequest(
            method="POST",
            url=endpoint,
            data=body_json,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(credentials, "bedrock", self.region).add_auth(request)

        response = await client.post(
            endpoint,
            content=body_json,
            headers=dict(request.headers),
        )
        response.raise_for_status()
        data = response.json()

        # Extract text based on model type
        if "anthropic" in model.model_name.lower():
            content = data.get("content", [])
            return "".join(b.get("text", "") for b in content if b.get("type") == "text")
        else:
            return data.get("results", [{}])[0].get("outputText", "")

    async def generate_stream(
        self, model: ModelTarget, prompt: str
    ) -> AsyncIterator[str]:
        """Streaming not fully implemented for Bedrock."""
        # Bedrock streaming is complex, return non-streaming for now
        result = await self.generate(model, prompt)
        yield result

    async def generate_with_logprobs(
        self,
        model: ModelTarget,
        prompt: str,
        top_logprobs: int = 5,
    ) -> LogprobResult:
        """Not supported by most Bedrock models."""
        raise NotImplementedError("Bedrock does not support logprobs")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
