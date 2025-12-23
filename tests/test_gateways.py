"""Tests for gateways with mocked HTTP responses."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from nerfprobe.gateways import (
    OpenAIGateway,
    AnthropicGateway,
    GoogleGateway,
    OllamaGateway,
)
from nerfprobe_core import ModelTarget


@pytest.fixture
def target():
    return ModelTarget(provider_id="test", model_name="test-model")


# =============================================================================
# OpenAIGateway Tests
# =============================================================================

class TestOpenAIGateway:
    @pytest.mark.asyncio
    async def test_generate(self, target):
        gateway = OpenAIGateway(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(gateway, "_get_client") as mock_client:
            client = AsyncMock()
            client.post.return_value = mock_response
            mock_client.return_value = client
            
            result = await gateway.generate(target, "Say hello")
            assert result == "Hello world"
        
        await gateway.close()

    @pytest.mark.asyncio
    async def test_generate_with_logprobs(self, target):
        gateway = OpenAIGateway(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello"},
                "logprobs": {
                    "content": [
                        {"token": "Hello", "logprob": -0.5, "top_logprobs": []}
                    ]
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(gateway, "_get_client") as mock_client:
            client = AsyncMock()
            client.post.return_value = mock_response
            mock_client.return_value = client
            
            result = await gateway.generate_with_logprobs(target, "test")
            assert result.text == "Hello"
            assert len(result.tokens) == 1
        
        await gateway.close()


# =============================================================================
# AnthropicGateway Tests
# =============================================================================

class TestAnthropicGateway:
    @pytest.mark.asyncio
    async def test_generate(self, target):
        gateway = AnthropicGateway(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello from Claude"}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(gateway, "_get_client") as mock_client:
            client = AsyncMock()
            client.post.return_value = mock_response
            mock_client.return_value = client
            
            result = await gateway.generate(target, "Say hello")
            assert result == "Hello from Claude"
        
        await gateway.close()

    @pytest.mark.asyncio
    async def test_logprobs_not_supported(self, target):
        gateway = AnthropicGateway(api_key="test-key")
        
        with pytest.raises(NotImplementedError):
            await gateway.generate_with_logprobs(target, "test")
        
        await gateway.close()


# =============================================================================
# GoogleGateway Tests
# =============================================================================

class TestGoogleGateway:
    @pytest.mark.asyncio
    async def test_generate(self, target):
        gateway = GoogleGateway(api_key="test-key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello from Gemini"}]}
            }]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(gateway, "_get_client") as mock_client:
            client = AsyncMock()
            client.post.return_value = mock_response
            mock_client.return_value = client
            
            result = await gateway.generate(target, "Say hello")
            assert result == "Hello from Gemini"
        
        await gateway.close()


# =============================================================================
# OllamaGateway Tests
# =============================================================================

class TestOllamaGateway:
    @pytest.mark.asyncio
    async def test_generate(self, target):
        gateway = OllamaGateway()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello from Ollama"}
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(gateway, "_get_client") as mock_client:
            client = AsyncMock()
            client.post.return_value = mock_response
            mock_client.return_value = client
            
            result = await gateway.generate(target, "Say hello")
            assert result == "Hello from Ollama"
        
        await gateway.close()

    @pytest.mark.asyncio
    async def test_list_models(self):
        gateway = OllamaGateway()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "mistral:7b"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(gateway, "_get_client") as mock_client:
            client = AsyncMock()
            client.get.return_value = mock_response
            mock_client.return_value = client
            
            models = await gateway.list_models()
            assert "llama3.2:3b" in models
            assert "mistral:7b" in models
        
        await gateway.close()
