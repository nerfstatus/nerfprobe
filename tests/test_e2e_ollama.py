"""End-to-end tests with Ollama (requires local Ollama server)."""

import pytest
import os

# Skip all tests if Ollama is not available
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_E2E", "1") == "1",
    reason="E2E tests disabled. Set SKIP_E2E=0 to run."
)

from nerfprobe.gateways import OllamaGateway
from nerfprobe.runner import run_probes
from nerfprobe_core import ModelTarget


@pytest.fixture
def ollama_gateway():
    """Create a synchronous Ollama gateway fixture."""
    gateway = OllamaGateway()
    yield gateway
    # Cleanup handled by test


class TestOllamaE2E:
    """End-to-end tests with real Ollama models."""

    @pytest.mark.asyncio
    async def test_list_models(self, ollama_gateway):
        """Verify Ollama is running and has models."""
        models = await ollama_gateway.list_models()
        assert len(models) > 0
        print(f"Available models: {models}")

    @pytest.mark.asyncio
    async def test_simple_generation(self, ollama_gateway):
        """Test basic generation."""
        target = ModelTarget(provider_id="ollama", model_name="qwen3:4b")
        response = await ollama_gateway.generate(target, "What is 2 + 2? Answer with just the number.")
        assert "4" in response

    @pytest.mark.asyncio
    async def test_streaming_generation(self, ollama_gateway):
        """Test streaming for timing analysis."""
        target = ModelTarget(provider_id="ollama", model_name="gemma2:2b")
        tokens = []
        async for token in ollama_gateway.generate_stream(target, "Count from 1 to 5"):
            tokens.append(token)
        
        full_response = "".join(tokens)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_run_math_probe(self, ollama_gateway):
        """Test MathProbe with real model."""
        results = await run_probes(
            model_name="qwen3:4b",
            gateway=ollama_gateway,
            probes=["math"],
            provider_id="ollama",
        )
        
        assert len(results) == 1
        result = results[0]
        print(f"Math probe: passed={result.passed}, score={result.score}")
        print(f"Response: {result.raw_response[:200]}...")

    @pytest.mark.asyncio
    async def test_run_style_probe(self, ollama_gateway):
        """Test StyleProbe with real model."""
        results = await run_probes(
            model_name="qwen3:4b",
            gateway=ollama_gateway,
            probes=["style"],
            provider_id="ollama",
        )
        
        assert len(results) == 1
        result = results[0]
        print(f"Style probe: passed={result.passed}, score={result.score}")
        print(f"TTR: {result.metric_scores.get('ttr', 'N/A')}")

    @pytest.mark.asyncio
    async def test_run_code_probe(self, ollama_gateway):
        """Test CodeProbe with real model."""
        results = await run_probes(
            model_name="qwen3:4b",
            gateway=ollama_gateway,
            probes=["code"],
            provider_id="ollama",
        )
        
        assert len(results) == 1
        result = results[0]
        print(f"Code probe: passed={result.passed}, score={result.score}")

    @pytest.mark.asyncio
    async def test_run_core_tier(self, ollama_gateway):
        """Run all core probes."""
        results = await run_probes(
            model_name="gemma2:2b",
            gateway=ollama_gateway,
            tier="core",
            provider_id="ollama",
        )
        
        assert len(results) == 4
        for result in results:
            print(f"{result.probe_name}: passed={result.passed}, score={result.score:.2f}, latency={result.latency_ms:.0f}ms")

    @pytest.mark.asyncio
    async def test_compare_models(self, ollama_gateway):
        """Compare two models on same probe."""
        models = ["gemma2:2b", "qwen3:4b"]
        
        for model_name in models:
            results = await run_probes(
                model_name=model_name,
                gateway=ollama_gateway,
                probes=["math"],
                provider_id="ollama",
            )
            result = results[0]
            print(f"{model_name}: passed={result.passed}, score={result.score:.2f}, latency={result.latency_ms:.0f}ms")
