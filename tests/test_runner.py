"""Tests for runner module."""

from unittest.mock import AsyncMock

import pytest

from nerfprobe.runner import (
    DEFAULT_CONFIGS,
    get_probes_for_tier,
    run_probe,
    run_probes,
)


@pytest.fixture
def mock_gateway():
    gateway = AsyncMock()
    gateway.generate = AsyncMock(return_value="42")
    gateway.generate_stream = AsyncMock()
    gateway.close = AsyncMock()
    return gateway


class TestGetProbesForTier:
    def test_core_tier(self):
        probes = get_probes_for_tier("core")
        assert "math" in probes
        assert "style" in probes
        assert "timing" in probes
        assert "code" in probes
        assert len(probes) == 4

    def test_advanced_tier(self):
        probes = get_probes_for_tier("advanced")
        assert "math" in probes  # Includes core
        assert "fingerprint" in probes
        assert "context" in probes
        assert len(probes) == 11  # 4 core + 7 advanced

    def test_all_tier(self):
        probes = get_probes_for_tier("all")
        assert len(probes) == 14

    def test_invalid_tier(self):
        with pytest.raises(ValueError):
            get_probes_for_tier("invalid")


class TestDefaultConfigs:
    def test_all_core_probes_have_configs(self):
        for probe_name in ["math", "style", "timing", "code"]:
            assert probe_name in DEFAULT_CONFIGS

    def test_all_advanced_probes_have_configs(self):
        for probe_name in ["fingerprint", "context", "routing", "repetition", "constraint", "logic", "cot"]:
            assert probe_name in DEFAULT_CONFIGS

    def test_all_optional_probes_have_configs(self):
        for probe_name in ["calibration", "zeroprint", "multilingual"]:
            assert probe_name in DEFAULT_CONFIGS


class TestRunProbe:
    @pytest.mark.asyncio
    async def test_run_math_probe(self, mock_gateway):
        mock_gateway.generate.return_value = "The answer is 252."
        result = await run_probe("gpt-4", mock_gateway, "math")
        assert result.probe_name == "math_probe"

    @pytest.mark.asyncio
    async def test_run_unknown_probe(self, mock_gateway):
        with pytest.raises(ValueError, match="Unknown probe"):
            await run_probe("gpt-4", mock_gateway, "nonexistent_probe")


class TestRunProbes:
    @pytest.mark.asyncio
    async def test_run_core_tier(self, mock_gateway):
        mock_gateway.generate.return_value = "Test response 252"

        # Mock streaming for timing probe
        async def mock_stream(*args):
            yield "token"

        mock_gateway.generate_stream = mock_stream

        results = await run_probes(
            model_name="test-model",
            gateway=mock_gateway,
            tier="core",
        )

        assert len(results) == 4
        probe_names = [r.probe_name for r in results]
        assert "math_probe" in probe_names

    @pytest.mark.asyncio
    async def test_run_specific_probes(self, mock_gateway):
        mock_gateway.generate.return_value = "The answer is 252."

        results = await run_probes(
            model_name="test-model",
            gateway=mock_gateway,
            probes=["math"],
        )

        assert len(results) == 1
        assert results[0].probe_name == "math_probe"

    @pytest.mark.asyncio
    async def test_handle_probe_error(self, mock_gateway):
        mock_gateway.generate.side_effect = Exception("API Error")

        results = await run_probes(
            model_name="test-model",
            gateway=mock_gateway,
            probes=["math"],
        )

        assert len(results) == 1
        assert results[0].passed is False
