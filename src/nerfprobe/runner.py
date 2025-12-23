"""Probe runner - orchestrates probe execution."""

from collections.abc import Sequence

from nerfprobe_core import LLMGateway, ModelTarget, ProbeResult
from nerfprobe_core.probes import (
    ADVANCED_PROBES,
    ALL_PROBES,
    CORE_PROBES,
    OPTIONAL_PROBES,
    PROBE_REGISTRY,
)
from nerfprobe_core.probes.config import (
    CalibrationProbeConfig,
    ChainOfThoughtProbeConfig,
    CodeProbeConfig,
    ConstraintProbeConfig,
    ContextProbeConfig,
    FingerprintProbeConfig,
    LogicPuzzleProbeConfig,
    MathProbeConfig,
    MultilingualProbeConfig,
    RepetitionProbeConfig,
    RoutingProbeConfig,
    StyleProbeConfig,
    TimingProbeConfig,
    ZeroPrintProbeConfig,
)

# Default probe configurations
DEFAULT_CONFIGS = {
    # Core tier
    "math": MathProbeConfig(
        name="math_probe",
        prompt="What is 15 * 12 + 8 * 9? Give only the final number.",
        expected_answer="252",
    ),
    "style": StyleProbeConfig(
        name="style_probe",
        topic="a robot who discovers ancient ruins",
    ),
    "timing": TimingProbeConfig(
        name="timing_probe",
        token_count=30,
    ),
    "code": CodeProbeConfig(
        name="code_probe",
    ),
    # Advanced tier
    "fingerprint": FingerprintProbeConfig(),
    "context": ContextProbeConfig(),
    "routing": RoutingProbeConfig(),
    "repetition": RepetitionProbeConfig(
        name="repetition_probe",
    ),
    "constraint": ConstraintProbeConfig(
        name="constraint_probe",
        min_words=80,
        max_words=120,
    ),
    "logic": LogicPuzzleProbeConfig(
        name="logic_probe",
    ),
    "cot": ChainOfThoughtProbeConfig(
        name="cot_probe",
    ),
    # Optional tier
    "calibration": CalibrationProbeConfig(
        name="calibration_probe",
    ),
    "zeroprint": ZeroPrintProbeConfig(
        name="zeroprint_probe",
    ),
    "multilingual": MultilingualProbeConfig(),
}


def get_probes_for_tier(tier: str) -> list[str]:
    """Get probe names for a given tier."""
    if tier == "core":
        return CORE_PROBES
    elif tier == "advanced":
        return CORE_PROBES + ADVANCED_PROBES
    elif tier == "optional":
        return CORE_PROBES + ADVANCED_PROBES + OPTIONAL_PROBES
    elif tier == "all":
        return ALL_PROBES
    else:
        raise ValueError(f"Unknown tier: {tier}. Use: core, advanced, optional, all")


async def run_probe(
    model_name: str,
    gateway: LLMGateway,
    probe_name: str,
    provider_id: str = "openai",
) -> ProbeResult:
    """Run a single probe."""
    if probe_name not in PROBE_REGISTRY:
        raise ValueError(f"Unknown probe: {probe_name}. Available: {list(PROBE_REGISTRY.keys())}")

    target = ModelTarget(
        provider_id=provider_id,
        model_name=model_name,
    )

    config = DEFAULT_CONFIGS.get(probe_name)
    if not config:
        raise ValueError(f"No default config for probe: {probe_name}")

    probe_class = PROBE_REGISTRY[probe_name]
    probe = probe_class(config)

    return await probe.run(target, gateway)


async def run_probes(
    model_name: str,
    gateway: LLMGateway,
    tier: str = "core",
    probes: Sequence[str] | None = None,
    provider_id: str = "openai",
) -> list[ProbeResult]:
    """
    Run multiple probes on a model.

    Args:
        model_name: The model to test (e.g., "gpt-4o")
        gateway: The LLM gateway to use
        tier: Probe tier ("core", "advanced", "optional", "all")
        probes: Specific probes to run (overrides tier)
        provider_id: Provider identifier for result tracking

    Returns:
        List of ProbeResult objects
    """
    target = ModelTarget(
        provider_id=provider_id,
        model_name=model_name,
    )

    # Determine which probes to run
    if probes:
        probe_names = list(probes)
    else:
        probe_names = get_probes_for_tier(tier)

    # Filter to only available probes
    available = [p for p in probe_names if p in PROBE_REGISTRY]

    results: list[ProbeResult] = []
    for probe_name in available:
        config = DEFAULT_CONFIGS.get(probe_name)
        if not config:
            continue

        probe_class = PROBE_REGISTRY[probe_name]
        probe = probe_class(config)

        try:
            result = await probe.run(target, gateway)
            results.append(result)
        except Exception as e:
            # Create a failed result for exceptions
            from nerfprobe_core import ProbeType

            results.append(
                ProbeResult(
                    probe_name=probe_name,
                    probe_type=ProbeType.MATH,  # Fallback type
                    target=target,
                    passed=False,
                    score=0.0,
                    latency_ms=0.0,
                    raw_response=f"ERROR: {e!s}",
                    metadata={"error": str(e)},
                )
            )

    return results
