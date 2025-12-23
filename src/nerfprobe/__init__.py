"""NerfProbe - Scientifically-grounded LLM degradation detection."""

# Re-export from core
from nerfprobe_core import (
    LLMGateway,
    ModelTarget,
    ProbeResult,
    ProbeType,
)
from nerfprobe_core.probes import (
    ADVANCED_PROBES,
    ALL_PROBES,
    CORE_PROBES,
)
from nerfprobe_core.probes.core import (
    CodeProbe,
    MathProbe,
    StyleProbe,
    TimingProbe,
)

from nerfprobe.gateways import AnthropicGateway, GoogleGateway, OpenAIGateway
from nerfprobe.runner import run_probe, run_probes

__all__ = [
    # Runner
    "run_probes",
    "run_probe",
    # Gateways
    "OpenAIGateway",
    "AnthropicGateway",
    "GoogleGateway",
    # Core types
    "ProbeResult",
    "ModelTarget",
    "ProbeType",
    "LLMGateway",
    # Probes
    "MathProbe",
    "StyleProbe",
    "TimingProbe",
    "CodeProbe",
    # Tier lists
    "CORE_PROBES",
    "ADVANCED_PROBES",
    "ALL_PROBES",
]

__version__ = "0.1.0"
