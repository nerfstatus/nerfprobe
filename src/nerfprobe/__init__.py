"""NerfProbe - Scientifically-grounded LLM degradation detection."""

from nerfprobe.runner import run_probes, run_probe
from nerfprobe.gateways import OpenAIGateway, AnthropicGateway, GoogleGateway

# Re-export from core
from nerfprobe_core import (
    ProbeResult,
    ModelTarget,
    ProbeType,
    LLMGateway,
)
from nerfprobe_core.probes.core import (
    MathProbe,
    StyleProbe,
    TimingProbe,
    CodeProbe,
)
from nerfprobe_core.probes import (
    CORE_PROBES,
    ADVANCED_PROBES,
    ALL_PROBES,
)

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
