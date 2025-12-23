# NerfProbe

**Scientifically-grounded LLM degradation detection for developers.**

[![PyPI](https://img.shields.io/pypi/v/nerfprobe.svg)](https://pypi.org/project/nerfprobe/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Installation

```bash
pip install nerfprobe
```

## Quick Start

### CLI

```bash
# Run core probes on a model
nerfprobe run gpt-5.2 --tier core

# Run specific probes
nerfprobe run gpt-5.2 --probe math --probe style --probe code

# Use different provider
nerfprobe run claude-opus-4.5 --provider anthropic

# Custom endpoint (vLLM, Ollama, local)
nerfprobe run my-model --base-url http://localhost:8000/v1

# Output formats
nerfprobe run gpt-5.2 --format json > results.json
nerfprobe run gpt-5.2 --format markdown
```

### Model Registry

```bash
# List known models (10 SOTA as of Dec 2025)
nerfprobe list-models

# Research unknown model
nerfprobe research qwen3:8b --provider alibaba
# -> Outputs prompt to paste into any LLM

# Parse research response
nerfprobe research qwen3:8b --provider alibaba --parse '{"context_window": 32768}'
```

### Python API

```python
import asyncio
from nerfprobe import run_probes, OpenAIGateway

async def main():
    gateway = OpenAIGateway(api_key="...")
    
    # Run core tier
    results = await run_probes("gpt-5.2", gateway, tier="core")
    
    for r in results:
        print(r.summary())
        # math_probe: PASS (1.00) in 234ms
        # style_probe: PASS (0.87) in 189ms
        # timing_probe: PASS (1.00) in 156ms
        # code_probe: PASS (1.00) in 312ms
    
    await gateway.close()

asyncio.run(main())
```

## Probes

| Tier | Probes | Description |
|------|--------|-------------|
| **core** | math, style, timing, code | Essential degradation signals |
| **advanced** | fingerprint, context, routing, repetition, constraint, logic, cot | Research-backed detection |
| **optional** | calibration, zeroprint, multilingual | Requires logprobs or multi-call |
| **all** | All 14 probes | Comprehensive testing |

## Gateways

| Gateway | Providers |
|---------|-----------|
| `OpenAIGateway` | OpenAI, OpenRouter, vLLM, Ollama, Together, Fireworks |
| `AnthropicGateway` | Claude models |
| `GoogleGateway` | Gemini models |
| `BedrockGateway` | AWS Bedrock (Claude, Titan) |
| `DashScopeGateway` | Alibaba Qwen models |
| `ZhipuGateway` | GLM models |
| `OllamaGateway` | Local Ollama models |

## Environment Variables

```bash
# API keys (or use --api-key flag)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Research Basis

All probes are grounded in peer-reviewed research:

- **MathProbe**: [2504.04823](https://arxiv.org/abs/2504.04823) - Quantization Hurts Reasoning
- **StyleProbe**: [2403.06408](https://arxiv.org/abs/2403.06408) - Perturbation Lens
- **TimingProbe**: [2502.20589](https://arxiv.org/abs/2502.20589) - LLMs Have Rhythm
- **CodeProbe**: [2512.08213](https://arxiv.org/abs/2512.08213) - Package Hallucinations
- **FingerprintProbe**: [2407.15847](https://arxiv.org/abs/2407.15847) - LLMmap
- **ContextProbe**: [2512.12008](https://arxiv.org/abs/2512.12008) - KV Cache Compression
- **RoutingProbe**: [2406.18665](https://arxiv.org/abs/2406.18665) - RouteLLM

## Dependencies

- `nerfprobe-core` - Probe implementations
- `httpx` - HTTP client
- `typer` - CLI framework
- `rich` - Terminal output

## License

Apache-2.0
