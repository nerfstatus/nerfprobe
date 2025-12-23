# NerfProbe

**Scientifically-grounded LLM degradation detection for developers.**

nerfprobe is a CLI tool and Python library for running [nerfprobe-core](https://pypi.org/project/nerfprobe-core/) instruments. It detects specific degradation patterns—reasoning drift, vocabulary collapse, quantization artifacts, and instruction failures—that general benchmarks miss.

[![PyPI](https://img.shields.io/pypi/v/nerfprobe.svg)](https://pypi.org/project/nerfprobe/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Installation

```bash
pip install nerfprobe
```

## Quick Start

### CLI

```bash
# Run core probes (Math, Style, Timing, Code, Fact)
nerfprobe run gpt-5.2 --tier core

# Run advanced probes (Json, Consistency, Fingerprint...)
nerfprobe run gpt-5.2 --tier advanced

# Run specific probes
nerfprobe run gpt-5.2 --probe json --probe consistency --probe math

# Use different provider
nerfprobe run claude-3-opus-20240229 --provider anthropic

# Output formats
nerfprobe run gpt-5.2 --format json > results.json
nerfprobe run gpt-5.2 --format markdown
```

### Model Registry

```bash
# List known models (10 SOTA as of Dec 2025)
nerfprobe list-models
# -> gpt-4o, claude-3-5-sonnet, gemini-1.5-pro...
```

### Python API

```python
import asyncio
from nerfprobe import run_probes, OpenAIGateway

async def main():
    # Automatically tracks usage and cost
    gateway = OpenAIGateway(api_key="...")
    
    # Run core tier
    results = await run_probes("gpt-4o", gateway, tier="core")
    
    for r in results:
        print(r.summary())
        # math_probe: PASS (1.00) in 234ms
        # style_probe: PASS (0.87) in 189ms
        # fact_probe: FAIL (0.00) in 112ms - Hallucinated
    
    await gateway.close()

asyncio.run(main())
```

## Probes

NerfProbe runs 17 distinct instruments organized by utility.

| Tier | Probes | Description |
|------|--------|-------------|
| **Core** | `math`, `style`, `timing`, `code`, `fact` | Essential signals. Low cost, high signal. |
| **Advanced** | `json`, `consistency`, `fingerprint`, `context`, `routing`, `repetition`, `constraint`, `logic`, `cot` | Structural integrity and complex failure modes. |
| **Optional** | `calibration`, `zeroprint`, `multilingual` | Experimental or high-cost probes. |

See [nerfprobe-core](https://github.com/skew202/nerfprobe-core) for deep dives into the science behind each probe.

## Gateways

Supports all major providers via native APIs or OpenAI-compatible endpoints.

| Gateway | Providers |
|---------|-----------|
| `OpenAIGateway` | OpenAI, OpenRouter, vLLM, Ollama, DeepSeek, Together |
| `AnthropicGateway` | Claude models |
| `GoogleGateway` | Gemini models |
| `BedrockGateway` | AWS Bedrock (Claude, Titan) |
| `DashScopeGateway` | Alibaba Qwen models |
| `ZhipuGateway` | GLM models |
| `OllamaGateway` | Local Ollama models (Official API) |

## Environment Variables

```bash
# API keys (auto-detected)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## License

Apache-2.0
