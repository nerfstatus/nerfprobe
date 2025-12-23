"""Gateway implementations for LLM providers."""

from nerfprobe.gateways.openai_compat import OpenAIGateway
from nerfprobe.gateways.anthropic import AnthropicGateway
from nerfprobe.gateways.google import GoogleGateway
from nerfprobe.gateways.bedrock import BedrockGateway
from nerfprobe.gateways.dashscope import DashScopeGateway
from nerfprobe.gateways.zhipu import ZhipuGateway
from nerfprobe.gateways.ollama import OllamaGateway

__all__ = [
    # Major providers
    "OpenAIGateway",
    "AnthropicGateway",
    "GoogleGateway",
    # Cloud
    "BedrockGateway",
    # Chinese providers
    "DashScopeGateway",
    "ZhipuGateway",
    # Local
    "OllamaGateway",
]
