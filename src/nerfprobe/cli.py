"""NerfProbe CLI - powered by Typer."""

import asyncio
import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from nerfprobe.gateways import OpenAIGateway, AnthropicGateway, GoogleGateway
from nerfprobe.runner import run_probes, get_probes_for_tier
from nerfprobe_core.probes import PROBE_REGISTRY, CORE_PROBES, ADVANCED_PROBES, OPTIONAL_PROBES

app = typer.Typer(
    name="nerfprobe",
    help="Scientifically-grounded LLM degradation detection.",
    add_completion=False,
)
console = Console()


def get_gateway(
    provider: str,
    api_key: Optional[str],
    base_url: Optional[str],
):
    """Create the appropriate gateway based on provider."""
    if provider == "openai" or base_url:
        return OpenAIGateway(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or "https://api.openai.com/v1",
        )
    elif provider == "anthropic":
        return AnthropicGateway(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "google":
        return GoogleGateway(
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        )
    elif provider == "openrouter":
        return OpenAIGateway(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        # Default to OpenAI-compatible
        return OpenAIGateway(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )


@app.command()
def run(
    model: str = typer.Argument(..., help="Model to test (e.g., gpt-4o, claude-3-opus)"),
    tier: str = typer.Option("core", "--tier", "-t", help="Probe tier: core, advanced, optional, all"),
    probe: Optional[list[str]] = typer.Option(None, "--probe", "-p", help="Specific probes to run"),
    provider: str = typer.Option("openai", "--provider", help="Provider: openai, anthropic, google, openrouter"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key (or use env var)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-u", help="Custom base URL for OpenAI-compatible endpoints"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, markdown"),
):
    """Run probes on an LLM model."""
    gateway = get_gateway(provider, api_key, base_url)

    async def _run():
        try:
            results = await run_probes(
                model_name=model,
                gateway=gateway,
                tier=tier,
                probes=probe,
                provider_id=provider,
            )
            return results
        finally:
            await gateway.close()

    results = asyncio.run(_run())

    if format == "json":
        output = [r.model_dump(mode="json") for r in results]
        console.print_json(json.dumps(output, indent=2, default=str))
    elif format == "markdown":
        console.print(f"# NerfProbe Results: {model}\n")
        for r in results:
            status = "✅" if r.passed else "❌"
            console.print(f"- {status} **{r.probe_name}**: {r.score:.2f} ({r.latency_ms:.0f}ms)")
    else:
        # Table format
        table = Table(title=f"NerfProbe Results: {model}")
        table.add_column("Probe", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Score", justify="right")
        table.add_column("Latency", justify="right")

        for r in results:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            table.add_row(
                r.probe_name,
                status,
                f"{r.score:.2f}",
                f"{r.latency_ms:.0f}ms",
            )

        console.print(table)

        # Summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        console.print(f"\n[bold]Summary:[/bold] {passed}/{total} probes passed")


@app.command()
def list_probes():
    """List available probes."""
    table = Table(title="Available Probes")
    table.add_column("Probe", style="cyan")
    table.add_column("Tier", style="yellow")
    table.add_column("Description")

    descriptions = {
        # Core
        "math": "Arithmetic reasoning degradation [2504.04823]",
        "style": "Vocabulary collapse (TTR) [2403.06408]",
        "timing": "Latency fingerprinting [2502.20589]",
        "code": "Syntax collapse detection [2512.08213]",
        # Advanced
        "fingerprint": "Framework detection [2407.15847]",
        "context": "KV cache compression [2512.12008]",
        "routing": "Model routing detection [2406.18665]",
        "repetition": "Phrase looping [2403.06408]",
        "constraint": "Instruction adherence [2409.11055]",
        "logic": "Reasoning drift [2504.04823]",
        "cot": "Chain-of-thought integrity [2504.04823]",
        # Optional
        "calibration": "Confidence calibration [2511.07585]",
        "zeroprint": "Mode collapse detection [2407.01235]",
        "multilingual": "Cross-language asymmetry [EMNLP.935]",
    }

    for probe_name in PROBE_REGISTRY:
        if probe_name in CORE_PROBES:
            tier = "core"
        elif probe_name in ADVANCED_PROBES:
            tier = "advanced"
        elif probe_name in OPTIONAL_PROBES:
            tier = "optional"
        else:
            tier = "unknown"
        desc = descriptions.get(probe_name, "")
        table.add_row(probe_name, tier, desc)

    console.print(table)


@app.command()
def list_models():
    """List known models in registry."""
    from nerfprobe_core.models import MODELS
    
    table = Table(title="Known Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Provider", style="yellow")
    table.add_column("Context", justify="right")
    table.add_column("Knowledge Cutoff")

    for model_id, info in MODELS.items():
        ctx = f"{info.context_window:,}" if info.context_window else "—"
        cutoff = info.knowledge_cutoff.isoformat() if info.knowledge_cutoff else "—"
        table.add_row(model_id, info.provider, ctx, cutoff)

    console.print(table)
    console.print("\n[dim]Use 'nerfprobe research <model>' for unknown models[/dim]")


@app.command()
def research(
    model: str = typer.Argument(..., help="Model name to research"),
    provider: str = typer.Option("unknown", "--provider", "-p", help="Provider name"),
    parse: Optional[str] = typer.Option(None, "--parse", help="Parse JSON response from LLM"),
):
    """Generate research prompt for unknown models."""
    from nerfprobe_core.models import get_model_info
    from nerfprobe_core.models.research import get_research_prompt, parse_research_response
    
    # Check if already known
    info = get_model_info(model)
    if info and not parse:
        console.print(f"[green]✓[/green] Model '{model}' already in registry:")
        console.print(f"  Provider: {info.provider}")
        console.print(f"  Context Window: {info.context_window:,}" if info.context_window else "  Context Window: unknown")
        console.print(f"  Knowledge Cutoff: {info.knowledge_cutoff}" if info.knowledge_cutoff else "  Knowledge Cutoff: unknown")
        return
    
    if parse:
        # Parse provided JSON response
        result = parse_research_response(model, provider, parse)
        if result:
            console.print(f"[green]✓[/green] Parsed successfully:")
            console.print(f"  ID: {result.id}")
            console.print(f"  Provider: {result.provider}")
            console.print(f"  Context Window: {result.context_window:,}" if result.context_window else "  Context Window: unknown")
            console.print(f"  Knowledge Cutoff: {result.knowledge_cutoff}" if result.knowledge_cutoff else "  Knowledge Cutoff: unknown")
        else:
            console.print("[red]✗[/red] Failed to parse JSON response")
        return
    
    # Generate research prompt
    prompt = get_research_prompt(model, provider)
    console.print("[bold]Research Prompt[/bold] (paste into any LLM):\n")
    console.print(f"[dim]{'─' * 50}[/dim]")
    console.print(prompt)
    console.print(f"[dim]{'─' * 50}[/dim]")
    console.print("\n[dim]Then run: nerfprobe research <model> --provider <provider> --parse '<json>'[/dim]")


@app.command()
def version():
    """Show version."""
    from nerfprobe import __version__
    console.print(f"nerfprobe {__version__}")


if __name__ == "__main__":
    app()
