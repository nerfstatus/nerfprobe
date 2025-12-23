"""NerfProbe CLI - powered by Typer."""

import asyncio
import json
import os
from typing import Optional

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from crontab import CronTab
from dotenv import load_dotenv

# Load environment variables (supports .env in CWD or ~/.nerfprobe/.env)
load_dotenv()
load_dotenv(Path.home() / ".nerfprobe" / ".env")

from nerfprobe.gateways import OpenAIGateway, AnthropicGateway, GoogleGateway
from nerfprobe.runner import run_probes, get_probes_for_tier
from nerfprobe.storage import ResultStore, BaselineStore
from nerfprobe_core.probes import PROBE_REGISTRY, CORE_PROBES, ADVANCED_PROBES, OPTIONAL_PROBES

app = typer.Typer(
    name="nerfprobe",
    help="Scientifically-grounded LLM degradation detection.",
    add_completion=False,
)
baseline_app = typer.Typer(help="Manage model baselines.")
schedule_app = typer.Typer(help="Manage automated schedules.")
config_app = typer.Typer(help="Configuration and API keys.")
app.add_typer(baseline_app, name="baseline")
app.add_typer(schedule_app, name="schedule")
app.add_typer(config_app, name="config")

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

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

# ... (imports)

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
    # Header Info
    console.print(f"[dim]nerfprobe v0.2.0 • {datetime.now().strftime('%H:%M:%S')}[/dim]")
    console.print(f"[dim]Target: {model} ({provider})[/dim]\n")

    async def _run_with_progress():
        from nerfprobe.runner import DEFAULT_CONFIGS, run_probe
        from nerfprobe_core import ModelTarget, ProbeResult

        # Initialize Gateway inside the loop
        gateway = get_gateway(provider, api_key, base_url)

        try:
            # Determine probes
            if probe:
                probe_names = list(probe)
            else:
                probe_names = get_probes_for_tier(tier)
            
            # Filter available
            probe_names = [p for p in probe_names if p in PROBE_REGISTRY]
            total_probes = len(probe_names)
            
            results = []
            
            # UI: Progress Bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True, # Disappear when done
            ) as progress:
                task_id = progress.add_task(f"Running {tier} probes...", total=total_probes)
                
                for name in probe_names:
                    progress.update(task_id, description=f"Probing [cyan]{name}[/cyan]...")
                    try:
                        # Run single probe
                        r = await run_probe(model, gateway, name, provider_id=provider)
                        results.append(r)
                    except Exception as e:
                        # Handle error cleanly without crashing UI
                        from nerfprobe_core import ProbeType
                        target = ModelTarget(provider_id=provider, model_name=model)
                        results.append(ProbeResult(
                            probe_name=name,
                            probe_type=ProbeType.MATH,
                            target=target,
                            passed=False,
                            score=0.0,
                            latency_ms=0.0,
                            raw_response=f"Error: {e}",
                            metadata={}
                        ))
                    
                    progress.advance(task_id)
                    
            return results
        finally:
            await gateway.close()

    results = asyncio.run(_run_with_progress())

    if format == "json":
        output = [r.model_dump(mode="json") for r in results]
        console.print_json(json.dumps(output, indent=2, default=str))
    elif format == "markdown":
        console.print(f"# NerfProbe Results: {model}\n")
        for r in results:
            status = "✅" if r.passed else "❌"
            console.print(f"- {status} **{r.probe_name}**: {r.score:.2f} ({r.latency_ms:.0f}ms)")
    else:
        # Table format (ccusage style)
        table = Table(box=None, padding=(0, 2))
        table.add_column("Probe", style="cyan bold")
        table.add_column("Status")
        table.add_column("Score", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("In", justify="right", style="dim")
        table.add_column("Out", justify="right", style="dim")
        table.add_column("Reason / Drift", style="red")

        # Persistence & Drift
        result_store = ResultStore()
        baseline_store = BaselineStore()
        drifts = []

        for r in results:
            result_store.append(r)
            
            # Drift Logic
            baseline = baseline_store.get_baseline_score(model, r.probe_name)
            detail_str = ""
            if baseline is not None and r.score < (baseline * 0.9):
                drop = ((baseline - r.score) / baseline) * 100
                detail_str = f"📉 -{drop:.1f}%"
                drifts.append(r)
            
            # Failure Reason overrides drift in Detail column
            if r.error_reason:
                detail_str = r.error_reason

            status_icon = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            if "📉" in detail_str:
                status_icon = "[yellow]DRIFT[/yellow]"

            table.add_row(
                r.probe_name,
                status_icon,
                f"{r.score:.2f}",
                f"{r.latency_ms:.0f}ms",
                str(r.input_tokens if r.input_tokens is not None else "-"),
                str(r.output_tokens if r.output_tokens is not None else "-"),
                detail_str or "-"
            )

        # Summary Panel
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        summary_text = f"Probes: {passed}/{total} Passing"
        if drifts:
            summary_text += f"\n[yellow]Performance Drift Detected in {len(drifts)} probes[/yellow]"
        
        summary_text += "\n[dim]Results saved to local storage[/dim]"

        # Layout
        console.print()
        console.print(Panel(
            table,
            title=f"[bold]NerfProbe Analysis: {model}[/bold]",
            subtitle=summary_text,
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2)
        ))


@baseline_app.command("set")
def baseline_set(
    model: str = typer.Argument(..., help="Model to establish baseline for"),
    tier: str = typer.Option("core", "--tier", "-t", help="Probe tier"),
    runs: int = typer.Option(3, "--runs", "-n", help="Number of runs to average"),
    provider: str = typer.Option("openai", "--provider", help="Provider"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k"),
):
    """Run probes multiple times to establish a baseline."""
    gateway = get_gateway(provider, api_key, None)
    store = BaselineStore()
    
    console.print(f"Establishing baseline for [cyan]{model}[/cyan] ({runs} runs)...")
    
    async def _collect():
        all_results = []
        for i in range(runs):
            console.print(f"  Run {i+1}/{runs}...")
            r = await run_probes(model, gateway, tier=tier, provider_id=provider)
            all_results.extend(r)
        return all_results
    
    try:
        results = asyncio.run(_collect())
        store.save_baseline(model, results)
        console.print(f"[green]✓[/green] Baseline saved for {model}")
    finally:
        asyncio.run(gateway.close())


@baseline_app.command("show")
def baseline_show(model: Optional[str] = typer.Argument(None)):
    """Show stored baselines."""
    store = BaselineStore()
    if model:
        data = store.get_model_baselines(model)
        if not data:
            console.print(f"No baseline found for {model}")
            return
        _print_baseline(model, data)
    else:
        # Show all (not implemented efficiently in store yet, assuming one file)
        # We need to expose _load or add get_all()
        # For now, just load file manually or strictly require model
        # Let's add get_all to store later. For now require model or fail.
        # Check storage file content
        from pathlib import Path
        from platformdirs import user_data_dir
        path = Path(user_data_dir("nerfprobe", "nerfstatus")) / "baselines.json"
        if not path.exists():
            console.print("No baselines recorded.")
            return
        
        with open(path) as f:
            all_data = json.load(f)
        
        for m, data in all_data.items():
            _print_baseline(m, data)


def _print_baseline(model: str, data: dict):
    table = Table(title=f"Baseline: {model}")
    table.add_column("Probe")
    table.add_column("Avg Score")
    table.add_column("Samples")
    table.add_column("Last Updated")
    
    for probe, info in data.items():
        table.add_row(
            probe,
            f"{info['score']:.2f}",
            str(info['samples']),
            info['last_updated']
        )
    console.print(table)


@schedule_app.command("add")
def schedule_add(
    model: str = typer.Argument(..., help="Model to schedule"),
    frequency: str = typer.Option("daily", "--frequency", help="daily, weekly, hourly"),
    tier: str = typer.Option("core", "--tier", help="Probe tier"),
):
    """Schedule automated runs via cron."""
    # Build command
    python_path = os.sys.executable
    cmd = f"{python_path} -m nerfprobe run {model} --tier {tier} --provider openai" # Defaults
    
    cron = CronTab(user=True)
    job = cron.new(command=cmd, comment=f"nerfprobe-{model}")
    
    if frequency == "daily":
        job.setall("0 9 * * *") # 9 AM
    elif frequency == "weekly":
        job.setall("0 9 * * 1") # Monday 9 AM
    elif frequency == "hourly":
        job.setall("0 * * * *")
    else:
        console.print("[red]Unknown frequency[/red]")
        return
        
    job.enable()
    cron.write()
    console.print(f"[green]✓[/green] Scheduled {model} ({frequency})")


@schedule_app.command("list")
def schedule_list():
    """List scheduled jobs."""
    cron = CronTab(user=True)
    table = Table(title="Scheduled NerfProbe Jobs")
    table.add_column("Model/Comment")
    table.add_column("Schedule")
    table.add_column("Command")
    
    for job in cron:
        if job.comment.startswith("nerfprobe"):
            table.add_row(job.comment, str(job.slices), job.command)
            
    console.print(table)


@app.command("tui")
def tui():
    """Launch terminal dashboard."""
    from nerfprobe.tui.app import NerfProbeApp
    app = NerfProbeApp()
    app.run()


@config_app.command("info")
def config_info():
    """Show configuration and API key status."""
    console.print("[bold]Configuration Status[/bold]")
    
    # Check common keys
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "OLLAMA_BASE_URL",
    ]
    
    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Current Value (Masked)")
    
    for key in keys:
        val = os.getenv(key)
        status = "[green]Set[/green]" if val else "[red]Missing[/red]"
        masked = f"{val[:4]}...{val[-4:]}" if val and len(val) > 8 else "—"
        table.add_row(key, status, masked)
        
    console.print(table)
    
    console.print("\n[bold]Usage Guide:[/bold]")
    console.print("1. create a `.env` file in your current directory or `~/.nerfprobe/.env`.")
    console.print("2. Add your keys:")
    console.print("   OPENAI_API_KEY=sk-...")
    console.print("   ANTHROPIC_API_KEY=sk-ant-...")
    console.print("\n[bold]Ollama:[/bold] (No key usually needed, sets base URL)")
    console.print("   OLLAMA_BASE_URL=http://localhost:11434/v1")
    console.print("   nerfprobe run llama3 --provider openai --base-url http://localhost:11434/v1 --api-key ollama")
    console.print("\n[bold]Groq:[/bold] (Uses OpenAI compatible gateway)")
    console.print("   OPENAI_API_KEY=gsk_... (or pass --api-key)")
    console.print("   OPENAI_BASE_URL=https://api.groq.com/openai/v1")
    console.print("   nerfprobe run llama3-70b-8192 --provider openai --base-url https://api.groq.com/openai/v1")



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
