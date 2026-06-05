#!/usr/bin/env python3
"""Kratos model setup wizard — idempotent.

Sets up all three abliterated models required by Kratos:

  Planner    : huihui_ai/qwen3-abliterated:8b    (pull from Ollama hub, ~5 GB)
  Coder      : huihui_ai/qwen3.5-abliterated:4b  (pull from Ollama hub, ~3.3 GB)
  Compressor : kratos-planner                    (local GGUF if available, else pull qwen3:8b)

All models are abliterated (no safety filters).
Hardware assumed: RTX 4050 Laptop (6 GB VRAM) — models load sequentially.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.panel import Panel
    from rich.rule import Rule
    from rich import box
except ImportError:
    sys.exit("pip install rich")

from kratos.bridge import OllamaBridge, win_to_wsl
from kratos.ui import console
from kratos.config import (
    KratosConfig,
    PLANNER_MODEL_NAME,
    CODER_MODEL_NAME,
    COMPRESSOR_MODEL,
    ALT_PLANNER_MODEL,
    _find_planner_gguf,
)

COMPRESSOR_SYSTEM = """\
You summarize conversation history and extract durable facts.
Be concise. Output only what was asked. No preamble."""


def _header(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def _pull(bridge: OllamaBridge, model: str, label: str) -> None:
    console.print(f"  Pulling [bold]{model}[/bold]…  (this may take several minutes)")
    last = ""
    for status in bridge.pull_model(model):
        if status != last:
            console.print(f"  [dim]{status}[/dim]", end="\r")
            last = status
    console.print(f"\n[green]✓[/green]  {label} ready.")


def setup() -> None:
    console.print(Panel.fit(
        "[bold cyan]Kratos Model Setup[/bold cyan]\n"
        "[dim]All models are abliterated — no safety filters[/dim]",
        box=box.ROUNDED, border_style="cyan",
    ))

    bridge = OllamaBridge()

    # ── Start Ollama ──────────────────────────────────────────────────────────
    _header("Starting Ollama")
    if bridge.is_running():
        console.print("[green]✓[/green]  Ollama already running.")
    else:
        console.print("[dim]Starting Ollama…[/dim]")
        if bridge.start():
            console.print("[green]✓[/green]  Ollama started.")
        else:
            console.print("[red]✗[/red]  Cannot start Ollama.")
            console.print("  Start it manually:  [cyan]ollama serve[/cyan]  then re-run.")
            sys.exit(1)

    # ── Planner: huihui_ai/qwen3-abliterated:8b ───────────────────────────────
    _header(f"Planner: {PLANNER_MODEL_NAME}")
    if bridge.model_exists(PLANNER_MODEL_NAME):
        console.print(f"[green]✓[/green]  [cyan]{PLANNER_MODEL_NAME}[/cyan] already installed.")
    else:
        _pull(bridge, PLANNER_MODEL_NAME, f"Planner ({PLANNER_MODEL_NAME})")

    # ── Coder: huihui_ai/qwen3.5-abliterated:4b ──────────────────────────────
    _header(f"Coder: {CODER_MODEL_NAME}")
    if bridge.model_exists(CODER_MODEL_NAME):
        console.print(f"[green]✓[/green]  [green]{CODER_MODEL_NAME}[/green] already installed.")
    else:
        _pull(bridge, CODER_MODEL_NAME, f"Coder ({CODER_MODEL_NAME})")

    # ── Compressor: kratos-planner (Phi-4-mini-abliterated local GGUF) ────────
    _header(f"Compressor: {COMPRESSOR_MODEL}")
    if bridge.model_exists(COMPRESSOR_MODEL):
        console.print(f"[green]✓[/green]  [magenta]{COMPRESSOR_MODEL}[/magenta] already installed.")
    else:
        gguf_win = _find_planner_gguf()
        if gguf_win:
            console.print(f"  Creating from local GGUF: [dim]{gguf_win}[/dim]")
            for status in bridge.create_from_gguf(
                model_name=COMPRESSOR_MODEL,
                gguf_win_path=gguf_win,
                system_prompt=COMPRESSOR_SYSTEM,
                gpu_layers=50,
                ctx=8192,
                temp=0.3,
            ):
                console.print(f"  [dim]{status}[/dim]", end="\r")
            console.print(f"\n[green]✓[/green]  [magenta]{COMPRESSOR_MODEL}[/magenta] created.")
        else:
            console.print(
                f"  [yellow]⚠[/yellow]  GGUF not found in models/ — "
                f"using [cyan]{ALT_PLANNER_MODEL}[/cyan] as compressor instead."
            )
            if not bridge.model_exists(ALT_PLANNER_MODEL):
                _pull(bridge, ALT_PLANNER_MODEL, "Compressor (fallback)")
            # Save alt model name so config uses it
            global _effective_compressor
            _effective_compressor = ALT_PLANNER_MODEL

    # ── Save config ───────────────────────────────────────────────────────────
    _header("Saving configuration")
    cfg = KratosConfig.load()
    cfg.planner_model    = PLANNER_MODEL_NAME
    cfg.coder_model      = CODER_MODEL_NAME
    cfg.compressor_model = globals().get("_effective_compressor", COMPRESSOR_MODEL)
    cfg_path = cfg.save("project")
    console.print(f"[green]✓[/green]  Config saved: [dim]{cfg_path}[/dim]")

    # ── Done ─────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold green]✓  Setup complete![/bold green]\n\n"
        "Start Kratos:  [cyan]kratos[/cyan]  (or [cyan]python kratos.py[/cyan])\n\n"
        f"  Planner    : [cyan]{PLANNER_MODEL_NAME}[/cyan]  (qwen3-abliterated:8b, 40K ctx)\n"
        f"  Coder      : [green]{CODER_MODEL_NAME}[/green]  (qwen3.5-abliterated:4b, 262K ctx)\n"
        f"  Compressor : [magenta]{cfg.compressor_model}[/magenta]  (history compression)\n\n"
        "All models are abliterated — no safety filters.",
        box=box.ROUNDED, border_style="green",
    ))


_effective_compressor = COMPRESSOR_MODEL   # may be overridden during setup()


if __name__ == "__main__":
    setup()
