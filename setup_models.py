#!/usr/bin/env python3
"""Kratos model setup wizard — idempotent.

Sets up all FOUR abliterated models (max-context, play perfectly together):

  Planner    : huihui_ai/qwen3-abliterated:8b    (reasoning + plans, 40K full ctx)
  Coder      : huihui_ai/qwen3.5-abliterated:4b  (implementation, 262K full ctx — giant repo capable)
  Verifier   : huihui_ai/qwen3-abliterated:8b    (strict step-by-step PROVEN_WORK judge)
  Compressor : kratos-planner                    (Phi-4-mini-abliterated GGUF) — LOSSLESS .kratos memory + history

All models are abliterated (no safety filters).
Kratos forces MAXIMUM context window on every single call to every model.
Hardware: laptop (RTX 4050 6 GB class) — roles load sequentially, never simultaneous.
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

from kratos.llm.bridge import OllamaBridge, win_to_wsl
from kratos.ui import console
from kratos.config import (
    KratosConfig,
    PLANNER_MODEL_NAME,
    CODER_MODEL_NAME,
    VERIFIER_MODEL_NAME,
    COMPRESSOR_MODEL,
    ALT_PLANNER_MODEL,
    _find_planner_gguf,
)

from kratos.prompts import get_system

# COMPRESSOR_SYSTEM for the GGUF modelfile bake is now loaded from the central prompts JSON
# (key "compressor_model_system"). This keeps it in sync with runtime prompts and editable.
COMPRESSOR_SYSTEM = get_system("compressor_model_system")


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

    # ── Planner (reasoning) ──────────────────────────────────────────────────
    _header(f"Planner: {PLANNER_MODEL_NAME}")
    if bridge.model_exists(PLANNER_MODEL_NAME):
        console.print(f"[green]✓[/green]  [cyan]{PLANNER_MODEL_NAME}[/cyan] already installed.")
    else:
        _pull(bridge, PLANNER_MODEL_NAME, f"Planner ({PLANNER_MODEL_NAME})")

    # ── Coder (huge-ctx implementation) ──────────────────────────────────────
    _header(f"Coder: {CODER_MODEL_NAME}")
    if bridge.model_exists(CODER_MODEL_NAME):
        console.print(f"[green]✓[/green]  [green]{CODER_MODEL_NAME}[/green] already installed.")
    else:
        _pull(bridge, CODER_MODEL_NAME, f"Coder ({CODER_MODEL_NAME})")

    # ── Verifier (same strong ablit model — strict judge) ────────────────────
    _header(f"Verifier: {VERIFIER_MODEL_NAME}")
    if bridge.model_exists(VERIFIER_MODEL_NAME):
        console.print(f"[green]✓[/green]  [yellow]{VERIFIER_MODEL_NAME}[/yellow] already installed.")
    else:
        _pull(bridge, VERIFIER_MODEL_NAME, f"Verifier ({VERIFIER_MODEL_NAME})")

    # ── Compressor / Auto-Composer (lossless memory in .kratos) ──────────────
    _header(f"Compressor (auto-composer): {COMPRESSOR_MODEL}")
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
                ctx=32768,  # max-ctx friendly
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
            global _effective_compressor
            _effective_compressor = ALT_PLANNER_MODEL

    # ── Save config ───────────────────────────────────────────────────────────
    _header("Saving configuration")
    cfg = KratosConfig.load()
    cfg.planner_model    = PLANNER_MODEL_NAME
    cfg.coder_model      = CODER_MODEL_NAME
    cfg.verifier_model   = VERIFIER_MODEL_NAME
    cfg.compressor_model = globals().get("_effective_compressor", COMPRESSOR_MODEL)
    cfg_path = cfg.save("project")
    console.print(f"[green]✓[/green]  Config saved: [dim]{cfg_path}[/dim]")

    # ── Done ─────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold green]✓  Setup complete![/bold green]\n\n"
        "Start Kratos:  [cyan]kratos[/cyan]  (or [cyan]python kratos.py[/cyan])\n\n"
        f"  Planner    : [cyan]{PLANNER_MODEL_NAME}[/cyan]   (full 40K ctx every call)\n"
        f"  Coder      : [green]{CODER_MODEL_NAME}[/green]   (full 262K — stepwise per plan item + test)\n"
        f"  Verifier   : [yellow]{VERIFIER_MODEL_NAME}[/yellow]   (full 40K — every test + step proven)\n"
        f"  Compressor : [magenta]{cfg.compressor_model}[/magenta] (lossless .kratos memory, max ctx)\n\n"
        "All models abliterated. Prompts live in JSON (edit .kratos/prompts.json or ~/.kratos/prompts.json).\n"
        "Every call uses MAX context window. The prompt *flow* is fully programmed.",
        box=box.ROUNDED, border_style="green",
    ))


_effective_compressor = COMPRESSOR_MODEL   # may be overridden during setup()


if __name__ == "__main__":
    setup()
