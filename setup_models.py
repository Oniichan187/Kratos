#!/usr/bin/env python3
"""Kratos model setup wizard.

Sets up:
  • Planner: kratos-planner  (Phi-4-mini-abliterated-Q5_K_M, local GGUF)
  • Coder:   huihui_ai/qwen2.5-coder-abliterate:7b-instruct-q4_K_M  (pulled from Ollama hub)

Both models are abliterated (safety-filter removed), as required.
Hardware: RTX 4050 Laptop (4 GB VRAM) — planner fits fully in VRAM,
coder uses partial GPU offload (~20 layers on GPU, rest on CPU RAM).
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
from kratos.ui import console  # UTF-8 / VT-enabled console
from kratos.config import (
    KratosConfig,
    PLANNER_MODEL_NAME,
    CODER_MODEL_NAME,
    FALLBACK_CODER_MODEL,
    ALT_PLANNER_MODEL,
    _find_planner_gguf,
)


PLANNER_SYSTEM = """\
You are Kratos Planner — a strategic AI that creates structured execution plans.
When given a task respond with a numbered list of clear, concrete, actionable steps.
Be concise and precise. Do NOT write code. Finish with: PLAN_COMPLETE"""

CODER_SYSTEM = """\
You are Kratos Coder — an expert software engineer and technical executor.
Implement tasks fully with clean, production-quality code.
Be direct and complete. Never leave TODOs unimplemented."""


def _header(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def setup() -> None:
    console.print(Panel.fit(
        "[bold cyan]Kratos Model Setup[/bold cyan]\n"
        "[dim]RTX 4050 Laptop · 4 GB VRAM · WSL2 + CUDA[/dim]",
        box=box.ROUNDED, border_style="cyan",
    ))

    bridge = OllamaBridge()

    # ── Start Ollama ──────────────────────────────────────────────────────────
    _header("Starting Ollama in WSL (CUDA)")
    if bridge.is_running():
        console.print("[green]✓[/green]  Ollama already running.")
    else:
        console.print("[dim]Launching Ollama in WSL…[/dim]")
        if bridge.start():
            console.print("[green]✓[/green]  Ollama started with CUDA.")
        else:
            console.print("[red]✗[/red]  Could not start Ollama.")
            console.print("  Install Ollama in WSL first:")
            console.print("  [cyan]bash setup_wsl.sh[/cyan]  (or run it inside WSL)")
            sys.exit(1)

    # ── Planner: create from local GGUF ──────────────────────────────────────
    _header(f"Planner: {PLANNER_MODEL_NAME}")
    gguf_win = _find_planner_gguf()

    if not gguf_win:
        console.print("[red]✗[/red]  Planner GGUF not found in models/ directory.")
        console.print("  Expected: models/Phi-4-mini-instruct-abliterated-Q5_K_M-GGUF/")
        _offer_alt_planner(bridge)
    elif bridge.model_exists(PLANNER_MODEL_NAME):
        console.print(f"[green]✓[/green]  [cyan]{PLANNER_MODEL_NAME}[/cyan] already exists.")
    else:
        gguf_wsl = win_to_wsl(gguf_win)
        console.print(f"  GGUF: [dim]{gguf_wsl}[/dim]")
        console.print("  Creating model from GGUF (first time may take ~30 s)…")
        for status in bridge.create_from_gguf(
            model_name=PLANNER_MODEL_NAME,
            gguf_win_path=gguf_win,
            system_prompt=PLANNER_SYSTEM,
            gpu_layers=50,   # Ollama caps at available VRAM; all 32 layers fit in 4 GB
            ctx=4096,
            temp=0.7,
        ):
            console.print(f"  [dim]{status}[/dim]", end="\r")
        console.print(f"\n[green]✓[/green]  [cyan]{PLANNER_MODEL_NAME}[/cyan] ready.")

    # ── Coder: prefer already-installed abliterated models ────────────────────
    # Priority: qwen3-abliterated:8b (already on disk) > NeuralDaredevil (on disk) > pull
    effective_coder = CODER_MODEL_NAME
    if bridge.model_exists(CODER_MODEL_NAME):
        _header(f"Coder: {CODER_MODEL_NAME}")
        console.print(f"[green]✓[/green]  [green]{CODER_MODEL_NAME}[/green] already installed — no download needed.")
    elif bridge.model_exists(FALLBACK_CODER_MODEL):
        effective_coder = FALLBACK_CODER_MODEL
        _header(f"Coder: {FALLBACK_CODER_MODEL}")
        console.print(f"[green]✓[/green]  Using [green]{FALLBACK_CODER_MODEL}[/green] (already installed, abliterated).")
    else:
        _header(f"Coder: {CODER_MODEL_NAME}")
        console.print(f"  Pulling {CODER_MODEL_NAME} (~5 GB)…")
        console.print("  [dim]This may take several minutes.[/dim]")
        last = ""
        for status in bridge.pull_model(CODER_MODEL_NAME):
            if status != last:
                console.print(f"  [dim]{status}[/dim]", end="\r")
                last = status
        console.print(f"\n[green]✓[/green]  [green]{CODER_MODEL_NAME}[/green] ready.")

    # ── Save config ───────────────────────────────────────────────────────────
    _header("Saving configuration")
    cfg = KratosConfig.load()
    cfg.planner_model = PLANNER_MODEL_NAME
    cfg.coder_model = effective_coder
    cfg_path = cfg.save("project")
    console.print(f"[green]✓[/green]  Config saved: [dim]{cfg_path}[/dim]")

    # ── Done ─────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold green]✓  Setup complete![/bold green]\n\n"
        "Start Kratos:  [cyan]python kratos.py[/cyan]\n\n"
        f"  Planner : [cyan]{PLANNER_MODEL_NAME}[/cyan]  (Phi-4-mini-abliterated Q5_K_M)\n"
        f"  Coder   : [green]{effective_coder}[/green]\n\n"
        "Both models are abliterated — no safety filters.",
        box=box.ROUNDED, border_style="green",
    ))


def _offer_alt_planner(bridge: OllamaBridge) -> None:
    """Offer to pull the alternative abliterated planner from Ollama hub."""
    console.print()
    console.print(f"  Alternatively, pull [cyan]{ALT_PLANNER_MODEL}[/cyan] as planner?")
    try:
        answer = input("  Pull qwen3-abliterated:8b as planner? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer == "y":
        console.print(f"  Pulling {ALT_PLANNER_MODEL} (~5 GB)…")
        last = ""
        for status in bridge.pull_model(ALT_PLANNER_MODEL):
            if status != last:
                console.print(f"  [dim]{status}[/dim]", end="\r")
                last = status
        console.print(f"\n[green]✓[/green]  {ALT_PLANNER_MODEL} ready.")
        # Patch the global for the config save
        import kratos.config as _cfg
        _cfg.PLANNER_MODEL_NAME = ALT_PLANNER_MODEL  # type: ignore[attr-defined]
    else:
        console.print("  Skipped. Edit [cyan]/models planner <name>[/cyan] in the REPL to change later.")


if __name__ == "__main__":
    setup()
