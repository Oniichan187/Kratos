"""Rich UI components for Kratos CLI.

All output goes through the module-level ``console`` instance which is
configured once at import time to work correctly on Windows:

  - Enables ENABLE_VIRTUAL_TERMINAL_PROCESSING (ANSI escape codes).
  - Sets the Windows console code page to UTF-8 (65001).
  - Creates the Console with ``force_terminal=True, legacy_windows=False``
    so Rich uses ANSI paths instead of the cp1252 legacy renderer.

Import ``console`` from here — never create a new ``Console()`` directly
in Kratos code, or the Windows encoding fix won't apply.
"""

from __future__ import annotations

import io
import sys

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box


def _make_console() -> Console:
    """Return a Rich Console with UTF-8 + ANSI output on Windows."""
    if sys.platform == "win32":
        # 1. Switch Windows console code page to UTF-8.
        try:
            import ctypes
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            k32.SetConsoleOutputCP(65001)
            h = k32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if k32.GetConsoleMode(h, ctypes.byref(mode)):
                k32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass
        # 2. Wrap stdout in a UTF-8 TextIOWrapper so Python encodes in UTF-8,
        #    not the legacy cp1252 codec.
        try:
            utf8_out = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            return Console(
                file=utf8_out,
                force_terminal=True,
                legacy_windows=False,
                highlight=False,
            )
        except Exception:
            pass
    return Console(force_terminal=True, legacy_windows=False, highlight=False)


console = _make_console()

# ── Colors ────────────────────────────────────────────────────────────────────
PLANNER_COLOR = "bold cyan"
CODER_COLOR = "bold green"
ERROR_COLOR = "bold red"
INFO_COLOR = "blue"
SUCCESS_COLOR = "green"
WARN_COLOR = "yellow"

BANNER = r"""
  ██╗  ██╗██████╗  █████╗ ████████╗ ██████╗ ███████╗
  ██║ ██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔════╝
  █████╔╝ ██████╔╝███████║   ██║   ██║   ██║███████╗
  ██╔═██╗ ██╔══██╗██╔══██║   ██║   ██║   ██║╚════██║
  ██║  ██╗██║  ██║██║  ██║   ██║   ╚██████╔╝███████║
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝    ╚═════╝ ╚══════╝"""


def print_banner(planner: str, coder: str, scope: str, permission: str = "mid") -> None:
    console.print(Align.center(Text(BANNER, style="bold cyan")))
    console.print(Align.center(Text("CLI AI Agent  v2.0  —  local coding agent", style="dim")))
    console.print()
    _status_line(planner, coder, scope, permission)
    console.print()


_PERM_COLOR = {"low": "yellow", "mid": "green", "high": "red"}
_PERM_LABEL = {
    "low":  "low  (read)",
    "mid":  "mid  (read+write)",
    "high": "high (read+write+delete)",
}


def _short_model(name: str, max_len: int = 32) -> str:
    """Trim long model names — drop registry prefix, keep tag."""
    # "huihui_ai/qwen3-abliterated:8b" → "qwen3-abliterated:8b"
    if "/" in name:
        name = name.split("/", 1)[1]
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def _status_line(
    planner: str, coder: str, scope: str,
    permission: str = "mid", goal: str | None = None,
) -> None:
    pcolor = _PERM_COLOR.get(permission, "green")
    parts = [
        f"[dim]scope[/dim] [bold]{scope}[/bold]",
        f"[dim]perm[/dim] [{pcolor}]{permission}[/{pcolor}]",
        f"[dim]planner[/dim] [cyan]{_short_model(planner)}[/cyan]",
        f"[dim]coder[/dim] [green]{_short_model(coder)}[/green]",
    ]
    if goal:
        short = goal[:40] + "…" if len(goal) > 40 else goal
        parts.append(f"[dim]goal[/dim] [yellow]{short}[/yellow]")
    bar = "  │  ".join(parts)
    console.print(Panel(bar, box=box.ROUNDED, border_style="dim", padding=(0, 1)))


def refresh_status(
    planner: str, coder: str, scope: str,
    permission: str = "mid", goal: str | None = None,
) -> None:
    _status_line(planner, coder, scope, permission, goal)


def planner_header(model: str) -> None:
    console.print()
    console.print(Rule(f"[{PLANNER_COLOR}]◈  PLANNER  ·  {model}[/{PLANNER_COLOR}]", style="cyan"))


def coder_header(model: str) -> None:
    console.print()
    console.print(Rule(f"[{CODER_COLOR}]◈  CODER  ·  {model}[/{CODER_COLOR}]", style="green"))


def verify_header(model: str) -> None:
    console.print()
    console.print(Rule(f"[bold yellow]◈  VERIFY  ·  {model}[/bold yellow]", style="yellow"))


def section_end() -> None:
    console.print()


def route_info(msg: str) -> None:
    console.print(f"[dim]  ⟶  {msg}[/dim]")


def tool_call(msg: str) -> None:
    console.print(f"  [dim blue]↳[/dim blue] [blue]tool[/blue]  [dim]{msg}[/dim]")


def direct_header() -> None:
    console.print()
    console.print(Rule("[bold blue]◈  SEARCH RESULT[/bold blue]", style="blue"))


def print_error(msg: str) -> None:
    console.print(f"[{ERROR_COLOR}]✗[/{ERROR_COLOR}]  {msg}")


def print_success(msg: str) -> None:
    console.print(f"[{SUCCESS_COLOR}]✓[/{SUCCESS_COLOR}]  {msg}")


def print_info(msg: str) -> None:
    console.print(f"[{INFO_COLOR}]ℹ[/{INFO_COLOR}]  {msg}")


def print_warn(msg: str) -> None:
    console.print(f"[{WARN_COLOR}]⚠[/{WARN_COLOR}]  {msg}")


def print_help() -> None:
    rows = [
        ("/goal [text]",                  "Set or show current goal"),
        ("/goal clear",                   "Clear goal"),
        ("/scope <global|project>",       "Switch config scope"),
        ("/permission",                    "Show current permission level"),
        ("/permission low",               "Read only — coder cannot write files"),
        ("/permission mid",               "Read + write (default)"),
        ("/permission high",              "Read + write + delete"),
        ("/models",                       "Show configured models"),
        ("/models planner <name>",        "Change planner model"),
        ("/models coder <name>",          "Change coder model"),
        ("/index",                        "Show indexed project files"),
        ("/index rebuild",                "Rescan project directory"),
        ("/memory list",                  "Show all memory entries"),
        ("/memory clear [session|project|all]", "Clear memory tier"),
        ("/build [cmd]",                  "Set or show build command"),
        ("/test [cmd]",                   "Set or show test command"),
        ("/status",                       "Show current status bar"),
        ("/history clear",                "Reset conversation history + session memory"),
        ("/clear",                        "Clear screen"),
        ("/setup",                        "Model setup instructions"),
        ("/logging",                       "Show logging status"),
        ("/logging on",                   "Enable session logging to .kratos/session_*.jsonl"),
        ("/logging off",                  "Disable logging"),
        ("/help",                         "Show this help"),
        ("/exit",                         "Quit Kratos"),
    ]
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(Panel(table, title="[bold]Kratos Commands[/bold]", border_style="cyan", box=box.ROUNDED))


def show_permission_level(level: str) -> None:
    rows = [
        ("low",  "read only",           "read files, list files, answer questions"),
        ("mid",  "read + write",        "create and overwrite files within project"),
        ("high", "read + write + delete", "create, overwrite, and delete files"),
    ]
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Level", width=6)
    table.add_column("Caps", width=26)
    table.add_column("Description")
    for lvl, caps, desc in rows:
        color = _PERM_COLOR.get(lvl, "white")
        marker = "►" if lvl == level else " "
        table.add_row(
            f"[{color}]{marker} {lvl}[/{color}]",
            f"[{'bold' if lvl == level else 'dim'}]{caps}[/{'bold' if lvl == level else 'dim'}]",
            f"[{'white' if lvl == level else 'dim'}]{desc}[/{'white' if lvl == level else 'dim'}]",
        )
    console.print(Panel(table, title=f"[bold]Permission[/bold]  current=[bold]{level}[/bold]", border_style="dim", box=box.ROUNDED))


def show_models(planner: str, coder: str) -> None:
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Role", width=10)
    table.add_column("Model")
    table.add_row("[cyan]Planner[/cyan]", planner)
    table.add_row("[green]Coder[/green]", coder)
    console.print(table)
