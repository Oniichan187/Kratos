"""Section headers, message prints, summary panels, and help/info displays."""

from __future__ import annotations

import time as _time

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from ..llm.tokens import model_max_ctx
from .theme import (
    CODER_COLOR,
    ERROR_COLOR,
    INFO_COLOR,
    PLANNER_COLOR,
    SUCCESS_COLOR,
    WARN_COLOR,
    _PERM_COLOR,
    _ROLE_ICON,
    _ROLE_STYLE,
    _fmt_time,
    _k_tokens,
    _short_model,
    console,
    elapsed_str,
)


def planner_header(model: str) -> None:
    from ..prompts import load_prompts
    console.print()
    pm = load_prompts()
    text = pm.get_snippet("role_banner_planner").format(model=model) or f"◈  PLANNER  ·  {model}"
    console.print(Rule(f"[{PLANNER_COLOR}]{text}[/{PLANNER_COLOR}]", style="cyan"))


def coder_header(model: str) -> None:
    from ..prompts import load_prompts
    console.print()
    pm = load_prompts()
    text = pm.get_snippet("role_banner_coder").format(model=model) or f"◈  CODER  ·  {model}"
    console.print(Rule(f"[{CODER_COLOR}]{text}[/{CODER_COLOR}]", style="green"))


def verify_header(model: str) -> None:
    from ..prompts import load_prompts
    console.print()
    pm = load_prompts()
    text = pm.get_snippet("role_banner_verify").format(model=model) or f"◈  VERIFY  ·  {model}"
    console.print(Rule(f"[bold yellow]{text}[/bold yellow]", style="yellow"))


def relay_header(model: str) -> None:
    from ..prompts import load_prompts
    console.print()
    pm = load_prompts()
    text = pm.get_snippet("role_banner_relay").format(model=model) or f"◈  RELAY  ·  {model}"
    console.print(Rule(f"[bold magenta]{text}[/bold magenta]", style="magenta"))


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


def print_ctx_bar(role: str, used: int, total: int) -> None:
    """Show context window utilization as an inline progress bar."""
    pct = min(100, int(used * 100 / total)) if total > 0 else 0
    bar_width = 20
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    color = "red" if pct > 80 else "yellow" if pct > 60 else "green"
    console.print(
        f"  [dim]ctx[/dim] [{color}]{bar}[/{color}] "
        f"[dim]{used:,}/{total:,} ({pct}%)[/dim]  "
        f"[dim italic]{role}[/dim italic]"
    )


def print_compress_event(msg: str) -> None:
    """Show auto-compression notification."""
    console.print(f"  [magenta]⇒ compress[/magenta]  [dim]{msg}[/dim]")


def print_section_time(elapsed_s: float, label: str = "") -> None:
    """Show elapsed time for a section (planner, coder, total task)."""
    if elapsed_s < 60:
        t = f"{elapsed_s:.1f}s"
    else:
        mins = int(elapsed_s // 60)
        secs = elapsed_s % 60
        t = f"{mins}m {secs:.0f}s"
    label_part = f"  [dim]{label}[/dim]" if label else ""
    console.print(f"  [dim]⏱ {t}[/dim]{label_part}")


def print_user_msg(text: str) -> None:
    """Show the user's submitted message clearly separated from AI output."""
    from rich.markup import escape
    short = text if len(text) <= 200 else text[:197] + "…"
    console.print()
    console.print(f"[bold cyan]┌─[/bold cyan] [bold white]you[/bold white]")
    for line in escape(short).splitlines():
        console.print(f"[bold cyan]│[/bold cyan]  {line}")
    console.print(f"[bold cyan]└{'─' * 60}[/bold cyan]")


def print_input_separator(project_name: str) -> None:
    """No-op: visual separation is now provided by the input_capsule panel."""


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
        ("/models planner <name>",         "Change planner model"),
        ("/models coder <name>",          "Change coder model"),
        ("/models verifier <name>",       "Change verifier model"),
        ("/models compressor <name>",     "Change compressor model"),
        ("/index",                        "Show indexed project files"),
        ("/index rebuild",                "Rescan project directory"),
        ("/memory list",                  "Show all memory entries"),
        ("/memory clear [session|project|all]", "Clear memory tier"),
        ("/build [cmd]",                  "Set or show build command"),
        ("/test [cmd]",                   "Set or show test command"),
        ("/tokens",                        "Show session token usage"),
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


def print_usage(prompt_tokens: int, completion_tokens: int) -> None:
    total = prompt_tokens + completion_tokens
    console.print(
        f"  [dim]tokens  prompt=[cyan]{prompt_tokens:,}[/cyan]  "
        f"completion=[green]{completion_tokens:,}[/green]  "
        f"total=[bold]{total:,}[/bold][/dim]"
    )


def show_models(
    planner: str,
    coder: str,
    verifier: str | None = None,
    compressor: str | None = None,
    ctx: dict[str, int] | None = None,
) -> None:
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Role", width=11)
    table.add_column("Model")
    table.add_column("Kratos ctx", justify="right", width=11)
    table.add_column("Model max", justify="right", width=10)

    rows = [
        ("Planner", "cyan", planner, "planner"),
        ("Coder", "green", coder, "coder"),
    ]
    if verifier:
        rows.append(("Verifier", "yellow", verifier, "verifier"))
    if compressor:
        rows.append(("Compressor", "magenta", compressor, "compressor"))

    for role, color, model, key in rows:
        table.add_row(
            f"[{color}]{role}[/{color}]",
            model,
            _k_tokens(ctx.get(key) if ctx else None),
            _k_tokens(model_max_ctx(model)),
        )

    if ctx and "relay" in ctx:
        table.add_row("[magenta]Relay[/magenta]", coder, _k_tokens(ctx.get("relay")), _k_tokens(model_max_ctx(coder)))
    if ctx and "vram_cap" in ctx:
        table.add_row("[dim]VRAM cap[/dim]", "[dim]all roles[/dim]", _k_tokens(ctx.get("vram_cap")), "")

    console.print(table)


def show_prompts(prompts_mgr=None) -> None:
    """Quick overview for /prompts list (full content via dump for editing)."""
    from rich.table import Table
    from ..prompts import load_prompts
    pm = prompts_mgr or load_prompts()
    eff = pm.get_all()
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Key", width=22)
    table.add_column("Len", justify="right", width=6)
    table.add_column("Preview (first 70 chars)")

    for k in ("planner_system", "coder_system", "verifier_system", "compress_system", "memory_system", "compressor_model_system"):
        val = eff.get(k, "")
        preview = (val or "")[:70].replace("\n", " ")
        table.add_row(k, str(len(val or "")), preview)

    snips = eff.get("snippets", {})
    table.add_row("snippets", str(len(snips)), ", ".join(list(snips.keys())[:6]) + ("..." if len(snips) > 6 else ""))
    console.print(table)
    console.print("[dim]Edit ~/.kratos/prompts.json or ./.kratos/prompts.json then /prompts reload[/dim]")


def user_message_panel(text: str) -> Panel:
    """Chat-bubble panel for the user's message."""
    from rich.markup import escape as _escape
    now = _fmt_time(_time.time())
    safe = _escape(text)
    body_lines: list[str] = []
    for line in safe.splitlines():
        body_lines.append(f"  {line}")
    body = Text("\n".join(body_lines) if body_lines else "  (empty)")
    return Panel(
        body,
        title=f"[bold white]You[/]  [dim]{now}[/]",
        border_style="bold cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def task_summary_panel(
    elapsed_s: float,
    files_changed: list[str],
    token_usage: tuple[int, int],
    *,
    status: str = "completed",
) -> Panel:
    """Post-task summary panel shown after the Live display exits."""
    lines: list[str] = []

    lines.append(f"[bold]⏱ {elapsed_str(elapsed_s)}[/]")

    p_tok, c_tok = token_usage
    total = p_tok + c_tok
    lines.append(f"[dim]tokens[/]  prompt={p_tok:,}  completion={c_tok:,}  total={total:,}")

    if files_changed:
        names = ", ".join(f'"{f}"' for f in files_changed[:5])
        suffix = f" +{len(files_changed) - 5} more" if len(files_changed) > 5 else ""
        lines.append(f"[dim]files[/]  {names}{suffix}")
    else:
        lines.append("[dim]files[/]  none")

    status_color = "green" if status == "completed" else "yellow"
    body = Text("\n".join(lines))
    return Panel(
        body,
        title=f"[{status_color}]Task {status}[/]",
        border_style=status_color,
        box=box.ROUNDED,
        padding=(0, 2),
    )


def section_banner(role: str, model: str) -> Text:
    """Compact section start marker for inside the Live body."""
    style = _ROLE_STYLE.get(role, "bold")
    icon = _ROLE_ICON.get(role, "●")
    short = _short_model(model) if model else role
    return Text(f"\n  {icon} {role.upper()}  ·  {short}\n", style=style)
