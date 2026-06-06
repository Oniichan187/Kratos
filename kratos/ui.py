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
import time as _time

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from .tokens import model_max_ctx


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


def _k_tokens(value: int | None) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value // 1024}k"
    return str(value)


def print_banner(
    planner: str,
    coder: str,
    scope: str,
    permission: str = "mid",
    verifier: str | None = None,
    compressor: str | None = None,
    ctx: dict[str, int] | None = None,
    *,
    show_status_panel: bool = True,
) -> None:
    console.print(Align.center(Text(BANNER, style="bold cyan")))
    console.print(Align.center(Text("CLI AI Agent  v4  —  MAX-CTX + STEPWISE + LOSSLESS .kratos  (all abliterated)", style="dim")))
    console.print()
    if show_status_panel:
        _status_line(planner, coder, scope, permission, verifier=verifier, compressor=compressor, ctx=ctx)
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
    verifier: str | None = None,
    compressor: str | None = None,
    ctx: dict[str, int] | None = None,
) -> None:
    pcolor = _PERM_COLOR.get(permission, "green")
    parts = [
        f"[dim]scope[/dim] [bold]{scope}[/bold]",
        f"[dim]perm[/dim] [{pcolor}]{permission}[/{pcolor}]",
        f"[dim]planner[/dim] [cyan]{_short_model(planner)}[/cyan]",
        f"[dim]coder[/dim] [green]{_short_model(coder)}[/green]",
    ]
    if verifier:
        parts.append(f"[dim]verify[/dim] [yellow]{_short_model(verifier)}[/yellow]")
    if compressor:
        parts.append(f"[dim]compress[/dim] [magenta]{_short_model(compressor)}[/magenta]")
    if ctx:
        ctx_text = (
            f"P{_k_tokens(ctx.get('planner'))}/"
            f"C{_k_tokens(ctx.get('coder'))}/"
            f"V{_k_tokens(ctx.get('verifier'))}/"
            f"K{_k_tokens(ctx.get('compressor'))}/"
            f"R{_k_tokens(ctx.get('relay'))}/"
            f"CAP{_k_tokens(ctx.get('vram_cap'))}"
        )
        parts.append(f"[dim]ctx[/dim] [white]{ctx_text}[/white]")
    if goal:
        short = goal[:40] + "…" if len(goal) > 40 else goal
        parts.append(f"[dim]goal[/dim] [yellow]{short}[/yellow]")
    bar = "  │  ".join(parts)
    console.print(Panel(bar, box=box.ROUNDED, border_style="dim", padding=(0, 1)))


def refresh_status(
    planner: str, coder: str, scope: str,
    permission: str = "mid", goal: str | None = None,
    verifier: str | None = None,
    compressor: str | None = None,
    ctx: dict[str, int] | None = None,
) -> None:
    _status_line(planner, coder, scope, permission, goal, verifier, compressor, ctx)


def planner_header(model: str) -> None:
    console.print()
    console.print(Rule(f"[{PLANNER_COLOR}]◈  PLANNER  ·  {model}[/{PLANNER_COLOR}]", style="cyan"))


def coder_header(model: str) -> None:
    console.print()
    console.print(Rule(f"[{CODER_COLOR}]◈  CODER  ·  {model}[/{CODER_COLOR}]", style="green"))


def verify_header(model: str) -> None:
    console.print()
    console.print(Rule(f"[bold yellow]◈  VERIFY  ·  {model}[/bold yellow]", style="yellow"))


def relay_header(model: str) -> None:
    console.print()
    console.print(Rule(f"[bold magenta]◈  RELAY  ·  {model}[/bold magenta]", style="magenta"))


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


def input_capsule(
    ctx_state: "dict | None",
    config,
    last_task_s: "float | None" = None,
    *,
    session_start: "float | None" = None,
    total_tokens: int = 0,
    project_name: str = "",
) -> None:
    """Render a capsule-style info panel directly above the kratos ❯ prompt.

    Displays: session lifetime (m/h) · token usage (xk) · % until auto-compose
    """
    import time as _t

    parts: list[str] = []

    # 1. Session lifetime
    if session_start is not None:
        age_s = _t.time() - session_start
        if age_s < 60:
            life = "<1m"
        elif age_s < 3600:
            life = f"{int(age_s // 60)}m"
        else:
            h, m = int(age_s // 3600), int((age_s % 3600) // 60)
            life = f"{h}h {m}m"
        parts.append(f"[dim]⏱[/dim] [bold white]{life}[/bold white]")

    # 2. Session token usage
    tok_str = _tok_short(total_tokens) if total_tokens > 0 else "0"
    parts.append(f"[dim]∑[/dim] [cyan]{tok_str}[/cyan]")

    # 3. % until auto-compose
    if config is not None:
        coder_total = getattr(config, "coder_num_ctx", 262144)
        threshold = float(getattr(config, "compress_threshold", 0.75))
        coder_used = 0
        if ctx_state:
            coder_used = ctx_state.get("coder", (0, coder_total))[0]
        compose_at = max(1, threshold * coder_total)
        pct = min(100, int(coder_used * 100 / compose_at))
        bar_w = 8
        filled = min(bar_w, int(bar_w * pct / 100))
        bar = "▓" * filled + "░" * (bar_w - filled)
        color = "red" if pct > 80 else "yellow" if pct > 50 else "green"
        parts.append(f"[{color}]{bar} {pct}%[/] [dim]→compose[/dim]")

    # 4. Project name + permission
    if config is not None:
        perm = getattr(config, "permission", None) or "mid"
        pcolor = _PERM_COLORS.get(perm, "green")
        proj = project_name or "·"
        parts.append(f"[dim]{proj}[/dim]  [{pcolor}]{perm}[/]")

    # 5. Last task duration (if any)
    if last_task_s is not None:
        t = f"{last_task_s:.0f}s" if last_task_s < 60 else f"{last_task_s / 60:.1f}m"
        parts.append(f"[dim]last {t}[/dim]")

    sep = "   [dim]│[/dim]   "
    capsule_text = sep.join(parts)
    console.print()
    console.print(Panel(
        Text.from_markup(capsule_text),
        box=box.ROUNDED,
        border_style="dim cyan",
        padding=(0, 2),
    ))


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
    from .prompts import load_prompts
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Live-Streaming UI  (rich.live.Live + layout)
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dc

_ROLE_STYLE: dict[str, str] = {
    "planner":   "bold cyan",
    "coder":     "bold green",
    "verify":    "bold yellow",
    "relay":     "bold magenta",
    "compress":  "magenta",
    "router":    "dim cyan",
    "tool":      "dim blue",
    "info":      "blue",
    "warn":      "yellow",
    "error":     "bold red",
    "success":   "green",
    "direct":    "bold blue",
    "question":  "cyan",
}

_ROLE_ICON: dict[str, str] = {
    "planner":   "◈",
    "coder":     "◇",
    "verify":    "○",
    "relay":     "◎",
    "router":    "⟶",
    "tool":      "↳",
    "direct":    "◆",
}

_PERM_COLORS = {"low": "yellow", "mid": "green", "high": "red"}


@_dc
class _Line:
    text: str
    style: str = ""


class LiveBuffer:
    """Accumulates styled output lines; trims from the top when full."""

    def __init__(self, max_lines: int = 300) -> None:
        self._lines: list[_Line] = []
        self._max = max_lines

    def add(self, text: str, style: str = "") -> None:
        for line in text.split("\n"):
            self._lines.append(_Line(text=line, style=style))
        self._trim()

    def _trim(self) -> None:
        if len(self._lines) > self._max:
            self._lines = self._lines[-self._max:]

    def render(self, max_visible: int = 200) -> Text:
        """Return a Rich Text renderable trimmed to *max_visible* lines."""
        lines = self._lines[-max_visible:] if len(self._lines) > max_visible else self._lines
        t = Text()
        for i, ln in enumerate(lines):
            t.append(ln.text, style=ln.style)
            if i < len(lines) - 1:
                t.append("\n")
        return t

    def clear(self) -> None:
        self._lines.clear()


def status_bar(
    scope: str,
    permission: str,
    ctx_state: dict[str, tuple[int, int]],
    elapsed_s: float,
    project_name: str,
    current_section: str = "",
    *,
    session_tokens: tuple[int, int] = (0, 0),
    goal: str | None = None,
    user_label: str = "",
    hint: str = "",
) -> Panel:
    """Build the persistent bottom status bar as a Rich Panel.

    Line 1: user message (if any) + hint
    Line 2: scope, permission, ctx bars, elapsed, project

    ctx_state is {role: (used_tokens, total_ctx)} per active role.
    """
    from rich.markup import escape as _me

    # ── line 1: user message ──────────────────────────────────────────────
    top_parts: list[str] = []
    if user_label:
        top_parts.append(f"[bold cyan]▸[/] [bold white]{user_label}[/]")
    if hint:
        top_parts.append(f"[dim]{hint}[/]")

    # ── line 2: status info ───────────────────────────────────────────────
    parts: list[str] = []

    # scope + permission
    pcolor = _PERM_COLORS.get(permission, "green")
    parts.append(f"[bold white]{scope}[/]")
    parts.append(f"[{pcolor}]{permission}[/]")

    # goal (if set) — escape user text before embedding in markup
    if goal:
        short_g = _me(goal[:30] + "…" if len(goal) > 30 else goal)
        parts.append(f"[yellow]{short_g}[/]")

    # per-role ctx mini-bars
    for role, abbr in (("planner", "P"), ("coder", "C"), ("verifier", "V")):
        if role in ctx_state:
            used, total = ctx_state[role]
            pct = min(100, int(used * 100 / total)) if total > 0 else 0
            ku = f"{used // 1024}k" if used >= 1024 else str(used)
            kt = f"{total // 1024}k" if total >= 1024 else str(total)
            color = "red" if pct > 80 else "yellow" if pct > 60 else "green"
            bw = 6
            filled = max(0, min(bw, int(bw * pct / 100)))
            bar = "█" * filled + "░" * (bw - filled)
            if role == current_section:
                parts.append(f"[bold {color}]{bar}[/] [bold white]{abbr}[/] [white]{ku}/{kt}[/]")
            else:
                parts.append(f"[{color}]{bar}[/] [bold white]{abbr}[/] [dim]{ku}/{kt}[/]")
        else:
            parts.append(f"[dim]{abbr}:—[/]")

    # elapsed time
    if elapsed_s > 0:
        ts = elapsed_str(elapsed_s)
        parts.append(f"[bold white]⏱{ts}[/]")

    # session tokens
    p_tok, c_tok = session_tokens
    if p_tok + c_tok > 0:
        parts.append(f"[dim]∑{_tok_short(p_tok + c_tok)}[/]")

    # project name
    parts.append(f"[dim italic]{_me(project_name)}[/]")

    # ── assemble with Text.from_markup so styles are actually rendered ────
    bar2 = "  │  ".join(parts)
    if top_parts:
        bar1 = "  ·  ".join(top_parts)
        body = Text.from_markup(f"{bar1}\n{bar2}")
    else:
        body = Text.from_markup(bar2)

    return Panel(body, box=box.ROUNDED, border_style="dim cyan", padding=(0, 1))


def _tok_short(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def _fmt_time(ts: float | None) -> str:
    if ts is None:
        return "--:--"
    lt = _time.localtime(ts)
    return _time.strftime("%H:%M:%S", lt)


def elapsed_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s}s"


def user_message_panel(text: str) -> Panel:
    """Chat-bubble panel for the user's message."""
    from rich.markup import escape as _escape
    now = _fmt_time(_time.time())
    display = text if len(text) <= 400 else text[:397] + "…"
    safe = _escape(display)
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
