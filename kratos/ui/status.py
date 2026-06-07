"""Status bar, capsule, and banner status-line rendering.

These pieces render the persistent "where are we" surfaces: the startup
banner's status line, the bottom input capsule, and the live status bar
panel used by the streaming UI.
"""

from __future__ import annotations

import time as _time

from rich.align import Align
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..llm.tokens import role_context_windows
from .theme import (
    BANNER,
    _PERM_COLOR,
    _PERM_COLORS,
    _k_tokens,
    _short_model,
    _tok_short,
    console,
    elapsed_str,
)


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

    Always shows: ⏱ (session or last task, grows with m/h) + ∑ tokens + P/C/V ctx + %→compose + project + perm
    The Uhr moves left (takes more space from left) as duration lengthens because it is placed early.
    """
    parts: list[str] = []

    # 1. Time (Uhr) first so it expands left as it grows to minutes/hours; always present
    display_t = last_task_s
    if display_t is None and session_start is not None:
        display_t = _time.time() - session_start
    t_str = elapsed_str(display_t) if display_t and display_t > 0 else "0s"
    parts.append(f"[dim]⏱[/dim] [cyan]{t_str}[/cyan]")

    # 2. Session token usage — exact, comma-grouped (live, no 'k' rounding)
    tok_str = f"{total_tokens:,}" if total_tokens > 0 else "0"
    parts.append(f"[dim]∑[/dim] [cyan]{tok_str}[/cyan]")

    # 3. Effective context windows + % until auto-compose (both % and time now always together with others)
    if config is not None:
        windows = role_context_windows(config)
        for role, abbr in (("planner", "P"), ("coder", "C"), ("verifier", "V")):
            used = 0
            if ctx_state:
                used = ctx_state.get(role, (0, windows[role]))[0]
            parts.append(f"[dim]{abbr}[/dim] [white]{_tok_short(used)}/{_tok_short(windows[role])}[/white]")

        coder_total = windows["coder"]
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

    sep = "   [dim]│[/dim]   "
    capsule_text = sep.join(parts)
    console.print()
    console.print(Panel(
        Text.from_markup(capsule_text),
        box=box.ROUNDED,
        border_style="dim cyan",
        padding=(0, 2),
    ))


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
    compose_pct: int | None = None,
) -> Panel:
    """Build the persistent bottom status bar as a Rich Panel.

    Line 1: user message (if any) + hint
    Line 2: scope, permission, ctx bars, elapsed, tokens, %->compose, project

    ctx_state is {role: (used_tokens, total_ctx)} per active role.
    compose_pct, when given, renders the same live "N%->compose" indicator
    shown in the input frame and TUI footer — kept consistent across all states.
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

    # session tokens — exact, comma-grouped (live ticking, no 'k' rounding)
    p_tok, c_tok = session_tokens
    total_tok = p_tok + c_tok
    if total_tok > 0:
        parts.append(f"[dim]∑{total_tok:,}[/]")

    # %->compose — same live indicator as the input frame and TUI footer
    if compose_pct is not None:
        bw = 8
        filled = min(bw, int(bw * compose_pct / 100))
        bar = "▓" * filled + "░" * (bw - filled)
        bc = "red" if compose_pct > 80 else "yellow" if compose_pct > 50 else "dim"
        parts.append(f"[{bc}]{bar} {compose_pct}%[/][dim]→compose[/]")

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
