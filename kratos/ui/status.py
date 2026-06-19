"""Status bar, capsule, and banner status-line rendering."""

from __future__ import annotations

import time as _time

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.text import Text

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
    console.print(Align.center(Text("CLI AI Agent  v4  -  MAX-CTX + STEPWISE + LOSSLESS .kratos  (all abliterated)", style="dim")))
    console.print()
    if show_status_panel:
        _status_line(planner, coder, scope, permission, verifier=verifier, compressor=compressor, ctx=ctx)
        console.print()


def _status_line(
    planner: str,
    coder: str,
    scope: str,
    permission: str = "mid",
    goal: str | None = None,
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
        short = goal[:40] + "..." if len(goal) > 40 else goal
        parts.append(f"[dim]goal[/dim] [yellow]{short}[/yellow]")
    bar = "  |  ".join(parts)
    console.print(Panel(bar, box=box.ROUNDED, border_style="dim", padding=(0, 1)))


def refresh_status(
    planner: str,
    coder: str,
    scope: str,
    permission: str = "mid",
    goal: str | None = None,
    verifier: str | None = None,
    compressor: str | None = None,
    ctx: dict[str, int] | None = None,
) -> None:
    _status_line(planner, coder, scope, permission, goal, verifier, compressor, ctx)


def _ctx_bar(role: str, used: int, total: int, *, active: bool) -> str:
    pct = min(100, int(used * 100 / total)) if total > 0 else 0
    ku = f"{used // 1024}k" if used >= 1024 else str(used)
    kt = f"{total // 1024}k" if total >= 1024 else str(total)
    color = "red" if pct > 80 else "yellow" if pct > 60 else "green"
    bw = 6
    filled = max(0, min(bw, int(bw * pct / 100)))
    bar = "█" * filled + "░" * (bw - filled)
    label = {"planner": "P", "coder": "C", "verifier": "V"}.get(role, "?")
    prefix = "bold " if active else ""
    return f"[{prefix}{color}]{bar}[/] [bold white]{label}[/] [white]{ku}/{kt}[/]"


def input_capsule(
    ctx_state: "dict | None",
    config,
    last_task_s: "float | None" = None,
    *,
    session_start: "float | None" = None,
    project_name: str = "",
) -> None:
    """Render the capsule above the prompt."""
    parts: list[str] = []
    display_t = last_task_s
    if display_t is None and session_start is not None:
        display_t = _time.time() - session_start
    t_str = elapsed_str(display_t) if display_t and display_t > 0 else "0s"
    parts.append(f"[dim]⏱[/dim] [cyan]{t_str}[/cyan]")

    if config is not None:
        windows = role_context_windows(config)
        for role, abbr in (("planner", "P"), ("coder", "C"), ("verifier", "V")):
            used = 0
            if ctx_state:
                used = ctx_state.get(role, (0, windows[role]))[0]
            parts.append(f"[dim]{abbr}[/dim] [white]{_tok_short(used)}/{_tok_short(windows[role])}[/white]")

    if config is not None:
        perm = getattr(config, "permission", None) or "mid"
        pcolor = _PERM_COLORS.get(perm, "green")
        proj = project_name or "·"
        parts.append(f"[dim]{proj}[/dim]  [{pcolor}]{perm}[/]")

    sep = "   │   "
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
    goal: str | None = None,
    user_label: str = "",
    hint: str = "",
    plan_state: dict | None = None,  # compact live todo: {"label": "PLAN 3/8", "compact": "..."}
) -> Panel:
    """Build the persistent bottom status bar as a Rich Panel.

    When a role is active, the bar shows only that live role window.
    """
    from rich.markup import escape as _me

    top_parts: list[str] = []
    if user_label:
        top_parts.append(f"[bold cyan]>[/] [bold white]{user_label}[/]")
    if hint:
        top_parts.append(f"[dim]{hint}[/]")

    parts: list[str] = []
    pcolor = _PERM_COLORS.get(permission, "green")
    parts.append(f"[bold white]{scope}[/]")
    parts.append(f"[{pcolor}]{permission}[/]")

    if goal:
        short_g = _me(goal[:30] + "..." if len(goal) > 30 else goal)
        parts.append(f"[yellow]{short_g}[/]")

    active_role = current_section if current_section in ctx_state else ""
    if active_role:
        used, total = ctx_state[active_role]
        parts.append(_ctx_bar(active_role, used, total, active=True))
    else:
        for role in ("planner", "coder", "verifier"):
            if role in ctx_state:
                used, total = ctx_state[role]
                parts.append(_ctx_bar(role, used, total, active=False))
            else:
                parts.append(f"[dim]{role[0].upper()}:—[/]")

    if elapsed_s > 0:
        # NOTE: the stopwatch emoji renders 2 cells wide in most terminals but
        # Rich counts 1 — without the space the first digit gets overdrawn
        # ("⏱12m" looked like "⏱2m"). Keep the explicit space.
        parts.append(f"[bold white]⏱ {elapsed_str(elapsed_s)}[/]")

    parts.append(f"[dim italic]{_me(project_name)}[/]")

    bar2 = "  │  ".join(parts)
    if top_parts:
        bar1 = "  ·  ".join(top_parts)
        body_lines = [bar1, bar2]
    else:
        body_lines = [bar2]

    # Live plan/todo integrated with the bottom stats (next to ctx P/C/V "live stats").
    # This is the persistent "live todo über dem userinput" the user wants in the CLI frame.
    # Show ONLY the single active item (the one currently being worked on) — not the whole
    # truncated checklist — so the line stays one item tall and advances live as items finish.
    if plan_state:
        plabel = plan_state.get("label") or ""
        pactive = plan_state.get("active") or ""
        if plabel or pactive:
            if pactive:
                plan_line = f"[magenta]{plabel}[/magenta]  [white]{_me(pactive[:72])}[/white]"
            else:
                plan_line = f"[magenta]{plabel}[/magenta]  [green]✓ all items done[/green]"
            body_lines.insert(0, plan_line)

    body = Text.from_markup("\n".join(body_lines))
    return Panel(body, box=box.ROUNDED, border_style="dim cyan", padding=(0, 1))
