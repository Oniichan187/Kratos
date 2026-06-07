"""Console setup, color palette, banner art, and small shared UI helpers.

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
from dataclasses import dataclass as _dc

from rich.console import Console
from rich.text import Text


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
        # 2. Force UTF-8 encoding on stdout so Python encodes in UTF-8, not the
        #    legacy cp1252 codec — *in place* via reconfigure() rather than
        #    wrapping it in a fresh TextIOWrapper. A fresh wrapper would be
        #    captured once as Console(file=...) and rich would then write to
        #    that pinned object forever — bypassing any later `sys.stdout`
        #    swap (e.g. prompt_toolkit's `patch_stdout()`, used to coexist
        #    with the persistent live-status bar; see _LiveStatus in kratos.py).
        #    Passing file=None makes rich re-resolve `sys.stdout` on every
        #    write, so it automatically routes through such a swap.
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
                return Console(
                    file=None,
                    force_terminal=True,
                    legacy_windows=False,
                    highlight=False,
                )
            # Fallback for stdout objects without reconfigure() (e.g. redirected
            # / piped output) — pin a UTF-8 wrapper as before.
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

_PERM_COLOR = {"low": "yellow", "mid": "green", "high": "red"}
_PERM_COLORS = {"low": "yellow", "mid": "green", "high": "red"}
_PERM_LABEL = {
    "low":  "low  (read)",
    "mid":  "mid  (read+write)",
    "high": "high (read+write+delete)",
}

BANNER = r"""
  ██╗  ██╗██████╗  █████╗ ████████╗ ██████╗ ███████╗
  ██║ ██╔╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔════╝
  █████╔╝ ██████╔╝███████║   ██║   ██║   ██║███████╗
  ██╔═██╗ ██╔══██╗██╔══██║   ██║   ██║   ██║╚════██║
  ██║  ██╗██║  ██║██║  ██║   ██║   ╚██████╔╝███████║
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝    ╚═════╝ ╚══════╝"""

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


def _k_tokens(value: int | None) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value // 1024}k"
    return str(value)


def _short_model(name: str, max_len: int = 32) -> str:
    """Trim long model names — drop registry prefix, keep tag."""
    # "huihui_ai/qwen3-abliterated:8b" → "qwen3-abliterated:8b"
    if "/" in name:
        name = name.split("/", 1)[1]
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


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
    """Compact duration for logs and live stats. Grows left in variable-width bars (s → m → h)."""
    if seconds < 60:
        return f"{int(seconds)}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}m {s}s" if s else f"{m}m"
    h = m // 60
    mm = m % 60
    return f"{h}h {mm}m" if mm else f"{h}h"


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
