"""Presentation shells — CLI loop, Textual TUI, shared input-frame/slash helpers."""
from __future__ import annotations

from .cli import main
from .slash import slash_completions
from .tui import run_tui

__all__ = ["run_tui", "slash_completions", "main"]
