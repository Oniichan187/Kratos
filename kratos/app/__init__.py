"""Presentation shells — CLI loop, Textual TUI, shared input-frame/slash helpers.

Re-exports are LAZY (PEP 562): importing ``kratos.app`` — or a light submodule
such as ``kratos.app.prompt_frame`` — must never pull in the heavy UI
dependencies (``rich`` / ``prompt_toolkit`` / ``textual``) nor abort the
interpreter via ``sys.exit`` when they are absent. ``main`` / ``run_tui`` /
``slash_completions`` resolve on first attribute access only, i.e. when the CLI
or TUI is actually launched.
"""
from __future__ import annotations

__all__ = ["run_tui", "slash_completions", "main"]

_LAZY = {
    "main": (".cli", "main"),
    "slash_completions": (".slash", "slash_completions"),
    "run_tui": (".tui", "run_tui"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(target[0], __name__)
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__)
