"""LLM transport + token-budget utilities (Ollama bridge, context-window math)."""
from __future__ import annotations

from .bridge import OllamaBridge, win_to_wsl
from .tokens import (
    estimate,
    estimate_messages,
    fit_to_budget,
    fit_excerpt,
    model_max_ctx,
    choose_num_ctx,
    effective_num_ctx,
    role_context_windows,
    relay_needed,
)

__all__ = [
    "OllamaBridge", "win_to_wsl",
    "estimate", "estimate_messages", "fit_to_budget", "fit_excerpt",
    "model_max_ctx", "choose_num_ctx", "effective_num_ctx",
    "role_context_windows", "relay_needed",
]
