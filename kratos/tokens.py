"""Token budget utilities — no Ollama required.

All estimates use a calibrated character-based heuristic (~3.6 chars/token for
mixed code+prose). A 15 % safety margin is built in so budgets are never
exceeded in practice.

Key functions
  estimate(text)         → int  rough token count
  fit_to_budget(text, n) → str  truncate to at most n tokens
  choose_num_ctx(...)    → int  pick a safe num_ctx for an Ollama call
  relay_needed(tokens, planner_ctx, threshold) → bool
"""
from __future__ import annotations

# ── model capability registry ─────────────────────────────────────────────────
# Verified via /api/show on the actual installed models (2026-06-05).
# All other models default to 32768 (conservative).
_MODEL_MAX_CTX: dict[str, int] = {
    "huihui_ai/qwen3-abliterated:8b":       40960,
    "huihui_ai/qwen3.5-abliterated:4b":     262144,
    "huihui_ai/qwen3.5-abliterated:4B":     262144,  # tag alias
    "huihui_ai/qwen3-abliterated:4b":       40960,
    "huihui_ai/qwen2.5-coder-abliterate:7b": 32768,
    "kratos-planner":                        16384,
    "kratos-planner:latest":                 16384,
}

# VRAM-safe ceiling: even if the model supports more, loading a giant KV-cache
# on a 4-6 GB GPU VRAM causes swapping / OOM. These are the practical limits.
# Users can raise via config (vram_ctx_ceiling).
_DEFAULT_VRAM_CEILING = 32768


# ── estimation ────────────────────────────────────────────────────────────────

_CHARS_PER_TOKEN = 3.6   # calibrated for code+prose mix (Qwen/Llama tokenizers)
_SAFETY_MARGIN   = 1.15  # 15 % over-estimate ensures we never blow the budget


def estimate(text: str) -> int:
    """Return a conservative token estimate for *text*."""
    return int(len(text) / _CHARS_PER_TOKEN * _SAFETY_MARGIN) + 8


def estimate_messages(messages: list[dict]) -> int:
    """Estimate tokens for a list of {'role': ..., 'content': ...} messages."""
    total = 0
    for m in messages:
        total += estimate(m.get("content", ""))
        total += 4   # role + framing tokens per message
    return total + 3  # priming


# ── budget enforcement ────────────────────────────────────────────────────────

def fit_to_budget(text: str, max_tokens: int) -> str:
    """Truncate *text* so it fits within *max_tokens* (conservative estimate).

    Cuts at a line boundary when possible to avoid broken mid-line output.
    Appends a short marker so downstream code knows truncation happened.
    """
    if max_tokens <= 0:
        return ""
    target_chars = int(max_tokens * _CHARS_PER_TOKEN / _SAFETY_MARGIN)
    if len(text) <= target_chars:
        return text
    truncated = text[:target_chars]
    # Cut at last newline to avoid broken lines
    last_nl = truncated.rfind("\n")
    if last_nl > target_chars * 0.7:
        truncated = truncated[:last_nl]
    kept = estimate(truncated)
    return truncated + f"\n... [truncated — {kept} tokens shown]"


def fit_excerpt(text: str, token_budget: int, max_lines: int | None = None) -> str:
    """Return an excerpt of *text* respecting both token_budget and max_lines."""
    if max_lines is not None:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return fit_to_budget(text, token_budget)


# ── num_ctx selection ─────────────────────────────────────────────────────────

def model_max_ctx(model: str) -> int:
    """Return the maximum context length for *model*, defaulting to 32768."""
    # Normalise: strip registry prefix variation
    key = model.lower()
    for k, v in _MODEL_MAX_CTX.items():
        if k.lower() == key:
            return v
    # Partial match: e.g. "huihui_ai/qwen3-abliterated" without tag
    base = key.split(":")[0]
    for k, v in _MODEL_MAX_CTX.items():
        if k.lower().split(":")[0] == base:
            return v
    return 32768


def choose_num_ctx(
    model: str,
    prompt_tokens: int,
    max_new_tokens: int,
    vram_ceiling: int = _DEFAULT_VRAM_CEILING,
    *,
    overhead: float = 1.30,   # 30 % headroom above prompt+output
) -> int:
    """Pick the smallest safe num_ctx for an Ollama call.

    Rules (applied in order):
      1. Must fit prompt + expected output with overhead.
      2. Must not exceed model's hardware maximum.
      3. Must not exceed vram_ceiling (VRAM budget).
      4. Round up to nearest power-of-two-friendly value for KV-cache alignment.
    """
    needed = int((prompt_tokens + max_new_tokens) * overhead)
    cap    = min(model_max_ctx(model), vram_ceiling)
    chosen = max(min(needed, cap), 1024)

    # Round up to nearest multiple of 1024 (KV-cache alignment)
    rem = chosen % 1024
    if rem:
        chosen += 1024 - rem

    return min(chosen, cap)


# ── relay decision ─────────────────────────────────────────────────────────────

def relay_needed(
    prompt_tokens: int,
    planner_num_ctx: int,
    relay_threshold: float = 0.80,
) -> bool:
    """Return True when the planner input is too large and needs relay pre-processing."""
    return prompt_tokens > planner_num_ctx * relay_threshold
