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
    # DeepSeek-R1-0528-Qwen3-8B abliterated — planner + verifier (128k native, VRAM-capped to 65536)
    "huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_k_m": 131072,
    "huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3":         131072,
    # Qwen3 abliterated (fallback ALT_PLANNER)
    "huihui_ai/qwen3-abliterated:8b":       40960,
    "huihui_ai/qwen3-abliterated:4b":       40960,
    # Qwen3.5 abliterated (fallback coder)
    "huihui_ai/qwen3.5-abliterated:4b":     262144,
    "huihui_ai/qwen3.5-abliterated:4B":     262144,
    # Qwen2.5-Coder abliterated — primary coder
    "huihui_ai/qwen2.5-coder-abliterate:7b": 32768,
    "huihui_ai/qwen2.5-coder-abliterate:7b-instruct-q4_k_m": 32768,
    # General
    "qwen3:4b":                              262144,
    "kratos-planner":                        131072,
    "kratos-planner:latest":                 131072,
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
    force_max_context: bool = True,
) -> int:
    """Pick num_ctx for an Ollama call.

    When force_max_context=True (default for Kratos "use every model's full window"):
      - Always allocate the largest supported context for that model (capped only by
        explicit vram_ceiling). This is what the user asked for: maximum context
        window on EVERY call for planner, coder, verifier and compressor.
      - Enables reliable work on huge repos that far exceed "small" budgets.
      - Small tasks still get the full window (Ollama only materializes used pages).

    When False: classic conservative "just big enough" calculation.
    """
    cap = min(model_max_ctx(model), vram_ceiling)

    if force_max_context:
        # Use the absolute maximum the model advertises (within hardware limit).
        # Align for KV cache friendliness (many backends like 1k/4k/8k boundaries).
        chosen = cap
    else:
        needed = int((prompt_tokens + max_new_tokens) * overhead)
        chosen = max(min(needed, cap), 1024)

    # Round up to nearest multiple of 1024
    rem = chosen % 1024
    if rem:
        chosen += 1024 - rem

    return min(chosen, cap)


def effective_num_ctx(
    model: str,
    configured_num_ctx: int,
    vram_ctx_ceiling: int,
    *,
    prompt_tokens: int = 0,
    max_new_tokens: int = 0,
    force_max_context: bool = True,
) -> int:
    """Return the real context window that will be passed to a model call."""
    return choose_num_ctx(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        vram_ceiling=min(configured_num_ctx, vram_ctx_ceiling),
        force_max_context=force_max_context,
    )


def role_context_windows(config) -> dict[str, int]:
    """Return effective display windows for Kratos roles."""
    force_max = getattr(config, "always_max_ctx", True)
    return {
        "planner": effective_num_ctx(
            getattr(config, "planner_model"),
            int(getattr(config, "planner_num_ctx")),
            int(getattr(config, "vram_ctx_ceiling")),
            force_max_context=force_max,
        ),
        "coder": effective_num_ctx(
            getattr(config, "coder_model"),
            int(getattr(config, "coder_num_ctx")),
            int(getattr(config, "vram_ctx_ceiling")),
            force_max_context=force_max,
        ),
        "verifier": effective_num_ctx(
            getattr(config, "verifier_model"),
            int(getattr(config, "verifier_num_ctx")),
            int(getattr(config, "vram_ctx_ceiling")),
            force_max_context=force_max,
        ),
        "compressor": effective_num_ctx(
            getattr(config, "compressor_model"),
            int(getattr(config, "compressor_num_ctx")),
            int(getattr(config, "compressor_num_ctx")),
            force_max_context=force_max,
        ),
        "relay": effective_num_ctx(
            getattr(config, "coder_model"),
            int(getattr(config, "relay_num_ctx")),
            int(getattr(config, "relay_num_ctx")),
            force_max_context=force_max,
        ),
        "vram_cap": int(getattr(config, "vram_ctx_ceiling")),
    }


# ── relay decision ─────────────────────────────────────────────────────────────

def relay_needed(
    prompt_tokens: int,
    planner_num_ctx: int,
    relay_threshold: float = 0.80,
) -> bool:
    """Return True when the planner input is too large and needs relay pre-processing."""
    return prompt_tokens > planner_num_ctx * relay_threshold
