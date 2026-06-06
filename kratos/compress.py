"""Model-based compression and large-input relay.

Two roles, two models, all abliterated, always sequential (never simultaneous):

  Compressor  — kratos-planner (Phi-4-mini-abliterated, ~16K)
    compress_history()   : condense old chat pairs into a compact summary note
                           so the rolling history stays within token budget.
    generate_memory()    : extract durable, project-generic facts from a
                           completed task (decisions, conventions, file roles).

  Relay       — coder model (qwen3.5-abliterated:4b, 262K) in a special mode
    relay_large_input()  : when a planner prompt would overflow the planner's
                           40K window, send the raw large input through the
                           high-capacity coder first to produce a compact
                           structured extract, then feed that to the planner.

Both functions have deterministic algo fallbacks so the agent never stalls if a
model is unavailable (Ollama not running, model not loaded yet).

Prompts are ultrashort "caveman style" to minimise token overhead.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bridge import OllamaBridge
    from .config import KratosConfig

# Prompts externalized to JSON (see kratos/prompts.py). Import the getters so the
# programmed flow (user_text construction, token est, _call_model) stays dynamic.
from .prompts import get_system, get_snippet, load_prompts


# ── algo fallbacks ────────────────────────────────────────────────────────────

def _algo_compress(history: list[dict], keep_pairs: int) -> None:
    """Fast in-place history trim when the model is unavailable.

    Drops the oldest pairs and prepends a minimal placeholder so the model
    still knows context was dropped.
    """
    if len(history) <= keep_pairs * 2:
        return
    surplus = len(history) - keep_pairs * 2
    if surplus % 2:
        surplus += 1
    surplus = min(surplus, len(history) - 2)

    facts: list[str] = []
    for i in range(0, surplus, 2):
        if i + 1 >= len(history):
            break
        files = re.findall(r'###\s+FILE:\s*(\S+)', history[i + 1].get("content", ""))
        if files:
            facts.append("wrote " + ", ".join(files[:3]))
        errs = [
            ln.strip()[:100]
            for ln in history[i].get("content", "").splitlines()
            if any(x in ln.lower() for x in ("error", "fail", "exception", "circular"))
        ]
        if errs:
            facts.append("issue: " + errs[0])

    del history[:surplus]
    if facts and history:
        note = "[Context compressed — prior actions: " + " | ".join(facts[:5]) + "]\n\n"
        history[0] = {**history[0], "content": note + history[0].get("content", "")}


def _algo_memory(coder_output: str, changed_files: list[str]) -> list[dict]:
    """Minimal deterministic memory extraction when the model is unavailable."""
    entries: list[dict] = []
    if changed_files:
        entries.append({
            "category": "file_role",
            "content": f"Modified: {', '.join(changed_files[:6])}",
        })
    # Detect circular-import hints
    if re.search(r"circular.{0,30}import", coder_output, re.I):
        entries.append({
            "category": "convention",
            "content": "Avoid circular imports between modules",
        })
    return entries


def _algo_relay(large_text: str, max_chars: int = 8000) -> str:
    """Algo relay fallback — keep first+last chunk and file listing."""
    lines = large_text.splitlines()
    file_lines = [l for l in lines if l.startswith("---") or l.startswith("###")]
    header = "\n".join(file_lines[:40])
    head = large_text[:max_chars // 2]
    tail = large_text[-max_chars // 4:] if len(large_text) > max_chars else ""
    parts = [p for p in [header, head, tail] if p]
    return "\n...\n".join(parts)[:max_chars]


# ── Compressor ────────────────────────────────────────────────────────────────

class Compressor:
    def __init__(self, bridge: "OllamaBridge", config: "KratosConfig") -> None:
        self._bridge = bridge
        self._config = config

    def _call_model(
        self,
        model: str,
        system: str,
        user: str,
        num_ctx: int,
        num_predict: int,
        temperature: float,
    ) -> str:
        """Run a model call and return the full text response (no streaming display).
        num_ctx passed in should already be the MAX for the model (enforced by caller).
        """
        full = ""
        try:
            for token, kind in self._bridge.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                num_predict=num_predict,
                num_ctx=num_ctx,
                think=False,
                keep_alive="0",
            ):
                if kind == "text":
                    full += token
                # skip "think" and "usage" silently
        except Exception:
            pass
        return full.strip()

    # ── history compression ───────────────────────────────────────────────────

    def compress_history(
        self,
        history: list[dict],
        keep_pairs: int = 4,
    ) -> bool:
        """Condense the oldest history pairs into a compact context note.

        Modifies *history* in-place. Returns True if model was used, False if
        algo fallback ran (model unavailable or response empty).

        keep_pairs: how many recent pairs to keep verbatim after compression.
        """
        if len(history) <= keep_pairs * 2:
            return False

        surplus_pairs = (len(history) - keep_pairs * 2) // 2
        surplus_pairs = max(1, surplus_pairs)
        to_compress = history[: surplus_pairs * 2]

        # Build a minimal transcript of the pairs being dropped
        transcript_parts: list[str] = []
        for i in range(0, len(to_compress), 2):
            if i + 1 >= len(to_compress):
                break
            u = to_compress[i].get("content", "")[:400]
            a = to_compress[i + 1].get("content", "")[:600]
            transcript_parts.append(f"User: {u}\nAssistant: {a}")
        transcript = "\n\n---\n\n".join(transcript_parts)

        from .tokens import choose_num_ctx, estimate as _est
        _sys = get_system("compress")
        _ptok = _est(transcript) + _est(_sys)
        _fmax = getattr(self._config, "always_max_ctx", True)
        _nctx = choose_num_ctx(
            model=self._config.compressor_model,
            prompt_tokens=_ptok,
            max_new_tokens=512,
            vram_ceiling=self._config.compressor_num_ctx,
            force_max_context=_fmax,
        )
        summary = self._call_model(
            model=self._config.compressor_model,
            system=_sys,
            user=transcript,
            num_ctx=_nctx,
            num_predict=512,
            temperature=self._config.compressor_temp,
        )

        if not summary or len(summary) < 20:
            # Fallback
            _algo_compress(history, keep_pairs)
            return False

        # Replace the compressed pairs with a single context note
        del history[: surplus_pairs * 2]
        note = f"[Compressed prior context]\n{summary}\n\n"
        if history:
            history[0] = {**history[0], "content": note + history[0].get("content", "")}
        else:
            history.insert(0, {"role": "user", "content": note})
        return True

    # ── memory extraction ─────────────────────────────────────────────────────

    def generate_memory(
        self,
        task: str,
        plan: str,
        coder_output: str,
        changed_files: list[str],
    ) -> list[dict]:
        """Extract durable project-generic memory entries from a completed task.

        Returns list of {category, content} dicts. Falls back to algo extraction
        if the model is unavailable or returns invalid JSON.
        """
        user_text = (
            f"Task: {task[:300]}\n\n"
            f"Plan summary: {plan[:400]}\n\n"
            f"Key output: {coder_output[:600]}\n\n"
            f"Files changed: {', '.join(changed_files[:8])}"
        )
        from .tokens import choose_num_ctx, estimate as _est
        _sys = get_system("memory")
        _ptok = _est(user_text) + _est(_sys)
        _fmax = getattr(self._config, "always_max_ctx", True)
        _nctx = choose_num_ctx(
            model=self._config.compressor_model,
            prompt_tokens=_ptok,
            max_new_tokens=256,
            vram_ceiling=self._config.compressor_num_ctx,
            force_max_context=_fmax,
        )
        raw = self._call_model(
            model=self._config.compressor_model,
            system=_sys,
            user=user_text,
            num_ctx=_nctx,
            num_predict=256,
            temperature=self._config.compressor_temp,
        )

        # Parse JSON array from response
        try:
            # strip markdown code fences if present
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.M)
            entries = json.loads(clean)
            if isinstance(entries, list):
                valid = [
                    e for e in entries
                    if isinstance(e, dict)
                    and "category" in e and "content" in e
                    and len(str(e["content"])) <= 220
                ]
                return valid[:12]
        except (json.JSONDecodeError, ValueError):
            pass

        return _algo_memory(coder_output, changed_files)

    # ── large-input relay ─────────────────────────────────────────────────────

    def relay_large_input(
        self,
        task: str,
        large_context: str,
    ) -> str:
        """Pre-process a large context through the coder model before the planner.

        Used when the estimated planner prompt tokens exceed relay_threshold ×
        planner_num_ctx. The coder model (262K context) ingests the full large
        context and returns a compact structured extract.

        Falls back to algo truncation if the coder model is unavailable.
        """
        user_text = (
            f"Task the planner will receive:\n{task[:500]}\n\n"
            f"Large input to summarise:\n{large_context}"
        )

        from .tokens import choose_num_ctx, estimate
        _sys = get_system("relay_detailed") or get_system("relay")
        prompt_tokens = estimate(user_text) + estimate(_sys)
        force_max = getattr(self._config, "always_max_ctx", True)
        num_ctx = choose_num_ctx(
            model=self._config.coder_model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=1024,
            vram_ceiling=self._config.relay_num_ctx,
            force_max_context=force_max,
        )

        result = self._call_model(
            model=self._config.coder_model,
            system=_sys,
            user=user_text,
            num_ctx=num_ctx,
            num_predict=1200,
            temperature=0.2,
        )

        if not result or len(result) < 50:
            return _algo_relay(large_context)
        return result
