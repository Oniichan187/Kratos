"""Kratos configuration — global (~/.kratos/) and project (.kratos/) scope."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

PermissionLevel = Literal["low", "mid", "high"]

_PERMISSION_CAPS: dict[str, frozenset[str]] = {
    "low":  frozenset({"read"}),
    "mid":  frozenset({"read", "write"}),
    "high": frozenset({"read", "write", "delete"}),
}

GLOBAL_DIR = Path.home() / ".kratos"

def _project_dir() -> Path:
    return Path.cwd() / ".kratos"

PROJECT_DIR = Path(".kratos")   # backward-compat; use _project_dir() for new code

_KRATOS_INSTALL_DIR = Path(__file__).resolve().parents[1]

# ── verified model names ──────────────────────────────────────────────────────
# All models MUST be abliterated (uncensored, no safety filters).
# These are strong abliterated models that play well together on a laptop
# (RTX 4050 6 GB class, sequential loading only — never all in VRAM at once).
# Each role uses the *best suitable* abliterated model for its job:
#   Planner   : strong reasoning + CoT (8b class, 40k native)
#   Coder     : huge context (262k) + solid coding, small enough for laptop
#   Verifier  : same strong reasoning as planner (strict PROVEN_WORK judge)
#   Compressor: tiny but faithful summarizer/memory extractor (Phi-4-mini ablit)
PLANNER_MODEL_NAME  = "huihui_ai/qwen3-abliterated:8b"
CODER_MODEL_NAME    = "huihui_ai/qwen3.5-abliterated:4b"
VERIFIER_MODEL_NAME = "huihui_ai/qwen3-abliterated:8b"
COMPRESSOR_MODEL    = "kratos-planner"         # Phi-4-mini-instruct-abliterated GGUF
FALLBACK_CODER_MODEL = "huihui_ai/qwen2.5-coder-abliterate:7b"
ALT_PLANNER_MODEL   = "huihui_ai/qwen3-abliterated:8b"

# GGUF for kratos-planner (Phi-4-mini-abliterated) — relative to install dir
_PLANNER_GGUF: Path = (
    _KRATOS_INSTALL_DIR
    / "models"
    / "Phi-4-mini-instruct-abliterated-Q5_K_M-GGUF"
    / "phi-4-mini-instruct-abliterated-q5_k_m.gguf"
)


def _find_planner_gguf() -> str:
    if _PLANNER_GGUF.exists():
        return str(_PLANNER_GGUF)
    models_dir = _KRATOS_INSTALL_DIR / "models"
    if models_dir.exists():
        for p in models_dir.rglob("*.gguf"):
            if "abliterat" in p.name.lower() or "abliterat" in p.parent.name.lower():
                return str(p)
    return ""


@dataclass
class KratosConfig:
    # ── Models ────────────────────────────────────────────────────────────────
    planner_model:    str = field(default_factory=lambda: PLANNER_MODEL_NAME)
    coder_model:      str = field(default_factory=lambda: CODER_MODEL_NAME)
    verifier_model:   str = field(default_factory=lambda: VERIFIER_MODEL_NAME)
    compressor_model: str = field(default_factory=lambda: COMPRESSOR_MODEL)
    planner_gguf_win: str = field(default_factory=_find_planner_gguf)

    # ── Session ───────────────────────────────────────────────────────────────
    scope:      Literal["global", "project"] = "project"
    goal:       str | None = None
    permission: PermissionLevel = "mid"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    gpu_layers:  int = 50

    # ── Context windows (num_ctx) ─────────────────────────────────────────────
    # KRATOS ALWAYS USES THE MAXIMUM CONTEXT WINDOW THE MODEL SUPPORTS
    # (capped only by vram_ctx_ceiling). This is intentional:
    # every planner, coder, verifier and compressor call gets the full window.
    # This is required for huge projects that massively exceed "normal" ctx,
    # and for the coder/verifier to see complete plans + full file state + memory.
    # choose_num_ctx(..., force_max_context=True) implements this.
    planner_num_ctx:    int = 40960    # full max for huihui_ai/qwen3-abliterated:8b
    coder_num_ctx:      int = 262144   # full max for huihui_ai/qwen3.5-abliterated:4b (huge-repo hero)
    verifier_num_ctx:   int = 40960    # full max — same strong ablit model as planner
    compressor_num_ctx: int = 32768    # generous for kratos-planner (Phi-4-mini-abliterated). Use its real max when known
    relay_num_ctx:      int = 131072   # coder relay gets big window for giant inputs before planner sees them

    # VRAM ceiling: hard cap for all roles even if model advertises more.
    # On 6 GB laptop keep conservative; raise on bigger cards.
    # The "always max" policy still respects this.
    vram_ctx_ceiling: int = 65536

    # ── Temperatures ──────────────────────────────────────────────────────────
    planner_temp:    float = 0.6
    coder_temp:      float = 0.15
    verifier_temp:   float = 0.15
    compressor_temp: float = 0.3

    # ── Auto-compression ──────────────────────────────────────────────────────
    auto_compress:       bool  = True    # compress history when it overflows
    compress_threshold:  float = 0.75   # trigger when prompt > X × num_ctx
    relay_threshold:     float = 0.80   # trigger relay when input > X × planner_num_ctx
    max_history_pairs:   int   = 8      # hard cap per model before forced compress

    # ── Max-context policy (user requirement) ─────────────────────────────────
    # When True (default), planner/coder/verifier/compressor ALWAYS receive
    # the model's full context window (within vram cap). No "small prompt = small ctx".
    always_max_ctx: bool = True

    # ── Build / test / verify ─────────────────────────────────────────────────
    build_cmd:             str | None = None
    test_cmd:              str | None = None
    build_test_retries:    int = 3
    max_verify_iterations: int = 10
    auto_discover_verification: bool = True
    require_proven_work:        bool = True
    require_test_for_verified:  bool = True
    verification_timeout_seconds: int = 120

    # ── Persistence ───────────────────────────────────────────────────────────
    # Prompts are loaded independently via kratos/prompts.py (same GLOBAL_DIR / _project_dir pattern).
    # See .kratos/prompts.json or ~/.kratos/prompts.json for system prompts + snippets (edit + /prompts reload).

    def save(self, scope: Literal["global", "project"] = "project") -> Path:
        target = GLOBAL_DIR if scope == "global" else _project_dir()
        target.mkdir(parents=True, exist_ok=True)
        cfg_file = target / "config.json"
        cfg_file.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return cfg_file

    @classmethod
    def load(cls) -> "KratosConfig":
        """Merge global + project config. Project overrides global.

        Global  : ~/.kratos/config.json
        Project : CWD/.kratos/config.json  (wherever `kratos` was invoked)
        """
        merged: dict = {}
        for path in [GLOBAL_DIR / "config.json", _project_dir() / "config.json"]:
            if path.exists():
                try:
                    merged.update(json.loads(path.read_text("utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass
        valid = {f for f in cls.__dataclass_fields__}
        inst = cls(**{k: v for k, v in merged.items() if k in valid})

        # Enforce "use maximum context window for every model" when requested.
        # This upgrades old/small saved configs automatically to the model's true max.
        if getattr(inst, "always_max_ctx", True):
            try:
                from .tokens import model_max_ctx
                inst.planner_num_ctx = max(inst.planner_num_ctx, model_max_ctx(inst.planner_model))
                inst.coder_num_ctx = max(inst.coder_num_ctx, model_max_ctx(inst.coder_model))
                inst.verifier_num_ctx = max(inst.verifier_num_ctx, model_max_ctx(inst.verifier_model))
                inst.compressor_num_ctx = max(inst.compressor_num_ctx, model_max_ctx(inst.compressor_model))
                inst.relay_num_ctx = max(inst.relay_num_ctx, 65536)
            except Exception:
                pass
        return inst

    # ── Permission helpers ────────────────────────────────────────────────────

    def can_read(self) -> bool:
        return True

    def can_write(self) -> bool:
        return "write" in _PERMISSION_CAPS.get(self.permission, _PERMISSION_CAPS["mid"])

    def can_delete(self) -> bool:
        return "delete" in _PERMISSION_CAPS.get(self.permission, _PERMISSION_CAPS["mid"])

    def is_allowed(self, action: str) -> bool:
        if action in ("read_file", "list_files"):
            return True
        if action == "file_write":
            return self.can_write()
        if action in ("delete_files", "delete_file"):
            return self.can_delete()
        if action == "bash_exec":
            return self.can_write()
        return False
