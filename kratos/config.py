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
# These are the ACTUALLY installed abliterated models (confirmed 2026-06-05).
PLANNER_MODEL_NAME  = "huihui_ai/qwen3-abliterated:8b"
CODER_MODEL_NAME    = "huihui_ai/qwen3.5-abliterated:4b"
COMPRESSOR_MODEL    = "kratos-planner"         # Phi-4-mini-abliterated local GGUF
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
    # Default VRAM-safe values; choose_num_ctx() adjusts dynamically per call.
    planner_num_ctx:    int = 12288    # qwen3:8b max=40960; 12K default is fast+safe
    coder_num_ctx:      int = 24576    # qwen3.5:4b max=262144; 24K for normal coding
    compressor_num_ctx: int = 8192     # Phi-4-mini max≈16K; 8K for history compression
    relay_num_ctx:      int = 32768    # Coder in relay mode — bigger window for huge inputs

    # VRAM ceiling: choose_num_ctx() will never exceed this regardless of model max.
    # Raise this if you have more VRAM (e.g. 48576 for 12 GB GPU).
    vram_ctx_ceiling: int = 32768

    # ── Temperatures ──────────────────────────────────────────────────────────
    planner_temp:    float = 0.6
    coder_temp:      float = 0.15
    compressor_temp: float = 0.3

    # ── Auto-compression ──────────────────────────────────────────────────────
    auto_compress:       bool  = True    # compress history when it overflows
    compress_threshold:  float = 0.75   # trigger when prompt > X × num_ctx
    relay_threshold:     float = 0.80   # trigger relay when input > X × planner_num_ctx
    max_history_pairs:   int   = 8      # hard cap per model before forced compress

    # ── Build / test / verify ─────────────────────────────────────────────────
    build_cmd:             str | None = None
    test_cmd:              str | None = None
    build_test_retries:    int = 3
    max_verify_iterations: int = 10

    # ── Persistence ───────────────────────────────────────────────────────────

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
        return cls(**{k: v for k, v in merged.items() if k in valid})

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
