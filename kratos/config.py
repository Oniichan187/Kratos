"""Kratos configuration — global (~/.kratos/) and project (.kratos/) scope."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# Permission levels
# low  — read only  (index, read files, answer questions)
# mid  — read + write (default: coder can create/overwrite files)
# high — read + write + delete
PermissionLevel = Literal["low", "mid", "high"]

_PERMISSION_CAPS: dict[str, frozenset[str]] = {
    "low":  frozenset({"read"}),
    "mid":  frozenset({"read", "write"}),
    "high": frozenset({"read", "write", "delete"}),
}

GLOBAL_DIR = Path.home() / ".kratos"

# PROJECT_DIR is always relative to CWD at runtime (wherever `kratos` was invoked).
# Do NOT resolve at import time — this module is imported once, CWD is the project.
def _project_dir() -> Path:
    return Path.cwd() / ".kratos"

PROJECT_DIR = Path(".kratos")   # kept for backward-compat imports; use _project_dir() for new code

_KRATOS_INSTALL_DIR = Path(__file__).resolve().parents[1]  # C:\Tools\Kratos

# Abliterated GGUF — path relative to install dir, not CWD
_PLANNER_GGUF: Path = (
    _KRATOS_INSTALL_DIR
    / "models"
    / "Phi-4-mini-instruct-abliterated-Q5_K_M-GGUF"
    / "phi-4-mini-instruct-abliterated-q5_k_m.gguf"
)

# Planner: qwen3-abliterated:8b — chain-of-thought reasoning, structured plans
PLANNER_MODEL_NAME = "huihui_ai/qwen3-abliterated:8b"
# Coder: qwen3.5-abliterated:4B — code changes, fixes, refactoring; fits well in 6 GB VRAM
CODER_MODEL_NAME = "huihui_ai/qwen3.5-abliterated:4B"

# Fallback coder if primary not available
FALLBACK_CODER_MODEL = "huihui_ai/qwen2.5-coder-abliterate:7b"

# Local GGUF planner (Phi-4-mini-abliterated) — registered as kratos-planner
LOCAL_PLANNER_MODEL = "kratos-planner"
ALT_PLANNER_MODEL = "huihui_ai/qwen3-abliterated:8b"


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
    # Models
    planner_model: str = field(default_factory=lambda: PLANNER_MODEL_NAME)
    coder_model: str = field(default_factory=lambda: CODER_MODEL_NAME)
    planner_gguf_win: str = field(default_factory=_find_planner_gguf)
    # Session
    scope: Literal["global", "project"] = "project"
    goal: str | None = None
    # Permission level: low=read | mid=read+write | high=read+write+delete
    permission: PermissionLevel = "mid"
    # Ollama
    ollama_host: str = "http://localhost:11434"
    gpu_layers: int = 50       # let Ollama auto-cap at VRAM limit
    context_length: int = 4096
    # Temperatures
    planner_temp: float = 0.7
    coder_temp: float = 0.2
    # Context windows (num_ctx): planner 4096–8192, coder 8192–16384
    planner_num_ctx: int = 8192
    coder_num_ctx: int = 16384
    # Build / test commands (optional — enables diagnostic loop)
    build_cmd: str | None = None
    test_cmd: str | None = None
    build_test_retries: int = 3
    # Planner→Coder→Verify loop: keep going until VERIFIED or UNSOLVABLE
    max_verify_iterations: int = 10

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, scope: Literal["global", "project"] = "project") -> Path:
        # project dir is always CWD/.kratos — correct regardless of where kratos was installed
        target = GLOBAL_DIR if scope == "global" else _project_dir()
        target.mkdir(parents=True, exist_ok=True)
        cfg_file = target / "config.json"
        cfg_file.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return cfg_file

    @classmethod
    def load(cls) -> "KratosConfig":
        """Merge global + project config; project overrides global.

        Global config  (~/.kratos/config.json)   — machine-wide defaults.
        Project config (CWD/.kratos/config.json) — per-project overrides.
        CWD is wherever `kratos` was invoked, not the install directory.
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

    # ── permission helpers ────────────────────────────────────────────────────

    def can_read(self) -> bool:
        return True  # always

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
