"""Centralized, JSON-driven prompt store for Kratos.

ALL prompt text lives in prompts_default.json (shipped with the package).
This file contains ONLY loading/merging machinery — zero hardcoded prompt strings.

To customize prompts:
  1. Run /prompts dump  →  writes .kratos/prompts.json with all defaults
  2. Edit .kratos/prompts.json (or ~/.kratos/prompts.json for machine-wide)
  3. Run /prompts reload  →  picks up your changes immediately

Load order (later wins, partial key overrides are OK):
  kratos/prompts_default.json  →  ~/.kratos/prompts.json  →  ./.kratos/prompts.json

Usage in code:
    from .prompts import load_prompts, get_system, get_snippet, get_predict, get_marker
    system  = get_system("coder")
    label   = get_snippet("task_label")
    n_toks  = get_predict("code")
    marker  = get_marker("file")
    # After editing JSON on disk:
    reload_prompts()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import GLOBAL_DIR, _project_dir

# Path to the shipped defaults — zero prompt text lives in this .py file.
_DEFAULT_JSON: Path = Path(__file__).with_name("prompts_default.json")


def _load_json_safe(path: Path) -> dict:
    """Load a JSON file; return {} on any error (missing, malformed, wrong type)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _deep_update(base: dict, override: dict) -> dict:
    """Deep-merge override into a copy of base (one level deep for dict values)."""
    result = dict(base)
    for k, v in override.items():
        if k.startswith("_"):
            continue  # skip comment keys
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = {**result[k], **v}
        else:
            result[k] = v
    return result


# Module-level cache: loaded once, reloaded on demand via reload_prompts().
# DEFAULT_PROMPTS is exposed for tests and /prompts dump — populated from JSON.
DEFAULT_PROMPTS: dict[str, Any] = {}


def _load_defaults() -> dict[str, Any]:
    """Load the shipped prompts_default.json. Raises on missing/corrupt file."""
    data = _load_json_safe(_DEFAULT_JSON)
    if not data:
        raise RuntimeError(
            f"Cannot load Kratos default prompts from {_DEFAULT_JSON}. "
            "The file is missing or corrupt — reinstall Kratos."
        )
    return data


# Populate DEFAULT_PROMPTS at import time so it is available immediately.
try:
    DEFAULT_PROMPTS = _load_defaults()
except Exception:
    DEFAULT_PROMPTS = {}


class PromptManager:
    """Loads defaults from JSON + optional global/project JSON overrides.

    All prompt text is sourced exclusively from JSON files.
    Supports reload() (for /prompts reload) and dump_defaults() (for /prompts dump).
    """

    def __init__(self) -> None:
        self._default_path  = _DEFAULT_JSON
        self._global_path   = GLOBAL_DIR / "prompts.json"
        self._project_path  = _project_dir() / "prompts.json"
        self._effective: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        base = _load_json_safe(self._default_path)
        if not base:
            # Fallback to already-loaded module default if file vanishes at runtime
            base = dict(DEFAULT_PROMPTS)
        merged = base
        for override_path in (self._global_path, self._project_path):
            override = _load_json_safe(override_path)
            if override:
                merged = _deep_update(merged, override)
        self._effective = merged

    def reload(self) -> None:
        """Re-read all JSON files from disk (call after editing any prompts JSON)."""
        self._load_all()

    def get_system(self, role: str) -> str:
        """Return the full system prompt for a role."""
        key = {
            "planner":          "planner_system",
            "verifier":         "verifier_system",
            "coder":            "coder_system",
            "relay":            "relay_system",
            "relay_detailed":   "relay_system_detailed",
            "compress":         "compress_system",
            "memory":           "memory_system",
            "compressor_model": "compressor_model_system",
        }.get(role, role)
        return self._effective.get(key, self._effective.get(role, ""))

    def get_snippet(self, key: str) -> str:
        """Return a small instruction/label/header snippet."""
        return self._effective.get("snippets", {}).get(key, "")

    def get_predict(self, key: str) -> int:
        return int(self._effective.get("predict", {}).get(key, 1024))

    def get_marker(self, key: str) -> str:
        return self._effective.get("markers", {}).get(key, key)

    def get_toolchain(self, key: str, default=None):
        """Return a value from the toolchain config section (e.g. safe_verify_prefixes)."""
        return self._effective.get("toolchain", {}).get(key, default)

    def get_plan_config(self, key: str, default=None):
        """Return a value from the plan config section (e.g. step_regex)."""
        return self._effective.get("plan", {}).get(key, default)

    def get_all(self) -> dict[str, Any]:
        """Return a copy of the effective prompts (for /prompts list or debugging)."""
        return dict(self._effective)

    def dump_defaults(self, target: Path | str) -> Path:
        """Write the shipped defaults to target as pretty JSON for user editing."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        defaults = _load_json_safe(self._default_path) or dict(DEFAULT_PROMPTS)
        target.write_text(
            json.dumps(defaults, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


# ── module-level cache ────────────────────────────────────────────────────────

_MANAGER: PromptManager | None = None


def load_prompts() -> PromptManager:
    """Return (and cache) the effective PromptManager. Safe to call many times."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = PromptManager()
    return _MANAGER


def get_system(role: str) -> str:
    return load_prompts().get_system(role)


def get_snippet(key: str) -> str:
    return load_prompts().get_snippet(key)


def get_predict(key: str) -> int:
    return load_prompts().get_predict(key)


def get_marker(key: str) -> str:
    return load_prompts().get_marker(key)


def get_toolchain(key: str, default=None):
    return load_prompts().get_toolchain(key, default)


def get_plan_config(key: str, default=None):
    return load_prompts().get_plan_config(key, default)


def reload_prompts() -> None:
    """Force re-read of all JSON files from disk (for /prompts reload)."""
    load_prompts().reload()
    # Also refresh the module-level DEFAULT_PROMPTS cache
    global DEFAULT_PROMPTS
    try:
        DEFAULT_PROMPTS = _load_defaults()
    except Exception:
        pass
