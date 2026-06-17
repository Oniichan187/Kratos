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
#   Coder     : dedicated code model, abliterated — best coding quality (32k native)
#   Verifier  : same strong reasoning as planner (strict PROVEN_WORK judge)
#   Compressor: tiny but faithful summarizer/memory extractor (Phi-4-mini ablit)
PLANNER_MODEL_NAME  = "huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M"
CODER_MODEL_NAME    = "huihui_ai/qwen2.5-coder-abliterate:7b-instruct-q4_K_M"
VERIFIER_MODEL_NAME = "huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M"
COMPRESSOR_MODEL    = "kratos-planner"         # Phi-4-mini-instruct-abliterated GGUF
FALLBACK_CODER_MODEL = "huihui_ai/qwen3.5-abliterated:4b"
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
    planner_num_ctx:    int = 131072   # full max for deepseek-r1-abliterated:8b-0528-qwen3 (Qwen3-8B base, 128k native)
    coder_num_ctx:      int = 32768    # full max for huihui_ai/qwen2.5-coder-abliterate:7b (verified)
    verifier_num_ctx:   int = 131072   # full max — same R1 model as planner
    compressor_num_ctx: int = 32768    # generous for kratos-planner (Phi-4-mini-abliterated). Use its real max when known
    relay_num_ctx:      int = 131072   # coder relay gets big window for giant inputs before planner sees them

    # VRAM ceiling: hard cap on num_ctx for every role, even if the model
    # advertises more. THIS IS THE SINGLE MOST IMPORTANT PERFORMANCE KNOB.
    #
    # Ollama allocates the FULL KV cache for num_ctx up front when the model
    # loads — it does NOT "only materialize used pages". For an 8B model the
    # KV cache costs ~0.14 MB/token, so:
    #     8192 ctx  -> ~1.2 GB KV   (fits a 6 GB card next to ~4.9 GB weights)
    #    65536 ctx  -> ~9.4 GB KV   (does NOT fit -> Ollama offloads to system
    #                                RAM -> generation collapses to ~3 tok/s,
    #                                i.e. the 10-minute planner stall).
    # Keep this at 8192 on a 6 GB laptop (RTX 4050 class). Raise to 16384/32768
    # only on bigger cards. Huge inputs are handled by the relay + compression
    # layers, NOT by a giant window.
    vram_ctx_ceiling: int = 8192

    # ── Temperatures ──────────────────────────────────────────────────────────
    # DeepSeek-R1 reasoning models must run at LOW temperature (0.1).
    # At 0.6 the model enters endlessly circular <think> chains and exhausts
    # the full predict budget without ever writing a single plan line (~22 min
    # observed on this hardware for a trivial task).  0.1 → focused, fast output.
    planner_temp:    float = 0.1
    coder_temp:      float = 0.15
    verifier_temp:   float = 0.1
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

    # ── Vector Knowledge Base / continuous "gets" (best-possible retrieval) ──
    # embed_model: small Ollama embed model (runs in same WSL as the coding models).
    # enable_semantic_retrieval: master switch for the vector DB layer.
    # retrieval_top_k: how many high-signal chunks to pull per "get".
    embed_model: str = "nomic-embed-text"
    enable_semantic_retrieval: bool = True
    retrieval_top_k: int = 16
    # knowledge_backend left for future (lancedb is the current implementation)

    # ── Build / test / verify ─────────────────────────────────────────────────
    build_cmd:             str | None = None
    test_cmd:              str | None = None
    build_test_retries:    int = 3
    # Verify-revision budget. <= 0 means UNBOUNDED: keep planning/coding/verifying
    # until the real tests pass (Kratos is intentionally NOT capped by an iteration
    # count — a converging run still finishes in a few iterations).
    max_verify_iterations: int = 0
    # Hard safety net for an UNBOUNDED loop: give up only when CLEARLY stuck —
    # the *identical* failure signature has repeated this many times with no
    # progress. 0 disables it (truly endless). Generous on purpose.
    no_progress_abort:     int = 40
    auto_discover_verification: bool = True
    require_proven_work:        bool = True
    require_test_for_verified:  bool = True
    # Deterministic verification: the pipeline ALREADY runs the real tests and
    # gates on their exit code (PROVEN_WORK) before any LLM verifier call. When
    # this is True, a satisfied PROVEN_WORK (real tests passed) is accepted
    # directly and the redundant LLM-verifier model swap is skipped. This makes
    # the loop a tight  planner -> coder -> run tests -> (pass? done : redo),
    # which is both faster (no extra 8B reload per iteration) and more honest
    # (the test exit code is the ground truth, not a second model's opinion).
    # Set False to restore the old "LLM judges PROVEN_WORK" behavior.
    deterministic_verify:       bool = True
    verification_timeout_seconds: int = 120
    # Repair-loop stall guard: after this many identical failure signatures the
    # agent escalates the diagnosis instead of re-running the same dead end
    # (motivated by the 94×-identical-pytest-failure session 2026-06-13).
    repair_stall_threshold:     int = 2
    # Restore pre-existing test files before the authoritative verification so a
    # weak model cannot make tests pass by weakening them (it may still add new
    # tests). A green result then means the ORIGINAL tests pass.
    protect_existing_tests:     bool = True

    # ── Adaptive ReAct coder action-loop ─────────────────────────────────────
    # coder_loop: when True (default), the coder runs as an observe->act loop
    # (writes/reads/runs commands, ingests real results, iterates until a test
    # passes and it signals ### DONE) instead of the rigid per-step driver.
    # max_coder_iterations bounds that loop so a non-converging model still
    # falls through to the existing verifier + outer retry. Set to <= 0 for
    # an unbounded loop that ends only on ### DONE / cancel / outer abort.
    coder_loop:            bool = True
    max_coder_iterations:  int = 0
    # Max model micro-turns per structured work-step checklist item before the
    # driver moves on (the outer verify loop will retry unfinished items).
    max_work_step_turns:   int = 4

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
                from .llm.tokens import model_max_ctx
                inst.planner_num_ctx = max(inst.planner_num_ctx, model_max_ctx(inst.planner_model))
                inst.coder_num_ctx = max(inst.coder_num_ctx, model_max_ctx(inst.coder_model))
                inst.verifier_num_ctx = max(inst.verifier_num_ctx, model_max_ctx(inst.verifier_model))
                inst.compressor_num_ctx = max(inst.compressor_num_ctx, model_max_ctx(inst.compressor_model))
                inst.relay_num_ctx = max(inst.relay_num_ctx, 65536)
            except Exception:
                pass
        # Legacy upgrade: the old hard cap of 10 becomes UNBOUNDED so existing
        # saved configs also stop giving up early (set a positive value to re-cap).
        if getattr(inst, "max_verify_iterations", 0) == 10:
            inst.max_verify_iterations = 0

        # ── Hardware safety net (prevents the 10-minute planner stall) ─────────
        # A config saved by an earlier version may still carry vram_ctx_ceiling
        # = 65536. On a 6-8 GB laptop that forces the 8B model's KV cache out of
        # VRAM into system RAM and generation collapses to ~3 tok/s (the planner
        # then "thinks" for 10+ minutes and never emits a plan). Clamp clearly
        # impossible ceilings back to a value that actually fits. 8192-16384 are
        # left untouched so users with bigger GPUs keep their explicit choice.
        try:
            if int(inst.vram_ctx_ceiling) > 16384:
                inst.vram_ctx_ceiling = 8192
        except (TypeError, ValueError):
            inst.vram_ctx_ceiling = 8192

        # DeepSeek-R1 / reasoning models spin in endless <think> loops at high
        # temperature (0.6 -> ~22 min, no plan). Clamp a stale saved temperature
        # so the planner/verifier stay focused and fast.
        for _m_attr, _t_attr in (("planner_model", "planner_temp"),
                                  ("verifier_model", "verifier_temp")):
            try:
                if "r1" in str(getattr(inst, _m_attr, "")).lower():
                    setattr(inst, _t_attr, min(float(getattr(inst, _t_attr)), 0.3))
            except (TypeError, ValueError):
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
