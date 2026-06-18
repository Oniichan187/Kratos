#!/usr/bin/env python3
"""Kratos — abliterated 4-role CLI AI agent (max context, stepwise, lossless memory).

ALL models are abliterated (no safety filters).

Roles (each uses the best abliterated model for a laptop + its job):
  Planner    : huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M  — plans + CoT, FULL 64K ctx (VRAM-capped)
  Coder      : huihui_ai/qwen2.5-coder-abliterate:7b-instruct-q4_K_M   — implements, FULL 32K ctx (code-optimised)
  Verifier   : huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M  — strict PROVEN_WORK, FULL ctx
  Auto-Composer (Compressor): kratos-planner (Phi-4-mini-abliterated GGUF) — .kratos memory + history
                              NEVER destroys information, always max ctx.

Pipeline guarantees:
- Coder walks the plan ONE STEP AT A TIME: think how, show verify cmd, code it, runtime runs test,
  only then next step.
- Verifier only says VERIFIED after real tests executed per step + final sweep.
- Compressor (auto-composer) + Memory in .kratos are lossless (exhaustive facts, verbatim quotes).
- Every model call uses the MAXIMUM context window the model supports (within vram cap).
- Works for tiny tasks and for repos that massively exceed any single ctx (relay + memory + compress).

Usage:
    python kratos.py            # interactive REPL
    python kratos.py --setup    # model setup wizard
    python kratos.py --tui      # full-screen Textual TUI
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import rich  # noqa: F401
except ImportError:
    sys.exit("Install dependencies first:  pip install -r requirements.txt")


def _run_tui() -> None:
    import time as _t
    from kratos.config import KratosConfig as _KC, _project_dir as _pd
    from kratos.llm.bridge import OllamaBridge as _OB
    from kratos.core.agent import KratosAgent as _KA
    from kratos.logger import SessionLogger as _SL
    from kratos.prompts import load_prompts as _lp
    from kratos.app.tui import run_tui as _run

    _cfg    = _KC.load()
    _scope  = _cfg.scope or "project"
    _bridge = _OB(_cfg.ollama_host)
    _prm    = _lp()
    _agent  = _KA(_cfg, _bridge, prompts=_prm)
    _logger = _SL(_pd())
    _run(
        config=_cfg,
        bridge=_bridge,
        agent=_agent,
        logger=_logger,
        project_root=Path.cwd(),
        session_start=_t.time(),
        scope=_scope,
    )


if __name__ == "__main__":
    if "--setup" in sys.argv:
        import setup_models as sm
        sm.setup()
    elif "--tui" in sys.argv:
        _run_tui()
    else:
        from kratos.app.cli import main
        main()
