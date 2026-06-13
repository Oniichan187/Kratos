# Kratos — Technical Analysis & Rebuild Plan

_Authored during a full systematic review of the Kratos local coding agent.
Every finding below is backed by reading the real source files and running the
test suite; nothing here is inferred from model text._

## 1. Architecture overview (as built)

Kratos is a local, Ollama-backed ReAct coding agent aimed at weak / abliterated
models. Layering:

- `kratos/app/` — CLI + Textual TUI front-ends (require `rich`, `prompt_toolkit`,
  `textual`).
- `kratos/core/agent.py` — the orchestration loop (plan → code → verify → report).
  Mixed in from `core/runners.py`, `core/retry.py`, `core/buildtest.py`.
- `kratos/roles/` — planner / coder / verifier prompt builders + the adaptive
  coder loop (`roles/coder.py`).
- `kratos/execution/` — the tool layer the coder actually drives:
  `tools.py` (marker parsing + file/command handlers), `search.py` (pattern/regex/
  range reads), `shell.py` (`ShellRunner`), `parsing.py`, `diagnostics.py`
  (failure analysis), `repair.py`, `circular.py`, `testguard.py`.
- `kratos/safety.py` — central command/path gate (`check_command`, `check_path`).
- `kratos/verification.py` — command discovery + `ProvenWork` evidence model.
- `kratos/reporter.py` — the anti-hallucination final-report gate.
- `kratos/web.py` — stdlib HTTP fetch / HTML scrape / provider web-search.

The honesty-critical path is: tool handlers record **real** evidence into
`ProvenWork` (files written to disk, real `CommandResult`s) → `reporter.py`
recomputes file-change reality from disk snapshots and derives test status only
from recorded commands → status is computed, never claimed.

## 2. Findings

### 2a. Real on-disk bugs

1. **`execution/search.py` — `smart_search` is truncated.** The final fallthrough
   branch is the bare token `ret` (an undefined name). For any short pattern with
   no matches it raises `NameError` instead of returning a result. **Severity: high**
   (crashes a core tool). Fix: return an honest empty result `([], "no matches…")`.

2. **`reporter.py` — `to_markdown()` never prints `TESTS_NOT_RUN_MSG`.** When no
   test ran, the Tests section said only "Grund: …" and omitted the canonical
   "Tests nicht ausgeführt". This fails the project's own gate test
   (`test_reporter_no_test_claim_without_test_run`) and weakens the honesty
   guarantee required by the spec. Fix: emit the canonical message.

3. **`app/cli.py` — `sys.exit()` at import time.** When `rich`/`prompt_toolkit`/
   `textual` are missing, `import kratos.app` (eagerly importing `.cli`) calls
   `sys.exit(...)`, which kills the interpreter for *any* programmatic or test
   import of the package. Fix: degrade gracefully on import; only exit when the
   CLI is actually launched.

4. **HEAD `core/agent.py` used a Python‑3.12-only f-string** (backslash inside an
   f-string expression, `f"{x}\n\n"`), which is a `SyntaxError` on 3.10/3.11.
   Already resolved in the working tree; recorded here as a portability guard.

### 2b. Architecture / design issues

5. **`execution/repair.py` is not a repair loop.** It is ~715 lines of hardcoded
   fixes for two specific probe projects (`try_repair_taskboard_probe`,
   `try_repair_inventory_probe`). Requirement: a *generic* loop that analyses a
   real test failure, requests a targeted fix, and retests. The analysis half
   already exists (`diagnostics.diagnose_command`, `RepairTracker`); the generic
   orchestration is missing. Fix: add `run_repair_loop(run_tests, apply_fix, …)`.

6. **Two parallel command-execution surfaces.** `execution/shell.py:ShellRunner`
   is hardened (SafetyGuard gate, split stdout/stderr, timeout→124, blocked→126,
   structured logging) but is **not** on the main verify path. The path actually
   used is `core/buildtest.py._run_verification_command` (`subprocess.run`). Both
   apply the SafetyGuard and capture structured results, so this is duplication
   rather than a safety hole — recommend consolidating onto `ShellRunner`.

7. **`reporter._git_diff_stat` uses `git diff --stat`** (working-tree only) and
   the changed-file *set* is seeded from `proof.files_changed`. Hardening:
   capture `git diff HEAD` (staged + unstaged) and cross-check the changed set
   against `git diff --name-only HEAD` so real changes can't be missed and none
   can be invented.

8. **`do_write` reports coarse line-count deltas, not a real diff.** Requirement
   asks for genuine diff detection. Fix: compute a real unified diff (`difflib`)
   for accurate +/- counts and a human-readable summary.

### 2c. Subsystems already in good shape (verified)

- `safety.py`, `execution/shell.py` — solid; 81 + targeted tests pass.
- `web.py` — clean provider adapter, honest "not configured"/"unavailable"
  errors, SSRF guard, size/redirect caps, real-source recording. (Note: the
  `_MAX_REDIRECTS` constant is dead — redirects are capped by urllib's default.)
- `verification.py`, `reporter.py` — strong evidence-based gate (after fix #2).

### 2d. Environment note (not a project bug)

In this review sandbox the Linux mount exposed stale/truncated copies of several
pre-existing files; the authoritative Windows files were read directly. All
testing was done against a reconstructed, consistent copy of the real sources.

## 3. Rebuild plan (executed in this session)

| # | Change | Requirement |
|---|--------|-------------|
| 1 | Fix `search.py` `ret` truncation | reliability |
| 2 | `reporter.to_markdown` emits `TESTS_NOT_RUN_MSG` | hard verify-gate |
| 3 | `cli.py` import-safe (no `sys.exit` on import) | reliability |
| 4 | New `execution/diffing.py`: `unified_diff`, `diff_stats`, `git_changed_files` | file diff / files_changed |
| 5 | `reporter`: `git diff HEAD`, git-name-only cross-check, difflib line counts | verify-gate / files_changed |
| 6 | New generic `run_repair_loop` in `repair.py` on top of the diagnostics engine | repair loop |
| 7 | New tests for diffing, git-changed detection, repair loop, cli import | tests |
| 8 | Docs: this file + verification / tools / safety updates | documentation |

All changes are verified by running the test suite; results are recorded in the
final report.
