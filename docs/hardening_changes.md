# Kratos — Hardening Notes (Architecture · Verification · Tools · Safety)

Concise reference for the reliability work described in
`docs/analysis_rebuild_plan.md`. Everything here is covered by tests in
`kratos/tests/`.

## Architecture

- **`kratos/execution/diffing.py` (new).** Model-free change detection:
  `diff_stats` (accurate `+added/-removed` via `difflib.ndiff`),
  `unified_diff` (real unified diff), `git_changed_files`
  (`git diff --name-only HEAD` + untracked), `git_diff_stat`
  (`git diff HEAD --stat`). All side-effect free; git helpers return empty on a
  non-repo and never raise.
- **`kratos/execution/repair_loop.py` (new).** A generic, project-agnostic
  `run_repair_loop(run_tests, apply_fix, …)` (the old `execution/repair.py`
  only hardcoded fixes for two bundled probes). It is driven by real
  `CommandResult`s and the diagnostics engine; see *Repair loop* below.
- **`kratos/app/__init__.py` — lazy shells.** Importing the package (or a light
  submodule such as `app.prompt_frame`) no longer eagerly imports the CLI/TUI
  and therefore no longer risks a `sys.exit` at import time when `rich` /
  `prompt_toolkit` / `textual` are absent. `main` / `run_tui` /
  `slash_completions` resolve lazily on first attribute access.

## Verification (the hard gate)

`reporter.build_final_report` computes status from evidence, never from model
text:

- **No SUCCESS without real changes** when the task required code changes —
  reality is recomputed from on-disk snapshots (`verify_files_changed`),
  no-op writes (identical content) count as `unchanged`.
- **No SUCCESS on failed/locally-missing tests**; **no "tests passed" when no
  test ran** — test status comes only from recorded commands carrying
  `is_test` + `exit_code`. The report now always prints the canonical
  `Tests nicht ausgeführt` when nothing ran.
- **No invented diff** — the diff summary comes from `git diff HEAD --stat`
  (staged + unstaged) or the real snapshot comparison; otherwise it says
  `Kein Diff vorhanden.`
- **Technical `files_changed`** — derived from real write/delete operations and
  hash comparison, with an independent `git_changed_files` cross-check surfaced
  in the report. Line counts use `difflib`, not coarse length differences.

## Tool usage (coder action markers)

The coder emits plain-text markers parsed by `execution/tools.py`:
`### READ`, `### READ_RANGE: path:a-b`, `### SEARCH text [:: glob]`,
`### GREP regex`, `### GLOB pattern`, `### FILE: path` + fenced body,
`### DELETE: path`, `### WEB_FETCH url`, `### WEB_SEARCH query`,
`### INSPECT cmd` (read-only), `### VERIFY`/`### RUN cmd`, `### DONE`.
Writes are applied before verify commands; every file write records sha256
before/after and real line deltas into `ProvenWork`.

`web.py` exposes a provider-adapter `web_search`: the keyless DuckDuckGo HTML
endpoint by default; any other provider returns the honest
`Web search provider '<x>' not configured` instead of fabricated results. A
failed fetch returns `web search unavailable: …`. Real sources are recorded to
`.kratos/research.jsonl` and read back by the reporter — never claimed.

## Safety

- `safety.check_command` blocks destructive / download-and-execute / credential
  patterns; `safety.check_path` confines writes to the project root and blocks
  git internals. Both are consulted by every shell and file surface.
- Command execution (`core/buildtest.py`, `execution/shell.py:ShellRunner`)
  applies the SafetyGuard as defense-in-depth, enforces a hard timeout
  (exit 124, `timed_out=True`), captures **separate** stdout/stderr and the real
  exit code, and returns a structured `CommandResult` (cmd, shell, cwd,
  duration, timeout, blocked/reason). Blocked commands return exit 126 and are
  never executed.

## Repair loop

`run_repair_loop(run_tests, apply_fix, max_attempts=3, stall_threshold=2)`:

1. `run_tests()` → real `CommandResult`. `exit_code == 0` ⇒ success (the only
   way success is ever reported).
2. Otherwise `diagnostics.diagnose_command` categorises the real output into a
   `Diagnosis` (`category`, `signature`, `fix_instruction`).
3. `apply_fix(diagnosis, attempt)` performs a targeted change and returns
   whether anything changed.
4. Re-test. The loop stops on success, a **stalled** signature (same failure
   repeating ⇒ the last fix did nothing useful), a no-op fix, or `max_attempts`
   — never spinning, never claiming an unproven repair.
