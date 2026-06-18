# Analysis And Rebuild Notes

This file replaces older session-specific notes with the current state of the
repository.

## What Kratos Is

Kratos is a local coding agent with a strict evidence loop:

1. classify the request;
2. build context;
3. retrieve memory and optional knowledge chunks;
4. plan the work when needed;
5. execute file/search/shell/web tools through parsed markers;
6. run real verification commands;
7. diagnose failures;
8. report only what the evidence supports.

The main reliability goal is to make a local model useful without allowing it to
fake success.

## Current Core Modules

| Area | Modules |
|---|---|
| UI and entry points | `kratos.py`, `kratos/app/cli.py`, `kratos/app/tui.py`, `kratos/app/slash.py` |
| Config and prompts | `kratos/config.py`, `kratos/prompts.py`, `kratos/prompts_default.json` |
| Routing | `kratos/analyzer.py`, `kratos/classifier.py`, `kratos/router.py` |
| Orchestration | `kratos/core/agent.py`, `kratos/core/runners.py`, `kratos/core/retry.py`, `kratos/core/buildtest.py` |
| Roles | `kratos/roles/planner.py`, `kratos/roles/coder.py`, `kratos/roles/verifier.py` |
| Tools | `kratos/execution/tools.py`, `kratos/execution/search.py`, `kratos/execution/edits.py`, `kratos/execution/shell.py` |
| Verification | `kratos/verification.py`, `kratos/reporter.py`, `kratos/execution/testguard.py`, `kratos/execution/diagnostics.py` |
| Memory and retrieval | `kratos/memory.py`, `kratos/context/*`, `kratos/knowledge/base.py`, `kratos/compress.py` |
| Safety | `kratos/safety.py`, `kratos/web.py` |

## Current Rebuild Outcome

The architecture now has:

- lazy app imports;
- guarded command execution;
- guarded project-root file writes;
- search/replace edits for existing files;
- smart search and read-range tools;
- real test and build command discovery;
- deterministic diagnostics;
- no-progress detection;
- protected pre-existing tests;
- evidence-only final reports;
- optional vector knowledge with fallback storage;
- prompt rules externalized to JSON.

## Remaining Engineering Notes

These are not blockers, but they are the areas to keep an eye on:

- `core/buildtest.py` and `execution/shell.py` both execute commands with safety
  checks. Consolidating them would reduce duplication.
- The package still contains both root docs and package-local docs. The
  package-local files now point to the root docs to avoid drift.
- The default model names are local Ollama names. Install and setup scripts
  should stay in sync with `kratos/config.py`.
- UI help and autocomplete should be kept aligned with `kratos/commands.py`
  whenever slash commands change.

## How To Validate Future Changes

Useful checks after code changes:

```powershell
python -m compileall kratos
python -m pytest
```

For documentation-only changes, scan for old context-window claims and encoding
damage. A simple ASCII scan is useful because these docs are intentionally plain
ASCII:

```powershell
rg -n "[^ -~\t\r\n]" README.md docs kratos/docs
```
