# Hardening Notes

This document summarizes the reliability mechanisms currently present in the
codebase.

## Import And UI Boundaries

`kratos/app/__init__.py` lazily resolves CLI and TUI exports. Importing helper
modules should not start the UI, import optional Textual/Rich dependencies, or
call `sys.exit`.

## Evidence-Only Reporting

`reporter.build_final_report` computes status from:

- disk snapshots;
- real file hashes;
- real command results;
- test flags and exit codes;
- deterministic diagnoses;
- git or snapshot diff data;
- real web research records.

The model cannot claim that tests passed or that files changed unless the tool
layer recorded matching evidence.

## Diffing

`kratos/execution/diffing.py` provides side-effect-free helpers for:

- unified diffs;
- added/removed line stats;
- git changed-file detection;
- git diff stat summaries.

Git helpers degrade cleanly outside a git repository.

## Generic Repair Loop

`kratos/execution/repair_loop.py` contains a project-agnostic repair loop that
accepts two callables:

- `run_tests()`
- `apply_fix(diagnosis, attempt)`

The loop succeeds only when a real command returns exit code `0`. It stops on
no-op fixes, repeated stalled signatures, or the configured attempt cap.

## Command Hardening

Command execution records stdout and stderr separately, captures real exit
codes, enforces timeouts, and reports blocked commands without running them.

The main build/test path and the general shell runner both call the central
safety guard.

## Tool Hardening

The tool layer includes:

- project-root path resolution for nested projects;
- search fallback when the model provides prose instead of exact regex;
- readback after failed edits;
- no-op write detection;
- explicit delete handling;
- guarded web fetch/search;
- structured observations for the model.

## Prompt Hardening

`kratos/prompts_default.json` declares marker names, tool usage rules,
prediction budgets, verification policy, and safe command prefixes. The prompt
bundle can be listed, reloaded, or dumped through `/prompts`.

Keeping these rules in JSON makes prompt changes reviewable without hiding them
inside Python control flow.
