# Agent Architecture

This document describes the current Kratos runtime as implemented in the source
tree. It is intentionally code-facing: module names below are the places to look
when changing behavior.

## Entry Points

| File | Role |
|---|---|
| `kratos.py` | Main launcher. Supports classic CLI, TUI, and setup mode. |
| `kratos.bat` | Windows launcher for the local checkout. |
| `kratos/app/cli.py` | Classic REPL, readiness checks, slash commands, live status. |
| `kratos/app/tui.py` | Textual UI entry point. |
| `setup_models.py` | Idempotent model and config setup. |
| `install.bat` / `setup_wsl.sh` | Windows and WSL/Ollama provisioning. |

`kratos/app/__init__.py` uses lazy exports so importing the package does not
eagerly import optional CLI/TUI dependencies or exit the interpreter.

## High-Level Pipeline

1. The CLI loads config, prompt JSON, history, memory, and the agent.
2. `Analyzer`, `TaskClassifier`, and `Router` inspect the request.
3. `ContextBuilder` creates a token-aware repository context.
4. `MemoryStore` and `KnowledgeBase` retrieve relevant prior facts and code
   chunks.
5. The selected route runs:
   - direct answer
   - clarification
   - planner-only answer
   - coder-only loop
   - planner-then-coder loop
   - diagnostic repair loop
6. Coder actions are parsed and executed by `kratos/execution/tools.py`.
7. Real verification commands populate `ProvenWork`.
8. `reporter.py` builds the final answer from evidence.
9. Successful work may write planner artifacts, memory, compression summaries,
   and knowledge chunks under `.kratos/`.

## Routing

Routing is split across three modules:

| Module | Responsibility |
|---|---|
| `kratos/analyzer.py` | Lightweight request analysis and metadata extraction. |
| `kratos/classifier.py` | Task kind classification. |
| `kratos/router.py` | Chooses the execution route. |

The main routes are:

| Route | Behavior |
|---|---|
| `direct_answer` | Search/read-only answer path. |
| `ask_clarification` | Used when acting would be risky without more detail. |
| `planner_only` | Planning or explanatory response without code changes. |
| `coder_only` | Direct tool loop without a separate plan. |
| `planner_then_coder` | Default for edits: plan, execute work steps, verify. |
| `diagnostic_loop` | Build/test failure repair path. |

## Planning And Coding

`kratos/core/agent.py` owns the orchestration. It mixes in support from:

- `kratos/core/runners.py` for model calls, trimming, compression, and logging.
- `kratos/core/buildtest.py` for command discovery and execution.
- `kratos/core/retry.py` for run state, planner artifacts, and success memory.

Planner prompts are built in `kratos/roles/planner.py`. Coder behavior is in
`kratos/roles/coder.py`.

There are two coder paths:

| Driver | Current role |
|---|---|
| `execute_structured_work_steps_for_plan` | Default planner-driven execution. Each checklist item gets bounded micro-turns, read/search/edit/verify cycles, status events, and failure readback. |
| `run_coder_loop` | Adaptive observe-act loop retained for direct or fallback use. |

The structured driver marks items as complete only when there is enough evidence
for that item, such as a successful targeted verification command.

## Context

`kratos/context/indexer.py` lists files while ignoring heavy or unsafe folders
such as `.git`, `.kratos`, `.claude`, `node_modules`, virtualenvs, build output,
and model directories.

`kratos/context/builder.py` builds a token-aware prompt context:

- all known files are listed up to a practical cap;
- likely relevant files are prioritized;
- content excerpts are capped by token budget;
- secret-looking files are skipped;
- nested project paths are handled.

## Memory

`kratos/memory.py` provides four practical scopes:

- session
- task
- project
- long-term/global

Project memory is stored at `.kratos/memory.json`. Global memory is stored at
`~/.kratos/memory.json`. The store filters secret-looking values and records
categories such as decisions, file roles, error causes, solutions, conventions,
and todos.

## Knowledge

`kratos/knowledge/base.py` provides optional retrieval over repository chunks.
It uses Ollama embeddings through `nomic-embed-text`.

Storage behavior:

- LanceDB is used when installed.
- A JSON fallback index is used when LanceDB is unavailable.
- The index includes symbol-centric chunks, fallback text chunks, and available
  compression artifacts.
- Empty indexes are rebuilt automatically when useful.

The knowledge layer is intentionally optional. Failure to build or query it
should degrade the answer, not crash the run.

## Model Calls And Compression

`kratos/core/runners.py` prepares model payloads, estimates token budgets, logs
payloads when logging is enabled, and compresses history when the conversation
is too large.

`kratos/llm/tokens.py` owns model context maxima and context selection. With the
current defaults, role windows are larger than the default 8192 VRAM ceiling, so
planner/coder/verifier calls are effectively capped for local reliability.

`kratos/compress.py` provides compression, relay, and memory extraction helpers.
Algorithmic fallbacks are used when the model call fails.

## Prompt System

Most role instructions and marker definitions live in `kratos/prompts_default.json`.
`kratos/prompts.py` loads, validates, reloads, and dumps prompt configuration.

The `/prompts` slash command can list sections, reload the prompt JSON, or dump
the active prompt bundle for inspection.

## Evidence And Reporting

`kratos/verification.py` defines `ProvenWork` and discovers verification
commands. `kratos/reporter.py` turns evidence into the final report.

Important rule: the final status is computed from structured evidence, never
from free-form model claims. The report can include only:

- real file changes from snapshots and disk comparison;
- real command results with exit codes;
- real test status;
- deterministic diagnoses from command output;
- real web sources recorded during the current run;
- real diff stats from git or snapshot comparison.
