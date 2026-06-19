# Kratos

Kratos is a local Ollama-backed coding agent for Windows and WSL. It reads a
project, classifies the request, builds a focused context, plans the work,
edits files through explicit tool markers, runs real build/test commands, and
finishes with an evidence-based report.

The project is optimized for local models that need strong scaffolding: command
and file operations are gated, verification is derived from real command output,
and final success is not taken from model text.

## Current Defaults

The checked-in defaults are conservative for a 6 GB VRAM laptop. Kratos requests
the largest useful context window, then caps planner, coder, and verifier calls
with `vram_ctx_ceiling`.

| Role | Default model | Configured role window |
|---|---|---:|
| Planner | `huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M` | 131072 |
| Coder | `huihui_ai/qwen2.5-coder-abliterate:7b-instruct-q4_K_M` | 32768 |
| Verifier | `huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M` | 131072 |
| Compressor | `kratos-planner` | 32768 |
| Relay | planner model | 131072 |
| Embeddings | `nomic-embed-text` | model default |

Important effective settings:

- `vram_ctx_ceiling`: `8192`
- `always_max_ctx`: `true`
- `deterministic_verify`: `true`
- `max_verify_iterations`: `0` means unbounded until success, no progress, or a
  hard safety stop.
- `no_progress_abort`: `40`
- `protect_existing_tests`: `true`

With these defaults, planner/coder/verifier calls are capped at 8192 tokens even
though the role and model maxima are larger. Compressor and relay windows use
their configured values.

## Install

Run the Windows installer from an elevated shell:

```powershell
.\install.bat
```

The installer checks or installs WSL2 Ubuntu, checks NVIDIA CUDA visibility,
installs Python requirements, provisions Ollama in WSL, pulls the configured
models, and writes the local config.

Manual setup is also available:

```powershell
pip install -r requirements.txt
python setup_models.py
```

`setup_models.py` is idempotent. It starts Ollama when needed, ensures the
planner/coder/verifier and embedding models exist, creates `kratos-planner` from
a local GGUF when available, and saves the current configuration.

## Run

From the Kratos checkout:

```powershell
python kratos.py
python kratos.py --tui
python kratos.py --setup
```

After `install.bat`, the `kratos.bat` launcher can be used from the install
location.

When Kratos starts, the classic CLI checks Ollama, checks the configured models,
loads config and prompt JSON, then opens a REPL. The TUI uses the same agent
pipeline with a Textual interface.

A live status bar is pinned to the bottom of the CLI. It shows the active role's
context window, elapsed time, and a single-line **live todo**: the plan label
(`PLAN 3/8`) plus only the one checklist item currently being worked on — the
first not-yet-done item — which advances on its own as items complete. File
operations are printed once, from the authoritative tool event with byte and
line deltas (for example `write_file('app.py') -> 590 bytes (+6 -0 lines)`).

## Request Flow

Kratos is not a single prompt wrapper. A normal coding request moves through
these stages:

1. `analyzer.py`, `classifier.py`, and `router.py` classify the request.
2. `context/builder.py` indexes the repository and builds a token-aware context.
3. `memory.py` and `knowledge/base.py` add relevant prior notes and optional
   vector-search chunks.
4. The planner creates a checklist for multi-step work.
5. The coder executes each work step through explicit markers such as
   `### READ`, `### EDIT`, and `### VERIFY`.
6. Build and test commands are discovered from user commands, README snippets,
   and project toolchain inference.
7. Deterministic verification accepts only real passing command output by
   default. The LLM verifier is used only when deterministic verification is
   disabled or additional judgment is required.
8. `reporter.py` writes a final report from structured evidence: changed files,
   command results, test status, diagnostics, diff stats, and real web sources.

Common routes:

| Route | Used for |
|---|---|
| `direct_answer` | File/code search or explanation that does not need edits |
| `ask_clarification` | Requests too ambiguous to act safely |
| `planner_only` | Planning, explanation, or non-editing analysis |
| `coder_only` | Direct shell/file follow-up work |
| `planner_then_coder` | Normal feature, bugfix, refactor, docs, config, and dependency work |
| `diagnostic_loop` | Build/test failures and repair requests |

## Slash Commands

Inside the CLI:

| Command | Purpose |
|---|---|
| `/help` | Show built-in command help |
| `/exit`, `/quit`, `/q` | Leave the REPL |
| `/clear` | Clear the terminal |
| `/goal [clear|text]` | Set or clear the session goal |
| `/scope [info|global|project]` | Show or change memory scope |
| `/permission [low|mid|high]` | Change command permission profile |
| `/models [planner|coder|verifier|compressor <name>]` | Show or change model names |
| `/prompts [list|reload|dump [path]]` | Inspect or reload prompt JSON |
| `/status` | Show config, model, memory, and verification status |
| `/history clear` | Clear session history |
| `/setup` | Run model setup |
| `/index [rebuild]` | Build or rebuild the project index |
| `/knowledge [status|rebuild [force]]` | Manage the optional knowledge index |
| `/memory [list|clear session|project|all]` | Inspect or clear memory |
| `/build [clear|cmd]` | Set or clear the build command |
| `/test [clear|cmd]` | Set or clear the test command |
| `/logging [status|on|off]` | Manage session payload logging |
| `/tokens` | Show context and token budget information |

## Tool Markers

The coder never receives direct filesystem or shell access. It emits markers
that `kratos/execution/tools.py` parses and executes:

| Marker | Purpose |
|---|---|
| `### READ: path` | Read a file |
| `### READ_RANGE: path:start-end` | Read exact line range |
| `### SEARCH: text [:: glob[, glob…]]` | Smart literal / `a\|b` / regex / keyword search, optional multi-glob scope |
| `### GREP: regex [:: glob[, glob…]]` | Regex search (rg-style `-n "pat" path` accepted) with smart fallback |
| `### GLOB: pattern[, pattern…]` | List files by name; comma/pipe-separated patterns are OR-ed (`rg -g a -g b` / `Get-ChildItem -Include *.py,*.md` style) |
| `### INSPECT: command` | Run read-only inspection command |
| `### FILE: path` | Create or replace a small file |
| `### EDIT: path` | Apply Aider-style search/replace edits |
| `### DELETE: path` | Delete a file inside the project |
| `### VERIFY` / `### RUN: command` | Run a guarded verification command |
| `### WEB_FETCH: url` | Fetch an HTTP(S) page with SSRF and size guards |
| `### WEB_SEARCH: query` | Run DuckDuckGo HTML search and record sources |
| `### DONE` | Mark the work step complete |

For existing files, `### EDIT` is preferred over whole-file rewrites. A failed
search block leaves the file untouched and tells the model to re-read the file.

## Verification

Kratos uses a hard evidence gate:

- No success when a code-changing task produced no real file changes.
- No success when tests failed.
- No "tests passed" claim when no test command actually ran.
- No fabricated web sources. Only sources recorded in `.kratos/research.jsonl`
  during the current run can appear in the report.
- Pre-existing tests are snapshotted and restored before authoritative
  verification when `protect_existing_tests` is enabled.

Verification commands can come from:

- `/build` and `/test`
- explicit commands in the user request
- README command snippets
- inferred toolchains for Python, Node, .NET, Cargo, Go, Maven, and Gradle

Shell commands still pass through the central `SafetyGuard` before execution.

## Memory And Knowledge

Kratos keeps several local, project-scoped stores:

| Store | Path |
|---|---|
| Session logs and payloads | `.kratos/session_*.jsonl` |
| Planner artifacts | `.kratos/plans/` |
| Project memory | `.kratos/memory.json` |
| Global memory | `~/.kratos/memory.json` |
| Knowledge index | `.kratos/knowledge/` |
| Compression artifacts | `.kratos/knowledge/compressions/` |
| Web research evidence | `.kratos/research.jsonl` |

The knowledge index uses LanceDB when installed. If LanceDB is unavailable,
Kratos falls back to a JSON index and still runs.

There is no file-size limit on indexing or knowledge ingestion — handling a
project of any size is the point of the vector store. It stays memory-safe by
reading proportionally rather than excluding files: files up to 4 MB are read and
chunked in full, while larger files are seek-sampled (only ~80 small windows at
evenly-spaced offsets are read), so even a multi-GB `.jsonl` is indexed and
retrievable with peak memory in the low hundreds of KB. The pre-planner retrieval
also records which sources/chunks it pulled — to the session log in full and as a
compact source list in the CLI — so the vector-DB step is auditable.

The knowledge base rebuilds automatically only when empty; after adding files,
force a refresh with `/knowledge rebuild force`.

## Safety

All command and file-write surfaces call `kratos/safety.py`. The guard blocks
destructive disk operations, recursive forced deletion patterns, registry and
scheduled-task changes, credential scraping, hidden elevation, encoded commands,
download-and-execute chains, and writes outside the project root. Git internals
such as `.git/objects`, `.git/refs`, and `.git/HEAD` are protected.

Permission profiles tune how much shell work Kratos may attempt, but the hard
blocklist still applies in every profile.

## Configuration

The config is stored under the local Kratos config path and loaded by
`kratos/config.py`. The current default shape is:

```json
{
  "planner_model": "huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M",
  "coder_model": "huihui_ai/qwen2.5-coder-abliterate:7b-instruct-q4_K_M",
  "verifier_model": "huihui_ai/deepseek-r1-abliterated:8b-0528-qwen3-q4_K_M",
  "compressor_model": "kratos-planner",
  "embed_model": "nomic-embed-text",
  "planner_num_ctx": 131072,
  "coder_num_ctx": 32768,
  "verifier_num_ctx": 131072,
  "compressor_num_ctx": 32768,
  "relay_num_ctx": 131072,
  "vram_ctx_ceiling": 8192,
  "always_max_ctx": true,
  "deterministic_verify": true,
  "coder_loop": true,
  "max_coder_iterations": 0,
  "max_work_step_turns": 4,
  "max_verify_iterations": 0,
  "no_progress_abort": 40,
  "auto_discover_verification": true,
  "protect_existing_tests": true,
  "permission": "mid"
}
```

Legacy configs with `max_verify_iterations: 10` are upgraded to unbounded
verification on load. Legacy VRAM ceilings above 16384 are clamped to 8192.

## Documentation

- [Agent architecture](docs/agent_architecture.md)
- [Tools](docs/tools.md)
- [Verification](docs/verification.md)
- [Safety](docs/safety.md)
- [Convergence safeguards](docs/convergence_fixes.md)
- [Hardening notes](docs/hardening_changes.md)
- [Analysis and rebuild notes](docs/analysis_rebuild_plan.md)
