# Tools

Kratos gives the coder model a text-marker interface. The model emits markers,
then `kratos/execution/tools.py` parses and executes them against the project
root.

This design keeps model output auditable: every file write, command, fetch, and
search result becomes structured evidence or a structured observation.

## Marker Reference

| Marker | Example | Behavior |
|---|---|---|
| `### READ` | `### READ: src/app.py` | Reads one file. |
| `### READ_RANGE` | `### READ_RANGE: src/app.py:20-80` | Reads exact 1-based line range. |
| `### SEARCH` | `### SEARCH: render_user :: **/*.py` | Smart literal/regex/keyword search, optional glob. |
| `### GREP` | `### GREP: class\s+User :: **/*.py` | Regex search with smart fallback. |
| `### GLOB` | `### GLOB: src/**/*.py` | Lists matching files. |
| `### INSPECT` | `### INSPECT: python -m pytest --collect-only` | Runs read-only inspection commands. |
| `### FILE` | `### FILE: docs/new.md` | Creates or replaces a file with the following fenced body. |
| `### EDIT` | `### EDIT: src/app.py` | Applies one or more search/replace blocks. |
| `### DELETE` | `### DELETE: obsolete.py` | Deletes a project file. |
| `### VERIFY` | `### VERIFY` | Runs discovered verification command. |
| `### RUN` | `### RUN: python -m pytest` | Runs a guarded build/test command. |
| `### WEB_FETCH` | `### WEB_FETCH: https://example.com` | Fetches and extracts text from a page. |
| `### WEB_SEARCH` | `### WEB_SEARCH: package release notes` | Searches the web and records returned sources. |
| `### DONE` | `### DONE` | Ends the current work step. |

## Search And Reads

`kratos/execution/search.py` implements repository-local search:

- `list_files`
- `glob_files`
- `search_text`
- `search_regex`
- `read_file_range`
- `smart_search`
- `resolve_project_path`

The search layer ignores heavy and generated directories, including `.git`,
`.kratos`, `.claude`, `node_modules`, virtualenvs, build output, cache folders,
and model directories. It skips binary files and very large files.

`smart_search` tries practical fallbacks before giving up:

1. literal search;
2. alternation-friendly search;
3. regex search;
4. keyword search;
5. path or glob redirect when the input looks like a file path.

## Edits

Whole-file writes are available through `### FILE`, but existing files should
usually be changed with `### EDIT`.

`### EDIT` uses Aider-style search/replace blocks. The block starts with
`### EDIT: path/to/file.py`, then uses the delimiters `<<<<<<< SEARCH`,
`=======`, and `>>>>>>> REPLACE` around the exact current lines and replacement
lines.

`kratos/execution/edits.py` applies exact matches first, then normalized matches
that tolerate CRLF and trailing-whitespace drift. It preserves indentation and
returns explicit statuses:

- `ok`
- `ok_normalized`
- `not_found`
- `ambiguous`
- `noop`
- `empty_search`

If the search block cannot be applied, the file is left untouched and the coder
gets an observation telling it to re-read the file.

## Shell And Verification Commands

`kratos/execution/shell.py` provides `ShellRunner`. The main verification path
also uses guarded subprocess execution from `kratos/core/buildtest.py`.

Every command path applies `SafetyGuard` before execution. Results include:

- command string;
- shell;
- working directory;
- exit code;
- stdout;
- stderr;
- duration;
- timeout flag;
- blocked flag and reason.

Blocked commands return exit code `126` and are not executed. Timed-out commands
return exit code `124`.

`### INSPECT` is restricted to read-only commands. If the model tries to use it
for a path or prose request, the tool redirects to file read or search when
possible.

## Web Tools

`kratos/web.py` provides standard-library HTTP and search helpers:

| Function | Behavior |
|---|---|
| `web_fetch` | HTTP(S)-only GET with timeout, user agent, status checks, content checks, private-host guard, and 2 MB cap. |
| `scrape_text_from_html` | Extracts readable text with `html.parser` and strips script/style/navigation noise. |
| `web_search` | DuckDuckGo HTML search by default; unconfigured providers return honest errors. |
| `record_research_note` | Writes successful web actions to `.kratos/research.jsonl`. |

Web results are data only. They are never executed as commands.

The final report reads only sources recorded during the current run. If the
task asked for web research and the agent did not successfully fetch or search,
the report says so instead of inventing citations.

## Observations

Tool outputs are formatted back to the coder by `format_observation`. Failure
observations include useful diagnostics where possible. For example, a failed
test command may include both raw output and a deterministic diagnosis such as
`circular_import`, `missing_raise`, or `numeric_on_string`.

This feedback loop is deliberately concrete: the model sees current disk state,
the files implicated by the failure, and a targeted fix instruction.
