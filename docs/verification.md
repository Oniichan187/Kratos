# Verification

Kratos treats verification as an evidence problem. A final answer can only
claim success when the recorded work supports that claim.

## Evidence Model

`kratos/verification.py` defines `ProvenWork`. During a run, tool handlers and
verification helpers add:

- files read;
- files written or deleted;
- file hashes before and after writes;
- real command results;
- verification command metadata;
- test/build flags;
- diagnostics;
- touched file lists.

`kratos/reporter.py` receives this evidence and the original file snapshots. It
does not trust free-form model text for facts.

## Status Gate

For code-changing work, the final report applies these rules:

- If no real file changed, status cannot be `SUCCESS`.
- If no test command ran, status cannot honestly say tests passed.
- If the last test command failed, status cannot be `SUCCESS`.
- If deterministic verification is enabled and real tests pass, Kratos accepts
  the result without asking the LLM verifier to re-judge it.
- If deterministic verification is disabled, the LLM verifier may participate,
  but the report still uses real command and file evidence.

No-op writes do not count as changes. `verify_files_changed` compares snapshots
with current disk contents and classifies files as created, modified, deleted,
or unchanged.

## Command Discovery

Verification commands can be supplied by the user or inferred by Kratos.

Sources:

- `/build <cmd>`
- `/test <cmd>`
- explicit commands in the request;
- command snippets found in README content;
- project toolchain inference.

Supported inference includes:

- Python: pytest, unittest, compile checks;
- Node: package scripts, TypeScript compile checks;
- .NET: build and test;
- Rust/Cargo;
- Go;
- Maven;
- Gradle.

`core/buildtest.py` normalizes common Windows and Python shims, handles nested
project working directories, and passes all commands through the safety gate.

## Test Protection

When `protect_existing_tests` is enabled, `kratos/execution/testguard.py`
snapshots pre-existing tests before the agent edits. Before authoritative
verification, changed or deleted original tests are restored.

New tests created by the agent are not removed by this guard. The intent is to
prevent a model from making the existing suite easier while still allowing it to
add coverage.

## Diagnostics

`kratos/execution/diagnostics.py` turns failing command output into structured
diagnoses. Categories include:

- circular imports;
- import and module-not-found errors;
- syntax and indentation errors;
- `NotImplementedError`;
- `NameError`;
- parse-returned-`None` bugs;
- attribute errors;
- signature mismatch;
- numeric operations on strings;
- missing expected raises;
- `TypeError`;
- assertions;
- pytest collection errors;
- timeouts;
- blocked commands;
- unknown failures.

`RepairTracker` tracks repeated failure signatures. If the same signature
repeats too many times, Kratos escalates the feedback instead of blindly
rerunning the same command forever.

## Deterministic Repairs

The main loop is model-driven, but Kratos has deterministic fallbacks for a few
well-defined failure patterns. The most important is removal of provably unused
cross-imports that create intra-package circular imports. The repair only
removes an import when all bound names are unused in that file.

These repairs still flow through the normal snapshot and verification machinery.

## Final Report

The report includes:

- result status;
- changed files with real line stats;
- found problems;
- executed commands;
- test status and reason;
- last failure diagnosis when relevant;
- web research status and sources;
- diff summary;
- remaining limitations.

Diff information comes from git when available and from snapshot comparison
otherwise. Empty or unavailable diffs are reported honestly.
