# Convergence Safeguards

Kratos is designed to keep weak local models from wasting a run on the same
mistake indefinitely. The current safeguards are practical rather than magical:
they improve feedback, reduce destructive rewrites, and stop repeated no-progress
loops.

## Surgical Edits

Existing files should be changed through `### EDIT`, not whole-file rewrites.
The edit tool uses explicit search/replace blocks, verifies the current text,
and leaves the file untouched when the search is missing or ambiguous.

This prevents a common failure mode where the model fixes one bug and
reintroduces another by rewriting a stale copy of the whole file.

## No-Op Write Suppression

`do_write` compares new content against the current on-disk content. When they
are byte-identical the write is skipped and reported as
`write_file('x') -> no change (identical content already on disk)` rather than a
fresh write. This stops a weak model from "fixing" a file by re-writing the exact
same bytes turn after turn (observed: the same module written twice in a row with
an identical `sha256`), and removes the misleading duplicate write line.

## Repeated Dead-Search Suppression

A `### SEARCH` / `### GREP` that returns zero results is remembered for the whole
coder loop, across checklist items. If the model emits the same zero-result query
again, the runtime skips it and injects a note steering it to read the target
file directly (`### READ` / `### READ_RANGE`) or change the term. An earlier run
burned an entire per-item turn budget repeating one identical zero-hit search
(`### SEARCH: re.sub :: numstats.py`, five-plus times); the per-item flail guard
reset on every item and never caught the cross-item repeat, so this memo is
loop-wide. Active in both the ReAct loop and the structured work-step driver.

## Failure Readback

After a failed verification command, `roles/coder.py` reads back:

- files touched by the model;
- files named or implicated by the deterministic diagnosis;
- the latest relevant failure output.

The next coder turn sees the current disk state instead of relying on memory of
what it wrote earlier.

## Diagnosis-Directed Repair

`execution/diagnostics.py` maps raw command output to concrete categories and
fix instructions. The coder receives targeted guidance such as:

- remove or restructure a circular import;
- convert CSV string values before numeric arithmetic;
- raise the expected exception instead of hiding it behind defaults;
- fix a signature mismatch;
- address pytest collection before assertion failures.

## Run Until Solved, Not Until A Small Counter Expires

`max_verify_iterations <= 0` means the verify-revise loop is unbounded. The
default is `0`.

The practical stop is `no_progress_abort`. With the current default of `40`,
Kratos stops when the same failure signature repeats enough times to show that
the run is stuck. It then reports the unresolved state honestly.

## Work-Step Limits

Planner-driven work uses structured work steps. Each checklist item has its own
micro-turn budget controlled by `max_work_step_turns` (default `4`). This keeps a
single item from consuming the entire run while still allowing the outer
verification loop to continue when progress is being made.

## Deterministic Fallbacks

Kratos includes deterministic repairs for narrow, provable cases. For example,
it can remove an unused import that forms a package-local circular import. It
does not remove imports whose names are used, and it still re-runs verification.

## Honest Limit

These safeguards do not guarantee that a small local model can solve every
programming problem. They make failures more diagnosable and reduce repeated
damage. The final report still returns `PARTIAL` or `FAILED` when the evidence
does not support success.

## Known Limitation: Non-Test Requirements

Authoritative success is gated on real passing tests. A task requirement that no
test exercises — for example "delete the deprecated module" — is not enforced by
that gate. If the planner also drops such a step from its checklist (plans
recovered from reasoning-only planner output can be noisy, duplicated, or
incomplete), the step can be silently skipped while the run still reports
`SUCCESS` on green tests. Spell out non-test requirements explicitly in the task,
or add a test or command that checks them.
