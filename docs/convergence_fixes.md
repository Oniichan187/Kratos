# Convergence Fixes — surgical edits + failure readback

Driven by a real failed run (`session_2026-06-14_00-37-20`): on the csvstats
task the agent ran **76 test commands, 0 ever passed**, rewrote `transform.py`
**26x**, and repeated the *same* failure signatures (`int + str` TypeError 16x,
circular-import 9x) without ever converging — the classic weak-model
whole-file-rewrite thrash.

## 1. Surgical `### EDIT` marker  (`kratos/execution/edits.py`)

Whole-file `### FILE:` rewrites were the only edit primitive, so every fix
risked re-introducing a bug the previous turn had fixed. Added an Aider-style
search/replace block:

```
### EDIT: path/to/file.py
<<<<<<< SEARCH
exact current lines
=======
new lines
>>>>>>> REPLACE
```

- `apply_search_replace` matches exactly first, then falls back to a line-based
  match that tolerates trailing-whitespace / CRLF drift while **preserving
  indentation**. Returns a status (`ok`, `ok_normalized`, `not_found`,
  `ambiguous`, `noop`, `empty_search`).
- `do_edit` (execution/tools.py) applies it, records the same hash/line-delta
  evidence into `ProvenWork` as a write, and on a non-matching SEARCH leaves the
  file **untouched** and tells the model to re-`### READ` — never silent
  corruption.
- Wired into both coder paths (free loop + stepwise) and `parse_actions`.
- Prompts now register the `edit` marker and instruct: *fix existing files with
  ### EDIT; use ### FILE only to create a file or fully rewrite a tiny one.*

## 2. Diagnosis-aware failure readback  (`roles/coder.py`)

`_inject_failure_readback` now, after a failed verify, surfaces the CURRENT
on-disk content of **both** the files the model touched **and** the files the
failure DIAGNOSIS blames (extracted from the real traceback) — even if the model
did not touch them this turn. The note nudges a targeted `### EDIT`. Also wired
into the free-loop path (previously stepwise-only).

## Tests

`kratos/tests/test_edits.py` (12), `test_do_edit.py` (5), `test_readback_diag.py`
(2). Full non-UI suite: **181 passed**.

## Honest limits

Stall detection (`RepairTracker`) and escalation were already wired; the
remaining non-convergence is the local model's own code quality. These changes
reduce regressions and give the model the real code + a surgical tool, but a
sufficiently weak model can still fail to fix a genuine logic bug — the verify
gate then correctly reports PARTIAL/FAILED rather than a fake success.

## 3. Run-until-solved (no iteration cap)  [2026-06-14]

The verify-revise loop was hard-capped at `max_verify_iterations=10`, so Kratos
gave up before a slow local model could converge. Now:

- `max_verify_iterations <= 0` means **UNBOUNDED** (new default `0`); a legacy
  saved value of `10` is auto-upgraded to unbounded on load.
- `no_progress_abort` (default `40`, `0` = never): the only stop for an unbounded
  run — it halts solely when the *identical* failure signature has repeated that
  many times (genuinely stuck), then reports honestly. Not an iteration cap.
- `max_coder_iterations` was already `0` (unbounded).

## 4. Targeted diagnoses for the recurring weak-model bugs

Added two specific, directive `Diagnosis` categories (from real csvstats runs):
- **`numeric_on_string`** — `TypeError: unsupported operand … 'int' and 'str'`
  / numeric comparison on str / `invalid literal for int()`: tells the model to
  `float(v)` the CSV string values before arithmetic and to raise `ValueError`
  on non-numeric input.
- **`missing_raise`** — pytest `DID NOT RAISE`: tells the model its function must
  actually `raise` (e.g. missing column / unknown op → `ValueError`) instead of
  using `dict.get()` defaults that hide the error.

Tests: `kratos/tests/test_convergence.py` (5). Full non-UI suite: **186 passed**.

## Honest limit (unchanged)

These give a weak local model more chances and far more precise guidance, but
cannot make a 4B abliterated model write correct code it doesn't "know". On the
csvstats probe the model still left 7/22 failing — the verify gate then correctly
reports PARTIAL/FAILED. For reliably green runs on non-trivial tasks, a stronger
coder model is the remaining lever.
