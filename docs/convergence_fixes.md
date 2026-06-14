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
