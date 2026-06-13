"""Generic test-repair loop — analyse a REAL failure, apply a targeted fix, retest.

Unlike :mod:`kratos.execution.repair` (which hardcodes complete replacement
files for two bundled sandbox probes), this loop is project-agnostic and driven
entirely by real command results:

    outcome = run_repair_loop(run_tests, apply_fix, max_attempts=3)

  - ``run_tests()``            -> a real ``CommandResult`` dict (e.g. from
                                 ``_run_verification_command`` / ``ShellRunner``).
                                 Must contain ``exit_code`` and the output.
  - ``apply_fix(diag, n)``     -> ``True`` iff it changed something on disk.
                                 ``diag`` is the structured
                                 :class:`~kratos.execution.diagnostics.Diagnosis`
                                 (or ``None`` if the failure was unrecognised).

Honesty guarantees (the whole point of the gate Kratos enforces elsewhere):

  * Success is defined ONLY as ``exit_code == 0`` returned by ``run_tests`` —
    the loop can never report a repair it did not prove by re-running.
  * The failure analysis comes from the real diagnostics engine
    (:func:`diagnose_command`), never from model text.
  * The loop stops — instead of spinning forever — on success, on a *stalled*
    signature (the same failure repeating, i.e. the last fix changed nothing
    relevant), when a fix reports no change, or at ``max_attempts``. This is the
    direct countermeasure to the real 94×-pytest non-convergence session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .diagnostics import Diagnosis, RepairTracker, diagnose_command

__all__ = ["RepairAttempt", "RepairOutcome", "run_repair_loop"]


@dataclass
class RepairAttempt:
    attempt: int
    exit_code: int
    signature: str
    category: str
    fix_instruction: str
    fix_applied: bool


@dataclass
class RepairOutcome:
    success: bool
    attempts: int
    stalled: bool = False
    reason: str = ""
    history: list[RepairAttempt] = field(default_factory=list)
    final_result: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "stalled": self.stalled,
            "reason": self.reason,
            "history": [a.__dict__ for a in self.history],
        }


def _exit_code(result: object) -> int:
    if isinstance(result, dict):
        try:
            return int(result.get("exit_code"))
        except (TypeError, ValueError):
            return 1
    return 1


RunTests = Callable[[], dict]
ApplyFix = Callable[[Optional[Diagnosis], int], bool]


def run_repair_loop(
    run_tests: RunTests,
    apply_fix: ApplyFix,
    max_attempts: int = 3,
    stall_threshold: int = 2,
) -> RepairOutcome:
    """Drive a diagnose → fix → retest cycle off real command results.

    Returns a :class:`RepairOutcome`. ``success`` is true only when ``run_tests``
    actually returned ``exit_code == 0``.
    """
    max_attempts = max(1, int(max_attempts))
    tracker = RepairTracker(stall_threshold=stall_threshold)
    outcome = RepairOutcome(success=False, attempts=0)

    for attempt in range(1, max_attempts + 1):
        result = run_tests()
        outcome.attempts = attempt
        outcome.final_result = result if isinstance(result, dict) else None

        if _exit_code(result) == 0:
            outcome.success = True
            outcome.reason = "tests passed"
            return outcome

        diag = diagnose_command(result) if isinstance(result, dict) else None
        signature = diag.signature if diag else "<unknown>"
        category = diag.category if diag else "unknown"
        fix_instruction = diag.fix_instruction if diag else ""
        tracker.register(signature)

        # No fix budget left — record the failure and stop (never claim success).
        if attempt >= max_attempts:
            outcome.history.append(RepairAttempt(
                attempt, _exit_code(result), signature, category, fix_instruction, False))
            outcome.reason = "max attempts reached without passing tests"
            return outcome

        # The same failure has repeated — the previous fix did not move the
        # outcome. Stop rather than loop on a guaranteed-failing edit.
        if tracker.is_stalled(signature):
            outcome.history.append(RepairAttempt(
                attempt, _exit_code(result), signature, category, fix_instruction, False))
            outcome.stalled = True
            outcome.reason = f"stalled on repeating failure ({signature})"
            return outcome

        applied = bool(apply_fix(diag, attempt))
        outcome.history.append(RepairAttempt(
            attempt, _exit_code(result), signature, category, fix_instruction, applied))
        if not applied:
            outcome.reason = "fix attempt changed nothing — stopping"
            return outcome

    return outcome
