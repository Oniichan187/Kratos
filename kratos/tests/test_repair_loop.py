"""Tests for the generic test-repair loop (execution/repair_loop.py).

The loop must: prove success only via a real exit_code==0, analyse real
failures, apply targeted fixes, retest, and stop (never spin) on stall /
no-op-fix / budget exhaustion.
"""

from kratos.execution.repair_loop import run_repair_loop


def _result(exit_code: int, output: str = "") -> dict:
    return {"cmd": "python -m pytest", "exit_code": exit_code,
            "output": output, "stdout": output, "stderr": "", "is_test": True}


def test_success_on_first_run_applies_no_fix():
    calls = {"fix": 0}

    def run_tests():
        return _result(0)

    def apply_fix(diag, n):
        calls["fix"] += 1
        return True

    out = run_repair_loop(run_tests, apply_fix, max_attempts=3)
    assert out.success is True
    assert out.attempts == 1
    assert calls["fix"] == 0          # never tried to fix a passing run
    assert out.reason == "tests passed"


def test_fix_then_pass():
    state = {"fixed": False}

    def run_tests():
        return _result(0) if state["fixed"] else _result(
            1, "NameError: name 'ret' is not defined")

    def apply_fix(diag, n):
        state["fixed"] = True
        return True

    out = run_repair_loop(run_tests, apply_fix, max_attempts=3)
    assert out.success is True
    assert out.attempts == 2
    assert out.history and out.history[0].fix_applied is True


def test_stall_detection_stops_repeating_failure():
    # Same failure every time, fix claims to change things but never helps.
    def run_tests():
        return _result(1, "NameError: name 'foo' is not defined")

    def apply_fix(diag, n):
        return True

    out = run_repair_loop(run_tests, apply_fix, max_attempts=5, stall_threshold=2)
    assert out.success is False
    assert out.stalled is True
    assert "stalled" in out.reason


def test_noop_fix_stops_immediately():
    def run_tests():
        return _result(1, "AssertionError: 1 != 2")

    def apply_fix(diag, n):
        return False          # nothing changed

    out = run_repair_loop(run_tests, apply_fix, max_attempts=5)
    assert out.success is False
    assert out.attempts == 1
    assert "changed nothing" in out.reason


def test_max_attempts_never_reports_success():
    # Distinct failures each time so it never "stalls" — must stop at budget.
    seq = iter([
        "NameError: name 'a' is not defined",
        "NameError: name 'b' is not defined",
        "NameError: name 'c' is not defined",
    ])

    def run_tests():
        try:
            return _result(1, next(seq))
        except StopIteration:
            return _result(1, "NameError: name 'z' is not defined")

    def apply_fix(diag, n):
        return True

    out = run_repair_loop(run_tests, apply_fix, max_attempts=3)
    assert out.success is False
    assert out.attempts == 3
    assert "max attempts" in out.reason


def test_diagnosis_is_surfaced_in_history():
    def run_tests():
        return _result(1, 'File "m.py", line 3\n    def f(\n        ^\nSyntaxError: invalid syntax')

    def apply_fix(diag, n):
        # The real diagnostics engine should have categorised this.
        assert diag is not None
        return False

    out = run_repair_loop(run_tests, apply_fix, max_attempts=2)
    assert out.history[0].category  # a non-empty category was recorded
