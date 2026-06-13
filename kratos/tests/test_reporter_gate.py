"""Tests for the anti-fake-success reporter gate.

Covers the acceptance criteria:
  - files_changed empty (no real change) + changes required → FAILED
  - real change but tests failed → never SUCCESS
  - tests pass + verifier accepted → SUCCESS
  - no-op rewrite is NOT counted as a change
  - the commands table is built from real recorded results (incl. shell)
"""

from pathlib import Path

import pytest

from kratos.reporter import build_final_report, verify_files_changed, NO_REAL_CHANGES_MSG
from kratos.verification import ProvenWork


def _proof_with_test(exit_code, cmd="python -m pytest", output=""):
    p = ProvenWork(iteration=1)
    p.commands = [{"cmd": cmd, "shell": "system", "is_test": True,
                   "exit_code": exit_code, "output": output}]
    return p


def test_no_change_when_required_is_failed(tmp_path: Path):
    proof = _proof_with_test(0)
    proof.files_changed = []
    rep = build_final_report(tmp_path, proof, {}, task_requires_changes=True)
    assert rep.status == "FAILED"
    assert any(NO_REAL_CHANGES_MSG in p for p in rep.problems)


def test_noop_rewrite_does_not_count(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("same")
    proof = _proof_with_test(0)
    proof.files_changed = ["a.py"]
    # snapshot identical to current content → unchanged
    rep = build_final_report(tmp_path, proof, {"a.py": "same"}, task_requires_changes=True)
    assert rep.status == "FAILED"  # nothing really changed


def test_real_change_but_tests_failed_is_not_success(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("new content")
    proof = _proof_with_test(2, output="cannot import name x from partially initialized module y")
    proof.files_changed = ["a.py"]
    rep = build_final_report(tmp_path, proof, {"a.py": "old"}, task_requires_changes=True,
                             verifier_accepted=False)
    assert rep.status != "SUCCESS"
    assert rep.tests_ran is True
    assert rep.tests_passed is False
    # a concrete diagnosis must be attached, not just the raw trace
    assert rep.diagnosis


def test_success_requires_tests_and_verifier(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("new content")
    proof = _proof_with_test(0)
    proof.files_changed = ["a.py"]
    rep = build_final_report(tmp_path, proof, {"a.py": "old"}, task_requires_changes=True,
                             verifier_accepted=True)
    assert rep.status == "SUCCESS"


def test_tests_not_run_blocks_success(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("new content")
    proof = ProvenWork(iteration=1)  # no commands at all
    proof.files_changed = ["a.py"]
    rep = build_final_report(tmp_path, proof, {"a.py": "old"}, task_requires_changes=True,
                             verifier_accepted=True)
    assert rep.status != "SUCCESS"
    assert rep.tests_ran is False


def test_markdown_contains_shell_column_and_web_section(tmp_path: Path):
    f = tmp_path / "a.py"
    f.write_text("new")
    proof = _proof_with_test(0)
    proof.files_changed = ["a.py"]
    rep = build_final_report(tmp_path, proof, {"a.py": "old"}, task_requires_changes=True,
                             verifier_accepted=True, web_requested=True)
    md = rep.to_markdown()
    assert "| Befehl | Shell | Exitcode | Ergebnis |" in md
    assert "## Websuche/Webscraping" in md
    assert "Verlangt: Ja" in md


def test_verify_files_changed_classifies(tmp_path: Path):
    (tmp_path / "created.py").write_text("x")
    (tmp_path / "modified.py").write_text("new")
    ev = verify_files_changed(
        tmp_path, ["created.py", "modified.py", "gone.py"],
        {"created.py": None, "modified.py": "old", "gone.py": "was here"},
    )
    kinds = {e.path: e.kind for e in ev}
    assert kinds["created.py"] == "created"
    assert kinds["modified.py"] == "modified"
    assert kinds["gone.py"] == "deleted"
