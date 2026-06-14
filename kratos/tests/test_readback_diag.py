"""The failure-readback must surface the file the DIAGNOSIS blames, even when
the model did not touch it this turn (so a weak model sees the real code to fix)."""

from pathlib import Path

from kratos.roles.coder import _inject_failure_readback


def test_readback_injects_diagnosis_blamed_file_when_not_touched(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "blamed.py").write_text("def f():\n    return summarize(x)\n", "utf-8")
    observations = [
        {"kind": "command", "cmd": "python -m pytest", "ok": False, "skipped": False,
         "exit_code": 1,
         "output": "tests/test_x.py F\nE  NameError: name 'summarize' is not defined\nsrc/blamed.py:2: NameError\n"},
    ]
    _inject_failure_readback(tmp_path, touched=[], observations=observations)
    notes = [o for o in observations if o.get("kind") == "note"]
    assert notes, "diagnosis-blamed file must be read back even if untouched"
    joined = "\n".join(n["detail"] for n in notes)
    assert "src/blamed.py" in joined
    assert "summarize" in joined
    assert "### EDIT" in joined          # nudges a targeted edit, not a rewrite


def test_readback_noop_when_command_passed(tmp_path: Path):
    (tmp_path / "ok.py").write_text("x = 1\n", "utf-8")
    observations = [{"kind": "command", "cmd": "python -m pytest", "ok": True,
                     "skipped": False, "exit_code": 0, "output": "1 passed"}]
    _inject_failure_readback(tmp_path, touched=["ok.py"], observations=observations)
    assert [o for o in observations if o.get("kind") == "note"] == []
