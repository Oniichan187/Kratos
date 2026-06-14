"""Integration tests for the ### EDIT handler (execution/tools.py do_edit)."""

from pathlib import Path

from kratos.execution.tools import do_edit, parse_actions, has_any_action
from kratos.verification import ProvenWork


def _drive(gen):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def test_do_edit_applies_change_and_records_proof(tmp_path: Path):
    f = tmp_path / "m.py"
    f.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    proof = ProvenWork(iteration=1)
    snaps: dict = {}
    obs = _drive(do_edit(tmp_path, "m.py", "    return a - b", "    return a + b", proof, 1, snaps))
    assert obs["ok"] is True
    assert f.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
    assert "m.py" in proof.files_changed
    assert snaps["m.py"] == "def add(a, b):\n    return a - b\n"
    assert proof.file_checks[-1]["operation"] == "edit"


def test_do_edit_search_not_found_leaves_file_and_records_no_change(tmp_path: Path):
    f = tmp_path / "m.py"
    original = "x = 1\n"
    f.write_text(original, encoding="utf-8")
    proof = ProvenWork(iteration=1)
    obs = _drive(do_edit(tmp_path, "m.py", "y = 2", "y = 3", proof, 1, {}))
    assert obs["ok"] is False
    assert obs["status"] == "not_found"
    assert f.read_text(encoding="utf-8") == original
    assert "m.py" not in proof.files_changed


def test_do_edit_missing_file_directs_to_FILE(tmp_path: Path):
    proof = ProvenWork(iteration=1)
    obs = _drive(do_edit(tmp_path, "nope.py", "a", "b", proof, 1, {}))
    assert obs["ok"] is False
    assert "### FILE" in obs["detail"]


def test_do_edit_escape_refused(tmp_path: Path):
    proof = ProvenWork(iteration=1)
    obs = _drive(do_edit(tmp_path, "../evil.py", "a", "b", proof, 1, {}))
    assert obs["ok"] is False
    assert "escapes" in obs["detail"]


def test_parse_actions_includes_edits():
    text = "### EDIT: m.py\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
    actions = parse_actions(text, None)
    assert actions["edits"] == [("m.py", "old", "new")]
    assert has_any_action(actions)
