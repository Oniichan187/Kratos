"""Tests for the test-file guard (anti-cheat: model must not weaken given tests)."""

from pathlib import Path
from kratos.execution.testguard import is_test_file, snapshot_test_files, restore_test_files


def test_is_test_file():
    assert is_test_file("tests/test_parser.py")
    assert is_test_file("test_cli.py")
    assert is_test_file("pkg/foo_test.py")
    assert is_test_file("src/app.test.ts")
    assert not is_test_file("log_report/parser.py")
    assert not is_test_file("README.md")


def test_snapshot_and_restore_modified(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    t = tmp_path / "tests" / "test_x.py"
    t.write_text("def test_real():\n    assert compute() == 42\n")
    snap = snapshot_test_files(tmp_path)
    assert "tests/test_x.py" in snap
    # model weakens the test
    t.write_text("def test_real():\n    assert True\n")
    restored = restore_test_files(tmp_path, snap)
    assert restored == ["tests/test_x.py"]
    assert "compute() == 42" in t.read_text()


def test_restore_recreates_deleted(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    t = tmp_path / "tests" / "test_x.py"
    t.write_text("assert 1\n")
    snap = snapshot_test_files(tmp_path)
    t.unlink()  # model deletes the test
    restored = restore_test_files(tmp_path, snap)
    assert restored == ["tests/test_x.py"]
    assert t.exists()


def test_new_tests_are_not_touched(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text("assert 1\n")
    snap = snapshot_test_files(tmp_path)
    # model ADDS a new test file — must be left intact
    (tmp_path / "tests" / "test_new.py").write_text("def test_added():\n    assert True\n")
    restored = restore_test_files(tmp_path, snap)
    assert restored == []
    assert (tmp_path / "tests" / "test_new.py").exists()


def test_unchanged_tests_not_restored(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("assert 1\n")
    snap = snapshot_test_files(tmp_path)
    restored = restore_test_files(tmp_path, snap)
    assert restored == []
