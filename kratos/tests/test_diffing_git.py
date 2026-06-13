"""Tests for real diff detection + git-backed changed-file cross-check.

Covers execution/diffing.py: difflib line stats, unified diff, and the
model-free git change detection the Reporter relies on (requirements: file
diff detection + technical files_changed).
"""

import subprocess
from pathlib import Path

import pytest

from kratos.execution.diffing import (
    diff_stats,
    unified_diff,
    git_changed_files,
    git_diff_stat,
)


def test_diff_stats_counts_added_and_removed():
    s = diff_stats("a\nb\nc\n", "a\nb\nc\nd\n")
    assert (s.added, s.removed) == (1, 0)
    assert s.changed is True
    s2 = diff_stats("a\nb\nc\n", "a\nc\n")
    assert (s2.added, s2.removed) == (0, 1)


def test_diff_stats_noop_is_not_a_change():
    s = diff_stats("same\ncontent\n", "same\ncontent\n")
    assert (s.added, s.removed) == (0, 0)
    assert s.changed is False


def test_diff_stats_handles_none():
    s = diff_stats(None, "new\n")
    assert s.added == 1 and s.removed == 0


def test_unified_diff_identical_is_empty():
    assert unified_diff("x\ny\n", "x\ny\n") == ""


def test_unified_diff_shows_change():
    d = unified_diff("x\ny\n", "x\nz\n", path="m.py")
    assert "-y" in d and "+z" in d
    assert "a/m.py" in d and "b/m.py" in d


def test_git_helpers_empty_when_not_a_repo(tmp_path: Path):
    assert git_changed_files(tmp_path) == []
    assert git_diff_stat(tmp_path) == ""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    try:
        _git(tmp_path, "init", "-q")
    except Exception:
        pytest.skip("git not available")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_git_changed_files_detects_modification(git_repo: Path):
    (git_repo / "a.py").write_text("print(2)\n", encoding="utf-8")
    assert "a.py" in git_changed_files(git_repo)


def test_git_changed_files_detects_untracked(git_repo: Path):
    (git_repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert "new.py" in git_changed_files(git_repo)


def test_git_changed_files_clean_repo_is_empty(git_repo: Path):
    assert git_changed_files(git_repo) == []


def test_git_diff_stat_reports_change(git_repo: Path):
    (git_repo / "a.py").write_text("print(2)\nprint(3)\n", encoding="utf-8")
    assert "a.py" in git_diff_stat(git_repo)
