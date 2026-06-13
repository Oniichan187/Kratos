"""Test-file guard — keep a "make the tests pass" result trustworthy.

A weak coder model can make pytest green the wrong way: by editing the provided
tests (weakening an assertion, deleting a case) instead of fixing the code.
Session 2026-06-13_20-55-49 showed the model rewriting ``tests/test_parser.py``
and ``tests/test_cli.py`` — harmlessly this time (it re-emitted them verbatim),
but the capability alone makes a green result unverifiable.

This guard snapshots the PRE-EXISTING test files at run start and restores them
before the authoritative verification. The model may still ADD new test files
(``Ergänze bei Bedarf weitere Tests``), but it can never alter the tests it was
asked to satisfy. SUCCESS therefore means the original tests passed.

Pure stdlib, project-root-confined.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["is_test_file", "snapshot_test_files", "restore_test_files"]

_IGNORE_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", ".venv", "venv", "env", "node_modules",
    "dist", "build", "bin", "obj", ".idea", ".vs", ".kratos", ".claude", ".pytest_cache",
}

# pytest (test_*.py / *_test.py) and common JS/TS test conventions.
_TEST_NAME_RE = re.compile(
    r"(?:^test_.*\.py$|^.*_test\.py$|\.test\.[jt]sx?$|\.spec\.[jt]sx?$)", re.I,
)


def is_test_file(rel_path: str) -> bool:
    """True if *rel_path* looks like a test file by name or location."""
    p = rel_path.replace("\\", "/")
    base = p.rsplit("/", 1)[-1]
    if _TEST_NAME_RE.search(base):
        return True
    parts = p.lower().split("/")
    return "tests" in parts or "test" in parts or "__tests__" in parts


def snapshot_test_files(project_root: Path, max_files: int = 200) -> dict[str, str]:
    """Capture the current contents of every pre-existing test file.

    Returns ``{relative_posix_path: content}``. Files that cannot be read are
    skipped. This is taken ONCE, before the agent makes any change.
    """
    root = Path(project_root)
    snap: dict[str, str] = {}
    try:
        for path in root.rglob("*"):
            if len(snap) >= max_files:
                break
            if not path.is_file():
                continue
            if any(part in _IGNORE_DIRS for part in path.parts):
                continue
            rel = path.relative_to(root).as_posix()
            if not is_test_file(rel):
                continue
            try:
                snap[rel] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    except OSError:
        pass
    return snap


def restore_test_files(project_root: Path, snapshot: dict[str, str]) -> list[str]:
    """Restore any snapshotted test file whose on-disk content was changed
    (or that was deleted). Returns the list of restored relative paths.

    New test files the agent added are NOT in the snapshot and are left intact.
    """
    root = Path(project_root).resolve()
    restored: list[str] = []
    for rel, original in snapshot.items():
        target = (root / rel)
        try:
            target.resolve().relative_to(root)
        except (ValueError, OSError):
            continue
        try:
            current = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
        except OSError:
            current = None
        if current == original:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original, encoding="utf-8")
            restored.append(rel)
        except OSError:
            continue
    return restored
