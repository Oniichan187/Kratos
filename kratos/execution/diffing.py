"""Real diff detection — unified diffs + git-backed changed-file detection.

Pure stdlib (``difflib`` + ``git`` via ``subprocess``). Used by the Reporter so
that file-change evidence, line counts and the diff summary are derived from
ACTUAL content / VCS state, never from model text.

  - :func:`diff_stats`        — accurate added/removed line counts (difflib).
  - :func:`unified_diff`      — a real unified diff between two text blobs.
  - :func:`git_changed_files` — files the repo reports as changed (tracked +
                                untracked), independent of any model claim.
  - :func:`git_diff_stat`     — ``git diff HEAD --stat`` (staged + unstaged).

Every function is side-effect free and never raises; git helpers return empty
results when git or the repository is unavailable.
"""

from __future__ import annotations

import difflib
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DiffStats", "diff_stats", "unified_diff", "git_changed_files", "git_diff_stat"]


@dataclass(frozen=True)
class DiffStats:
    added: int
    removed: int

    @property
    def changed(self) -> bool:
        return self.added > 0 or self.removed > 0

    def as_suffix(self) -> str:
        return f"+{self.added}/-{self.removed}"


def diff_stats(before: str | None, after: str | None) -> DiffStats:
    """Accurate added/removed line counts between two text blobs.

    Uses ``difflib.ndiff`` (line-level), so a one-line insertion into an
    otherwise-unchanged file reports exactly ``+1/-0`` — not the coarse
    ``len(after) - len(before)`` estimate the tool layer used before.
    """
    before_lines = (before or "").splitlines()
    after_lines = (after or "").splitlines()
    added = removed = 0
    for line in difflib.ndiff(before_lines, after_lines):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return DiffStats(added=added, removed=removed)


def unified_diff(before: str | None, after: str | None,
                 path: str = "file", context: int = 3) -> str:
    """A real unified diff (``difflib.unified_diff``) between two text blobs.

    Returns '' when the two sides are identical."""
    before_lines = (before or "").splitlines(keepends=True)
    after_lines = (after or "").splitlines(keepends=True)
    if before_lines and not before_lines[-1].endswith("\n"):
        before_lines[-1] += "\n"
    if after_lines and not after_lines[-1].endswith("\n"):
        after_lines[-1] += "\n"
    return "".join(difflib.unified_diff(
        before_lines, after_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}", n=context,
    ))


def _git(args: list[str], project_root: Path, timeout: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            cwd=str(project_root),
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def git_changed_files(project_root: Path, timeout: int = 15) -> list[str]:
    """Real changed files (relative posix paths) per git — tracked modifications
    (``git diff --name-only HEAD``, i.e. staged + unstaged) plus untracked files
    (``git ls-files --others --exclude-standard``).

    This is a technical cross-check the Reporter unions with disk-snapshot
    evidence so a real change is never missed and a fake one is never invented.
    Returns ``[]`` when git/the repo is unavailable. Never raises.
    """
    found: set[str] = set()
    for args in (["diff", "--name-only", "HEAD"],
                 ["ls-files", "--others", "--exclude-standard"]):
        code, out = _git(args, project_root, timeout)
        if code == 0:
            found.update(line.strip().replace("\\", "/")
                         for line in out.splitlines() if line.strip())
    return sorted(found)


def git_diff_stat(project_root: Path, timeout: int = 15) -> str:
    """``git diff HEAD --stat`` (captures staged AND unstaged changes).

    Returns '' when git/the repo is unavailable. Never raises."""
    code, out = _git(["diff", "HEAD", "--stat"], project_root, timeout)
    return out.strip() if code == 0 else ""
