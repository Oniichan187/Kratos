"""Marker-based output parsing for coder responses.

The coder doesn't use Ollama's native ``tools`` schema — instead it emits
plain-text markers (``### FILE:``, ``### DELETE:``, ...) that the runtime
regex-parses here. Marker text is sourced from ``prompts_default.json`` at
call time so user customization stays in sync with the parser.
"""

from __future__ import annotations

import re

from ..prompts import load_prompts

# ── output parsers (markers sourced from prompts at runtime for perfect sync with JSON) ──

def _marker_pattern(marker: str) -> str:
    marker = (marker or "").strip()
    if marker.endswith(":"):
        marker = marker[:-1].rstrip()
    escaped = re.escape(marker).replace(r"\ ", r"\s+")
    return rf"{escaped}\s*:?"


# Between the "### FILE: <path>" line and the opening code fence, small local
# models often emit 1-2 annotation lines such as "*(full file content ...)*".
# The pattern tolerates up to 3 such non-fence, non-marker lines so those
# implementations are not silently dropped (the exact failure seen in real
# session logs). Lines starting a new ### marker are never skipped, so a
# malformed FILE marker cannot swallow the next marker's code fence.
_JUNK_LINES = r'(?:[ \t]*(?!```|###)[^\r\n]*\r?\n){0,3}?'


def _get_file_change_re():
    pm = load_prompts()
    fm = _marker_pattern(pm.get_marker("file") or "### FILE:")
    return re.compile(
        rf'^\s*{fm}\s*([^\r\n]+?)[ \t]*\r?\n{_JUNK_LINES}[ \t]*```(?:\w+)?[^\S\r\n]*\r?\n(.*?)^[ \t]*```\s*$',
        re.S | re.M | re.I,
    )


def _get_file_delete_re():
    pm = load_prompts()
    dm = _marker_pattern(pm.get_marker("delete") or "### DELETE:")
    return re.compile(rf'^\s*{dm}\s*(.+?)\s*$', re.M | re.I)


# Fallback module-level compiled (using defaults at import time)
_FILE_CHANGE_RE = re.compile(
    r'^\s*###\s+FILE\s*:?\s*([^\r\n]+?)[ \t]*\r?\n'
    + _JUNK_LINES +
    r'[ \t]*```(?:\w+)?[^\S\r\n]*\r?\n(.*?)^[ \t]*```\s*$',
    re.S | re.M | re.I,
)
_FILE_DELETE_RE = re.compile(r'^\s*###\s+DELETE\s*:?\s*(.+?)\s*$', re.M | re.I)


def _clean_file_path(path: str) -> str:
    """Strip markdown decoration models wrap around paths: backticks, bold
    asterisks, trailing annotations like '(updated)'."""
    path = path.strip().strip('`').strip()
    # `**path**` bold markers — but never break a leading '**/' glob
    if path.startswith("**") and not path.startswith("**/"):
        path = path[2:]
    if path.endswith("**") and not path.endswith("/**"):
        path = path[:-2]
    path = path.strip().strip('`').strip()
    path = re.sub(r'\s+\((?:updated|modified|new)\)\s*$', '', path, flags=re.I)
    path = re.sub(
        r'(\.[A-Za-z0-9][A-Za-z0-9._-]*)\s+\([^/\r\n]*\)\s*$',
        r'\1',
        path,
    )
    return path.strip()


def _parse_file_changes(text: str) -> list[tuple[str, str]]:
    changes: list[tuple[str, str]] = []
    for m in _get_file_change_re().finditer(text):
        path = _clean_file_path(m.group(1))
        if not path or any(ch in path for ch in "\r\n`<>"):
            continue
        changes.append((path, m.group(2)))
    return changes


def _parse_file_deletions(text: str) -> list[str]:
    return [_clean_file_path(m.group(1)) for m in _get_file_delete_re().finditer(text)]
