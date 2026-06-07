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

def _get_file_change_re():
    pm = load_prompts()
    fm = re.escape(pm.get_marker("file") or "### FILE:")
    return re.compile(
        rf'^\s*{fm}\s*([^\r\n]+?)\s*\r?\n```(?:\w+)?[^\S\r\n]*\r?\n(.*?)^```\s*$',
        re.S | re.M,
    )


def _get_file_delete_re():
    pm = load_prompts()
    dm = re.escape(pm.get_marker("delete") or "### DELETE:")
    return re.compile(rf'{dm}\s*(.+?)\s*$', re.M)


# Fallback module-level compiled (using defaults at import time)
_FILE_CHANGE_RE = re.compile(
    r'^\s*###\s+FILE:\s*([^\r\n]+?)\s*\r?\n```(?:\w+)?[^\S\r\n]*\r?\n(.*?)^```\s*$',
    re.S | re.M,
)
_FILE_DELETE_RE = re.compile(r'###\s+DELETE:\s*(.+?)\s*$', re.M)


def _parse_file_changes(text: str) -> list[tuple[str, str]]:
    changes: list[tuple[str, str]] = []
    for m in _get_file_change_re().finditer(text):
        path = m.group(1).strip()
        path = re.sub(r'\s+\((?:updated|modified|new)\)\s*$', '', path, flags=re.I)
        path = re.sub(
            r'(\.[A-Za-z0-9][A-Za-z0-9._-]*)\s+\([^/\r\n]*\)\s*$',
            r'\1',
            path,
        )
        if not path or any(ch in path for ch in "\r\n`<>"):
            continue
        changes.append((path, m.group(2)))
    return changes


def _parse_file_deletions(text: str) -> list[str]:
    return [m.group(1).strip() for m in _get_file_delete_re().finditer(text)]
