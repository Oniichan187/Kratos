"""Surgical search/replace edits — the anti-rewrite-thrash primitive.

Weak/local models converge far better when they can change ONE thing instead of
re-emitting a whole file every turn (full rewrites routinely reintroduce bugs the
previous turn fixed — observed directly in the csvstats session: transform.py
rewritten 26x, never converging). This module parses and applies Aider-style
search/replace blocks:

    ### EDIT: path/to/file.py
    <<<<<<< SEARCH
    old text exactly as it appears
    =======
    new text
    >>>>>>> REPLACE

Pure stdlib, side-effect free. ``apply_search_replace`` returns a status so the
caller can give the model an honest, actionable observation when the SEARCH
block does not match (instead of silently corrupting the file).
"""

from __future__ import annotations

import re

__all__ = ["parse_edit_blocks", "apply_search_replace"]

# ``### EDIT: <path>`` then a SEARCH/REPLACE block. Tolerant of marker casing and
# of the exact number of <,=,> fence characters (>=3), which weak models vary.
_EDIT_RE = re.compile(
    r'^[ \t]*#{2,4}\s*EDIT\s*:?[ \t]*(?P<path>[^\r\n]+?)[ \t]*\r?\n'
    r'[ \t]*<{3,}\s*SEARCH[ \t]*\r?\n'
    r'(?P<search>.*?)\r?\n'
    r'[ \t]*={3,}[ \t]*\r?\n'
    r'(?P<replace>.*?)'
    r'\r?\n[ \t]*>{3,}\s*REPLACE[ \t]*$',
    re.S | re.M | re.I,
)


def _clean_path(path: str) -> str:
    path = path.strip().strip('`').strip()
    if path.startswith("**") and not path.startswith("**/"):
        path = path[2:]
    if path.endswith("**") and not path.endswith("/**"):
        path = path[:-2]
    return path.strip().strip('`').strip()


def parse_edit_blocks(text: str) -> list[tuple[str, str, str]]:
    """Return ``[(path, search, replace), ...]`` for every ### EDIT block."""
    out: list[tuple[str, str, str]] = []
    for m in _EDIT_RE.finditer(text or ""):
        path = _clean_path(m.group("path"))
        if not path or any(ch in path for ch in "\r\n`<>"):
            continue
        out.append((path, m.group("search"), m.group("replace")))
    return out


def apply_search_replace(content: str, search: str, replace: str) -> tuple[str, str]:
    """Apply one search→replace to *content*.

    Returns ``(new_content, status)`` where status is one of:
      - ``"ok"``            exact unique match replaced
      - ``"ok_normalized"`` matched after tolerating CRLF / trailing whitespace
      - ``"noop"``          search == replace (or replace already present)
      - ``"ambiguous"``     search occurs more than once (first one replaced)
      - ``"not_found"``     search text not present
      - ``"empty_search"``  search block was blank

    Exact matching is tried first (byte-faithful). Only on failure does it fall
    back to a line-based match that ignores trailing whitespace and CRLF/LF
    differences while PRESERVING indentation (which is significant in Python).
    """
    if search is None or search.strip() == "":
        return content, "empty_search"
    if search == replace:
        return content, "noop"

    occurrences = content.count(search)
    if occurrences == 1:
        return content.replace(search, replace, 1), "ok"
    if occurrences > 1:
        return content.replace(search, replace, 1), "ambiguous"

    # Fallback: line-based, trailing-whitespace/CRLF tolerant, indentation-faithful.
    newline = "\r\n" if "\r\n" in content else "\n"
    c_lines = content.replace("\r\n", "\n").split("\n")
    s_lines = [ln.rstrip() for ln in search.replace("\r\n", "\n").strip("\n").split("\n")]
    if not s_lines:
        return content, "not_found"
    c_stripped = [ln.rstrip() for ln in c_lines]
    n = len(s_lines)
    hits = [i for i in range(len(c_stripped) - n + 1) if c_stripped[i:i + n] == s_lines]
    if len(hits) == 1:
        i = hits[0]
        r_lines = replace.replace("\r\n", "\n").split("\n")
        new_lines = c_lines[:i] + r_lines + c_lines[i + n:]
        return newline.join(new_lines), "ok_normalized"
    if len(hits) > 1:
        return content, "ambiguous"
    return content, "not_found"
