"""PatternSearch + targeted file reading for the coder loop.

Pure-stdlib, project-root-confined helpers:

  - :func:`list_files`       — recursive listing with ignore patterns
  - :func:`glob_files`       — filename glob search (``**/*.py`` style)
  - :func:`search_text`      — literal substring search (case-(in)sensitive)
  - :func:`search_regex`     — regex search
  - :func:`read_file_range`  — exact line-range read with robust encoding

All search functions return :class:`SearchMatch` records carrying file, line,
column and a context window, so observations can show the model exactly where
a hit landed. Binary files and ignored directories are skipped.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "SearchMatch", "RangeRead",
    "list_files", "glob_files", "search_text", "search_regex", "read_file_range",
    "smart_search", "extract_keywords", "resolve_project_path",
    "DEFAULT_IGNORE_DIRS",
]

DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".svn", ".hg", ".idea", ".vscode", "__pycache__", ".mypy_cache",
    ".pytest_cache", "node_modules", ".venv", "venv", "env", "dist", "build",
    "target", "out", "bin", "obj", ".next", ".nuxt", ".cache", "coverage",
    "htmlcov", ".tox", ".eggs", ".kratos", ".claude", "models",
})

_MAX_FILE_BYTES = 2_000_000   # skip giant files during search
_BINARY_SNIFF = 4096


@dataclass
class SearchMatch:
    rel_path: str
    line: int          # 1-based
    column: int        # 1-based
    text: str          # the matching line, stripped, capped
    context: list[str] = field(default_factory=list)  # surrounding lines

    def to_dict(self) -> dict:
        return {"path": self.rel_path, "line": self.line, "column": self.column,
                "text": self.text, "context": self.context}


@dataclass
class RangeRead:
    rel_path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
    ok: bool = True
    error: str = ""


def _is_ignored(path: Path, ignore_dirs: frozenset[str]) -> bool:
    return any(part in ignore_dirs for part in path.parts)


def _looks_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(_BINARY_SNIFF)
        return b"\x00" in chunk
    except OSError:
        return True


def _resolve_in_root(root: Path, rel: str) -> Path | None:
    """Resolve *rel* (windows or unix separators) inside *root*, or None."""
    cleaned = rel.strip().strip('`"\'').strip().replace("\\", "/")
    target = (root / cleaned).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def resolve_project_path(root: Path, rel: str) -> tuple[str | None, str]:
    """Resolve *rel* against the project, tolerating subdir-relative paths.

    Models working on nested layouts emit paths relative to the SUBPROJECT
    (e.g. ``fixtures/weather_sample.html`` when the file really lives at
    ``starter_project/fixtures/weather_sample.html``). Real runs failed with
    'file does not exist' three times in a row because of this.

    Returns ``(resolved_rel_path, note)``:
      - exact path exists            → (cleaned rel, "")
      - exactly ONE suffix match     → (matched rel, "resolved via suffix match")
      - ambiguous or no match        → (None, reason)
    """
    cleaned = (rel or "").strip().strip('`"\'').strip().replace("\\", "/")
    # strip './' prefixes ONLY (never '../' — that must hit the escape check)
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned:
        return None, "empty path"
    direct = _resolve_in_root(root, cleaned)
    if direct is None:
        return None, "path escapes project root"
    if direct.exists():
        return cleaned, ""
    # suffix match: every project file whose relative path ends with the
    # requested path (component-aligned)
    suffix = "/" + cleaned
    matches = [f for f in list_files(root) if f == cleaned or f.endswith(suffix)]
    if len(matches) == 1:
        return matches[0], f"resolved {rel!r} -> {matches[0]!r} (suffix match)"
    if len(matches) > 1:
        return None, f"ambiguous path {rel!r}: matches {', '.join(matches[:4])}"
    return None, "file does not exist"


def list_files(root: Path, ignore_patterns: frozenset[str] | None = None,
               max_files: int = 5000) -> list[str]:
    """Recursive file listing (relative posix paths), honoring ignore dirs."""
    ignore = ignore_patterns or DEFAULT_IGNORE_DIRS
    out: list[str] = []
    try:
        for path in sorted(root.rglob("*")):
            if len(out) >= max_files:
                break
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if _is_ignored(rel, ignore):
                continue
            out.append(rel.as_posix())
    except OSError:
        pass
    return out


def glob_files(root: Path, pattern: str, max_results: int = 200) -> list[str]:
    """Glob filename search. Accepts ``*.py``, ``**/*.test.ts``,
    ``src/**/util*`` — both separators are normalized."""
    pattern = (pattern or "").strip().replace("\\", "/")
    if not pattern:
        return []
    bare = "/" not in pattern  # bare name pattern -> match basename anywhere
    matches: list[str] = []
    for rel in list_files(root):
        candidate = rel.split("/")[-1] if bare else rel
        if fnmatch.fnmatch(candidate, pattern):
            matches.append(rel)
            if len(matches) >= max_results:
                break
    return matches


def _iter_search_files(root: Path, glob: str | None):
    candidates = glob_files(root, glob, max_results=2000) if glob else list_files(root)
    for rel in candidates:
        path = root / rel
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if _looks_binary(path):
            continue
        yield rel, path


def _collect_matches(root: Path, pattern: re.Pattern[str], glob: str | None,
                     max_results: int, context_lines: int) -> list[SearchMatch]:
    results: list[SearchMatch] = []
    for rel, path in _iter_search_files(root, glob):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines):
            m = pattern.search(line)
            if not m:
                continue
            lo = max(0, i - context_lines)
            hi = min(len(lines), i + context_lines + 1)
            results.append(SearchMatch(
                rel_path=rel, line=i + 1, column=m.start() + 1,
                text=line.strip()[:240],
                context=[ln[:240] for ln in lines[lo:hi]],
            ))
            if len(results) >= max_results:
                return results
    return results


def search_text(root: Path, pattern: str, glob: str | None = None,
                case_sensitive: bool = False, max_results: int = 50,
                context_lines: int = 1) -> list[SearchMatch]:
    """Literal substring search across project files."""
    if not pattern:
        return []
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(re.escape(pattern), flags)
    return _collect_matches(root, compiled, glob, max_results, context_lines)


def search_regex(root: Path, regex: str, glob: str | None = None,
                 case_sensitive: bool = True, max_results: int = 50,
                 context_lines: int = 1) -> list[SearchMatch] | str:
    """Regex search across project files. Returns an error string for an
    invalid pattern (so callers can surface it as an observation, not a crash)."""
    if not regex:
        return []
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(regex, flags)
    except re.error as exc:
        return f"invalid regex: {exc}"
    return _collect_matches(root, compiled, glob, max_results, context_lines)


def read_file_range(root: Path, rel_path: str, start_line: int, end_line: int,
                    max_chars: int = 20_000) -> RangeRead:
    """Read lines [start_line, end_line] (1-based, inclusive) of a project file.

    Paths are resolved tolerantly (suffix match for nested layouts)."""
    resolved, note = resolve_project_path(root, rel_path)
    if resolved is None:
        return RangeRead(rel_path, start_line, end_line, 0, "", ok=False, error=note)
    rel_path = resolved
    target = (root / resolved).resolve()
    if not target.is_file():
        return RangeRead(rel_path, start_line, end_line, 0, "", ok=False,
                         error="file does not exist")
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return RangeRead(rel_path, start_line, end_line, 0, "", ok=False, error=str(exc))
    lines = raw.splitlines()
    total = len(lines)
    if total == 0:
        return RangeRead(rel_path, 0, 0, 0, "", ok=True)
    start = max(1, min(int(start_line or 1), total))
    end = max(start, min(int(end_line or total), total))
    chunk = "\n".join(lines[start - 1:end])
    if len(chunk) > max_chars:
        chunk = chunk[:max_chars] + "\n... [truncated]"
    return RangeRead(rel_path, start, end, total, chunk)


# ── smart search — one command that always finds SOMETHING useful ────────────

_REGEX_META_RE = re.compile(r"[\\^$.*+?()\[\]{}]")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_RE = re.compile(r'[`"“”\'‘’]([^`"“”\'‘’]{2,60})[`"“”\'‘’]')
_STOPWORDS = frozenset("""
the a an and or for with that this from into onto of in on at to we our you your
exact line range contains find search look looking get check show list need needs
file files pattern parse parsing implement implementation function code python
der die das und oder für mit von in auf zu nach bei finde suche zeile bereich
""".split())


def extract_keywords(text: str, max_keywords: int = 6) -> list[str]:
    """Extract searchable code tokens from a prose description.

    Models love to 'search' for sentences like
    'Find the exact line range that contains the weather card for "Feldkirch"'.
    This pulls out the parts that can actually hit: quoted/backticked strings,
    identifiers with `_`/CamelCase, and capitalized or long non-stopword words.
    """
    keywords: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        tok = tok.strip()
        key = tok.lower()
        if len(tok) >= 3 and key not in _STOPWORDS and key not in seen:
            seen.add(key)
            keywords.append(tok)

    for m in _QUOTED_RE.finditer(text):          # quoted strings first (highest signal)
        _add(m.group(1))
    for w in _WORD_RE.findall(text):
        if "_" in w or (w[:1].islower() and any(c.isupper() for c in w[1:])):
            _add(w)                              # snake_case / camelCase identifiers
    for w in _WORD_RE.findall(text):
        if w[:1].isupper() and w.lower() not in _STOPWORDS:
            _add(w)                              # proper nouns (Feldkirch, WeatherCard)
    for w in _WORD_RE.findall(text):
        if len(w) >= 5:
            _add(w)                              # remaining long words
    return keywords[:max_keywords]


def smart_search(root: Path, pattern: str, glob: str | None = None,
                 max_results: int = 50, context_lines: int = 1) -> tuple[list[SearchMatch], str]:
    """One search to rule them all. Returns ``(matches, strategy_note)``.

    Pipeline (stops at the first stage with hits):
      1. literal, case-insensitive
      2. ``|`` alternation: each segment literal, OR-combined
      3. raw regex (when the pattern contains regex metacharacters)
      4. prose fallback: extract code keywords, OR-search them
    """
    pattern = (pattern or "").strip()
    if not pattern:
        return [], "empty pattern"

    # 1. literal
    matches = search_text(root, pattern, glob=glob, case_sensitive=False,
                          max_results=max_results, context_lines=context_lines)
    if matches:
        return matches, "literal"

    # 2. explicit | alternation (each side literal)
    if "|" in pattern:
        parts = [p.strip() for p in pattern.split("|") if p.strip()]
        if parts:
            alt = "|".join(re.escape(p) for p in parts)
            result = search_regex(root, alt, glob=glob, case_sensitive=False,
                                  max_results=max_results, context_lines=context_lines)
            if isinstance(result, list) and result:
                return result, f"alternation ({len(parts)} terms)"

    # 3. raw regex attempt
    if _REGEX_META_RE.search(pattern):
        result = search_regex(root, pattern, glob=glob, case_sensitive=False,
                              max_results=max_results, context_lines=context_lines)
        if isinstance(result, list) and result:
            return result, "regex"

    # 4. prose fallback — keyword OR-search
    words = pattern.split()
    if len(words) >= 3:
        keywords = extract_keywords(pattern)
        if keywords:
            alt = "|".join(re.escape(k) for k in keywords)
            result = search_regex(root, alt, glob=glob, case_sensitive=False,
                                  max_results=max_results, context_lines=context_lines)
            if isinstance(result, list) and result:
                return result, f"keyword fallback: searched {' | '.join(keywords)}"
            return [], f"no hits — even keyword fallback ({' | '.join(keywords)}) found nothing"

    return [], f"no matches for {pattern!r}"
