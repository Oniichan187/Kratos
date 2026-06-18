"""Action/tool layer for the adaptive ReAct coder loop.

The coder doesn't use Ollama's native ``tools`` schema (flaky for the small
abliterated models Kratos targets) — it emits plain-text action markers that
this module parses and dispatches to small handlers. Each handler is a
generator: it yields the same ``("tool"/"warn"/"error"/"log", message, kind)``
UI events the rest of the runtime emits, and *returns* (via the generator's
return value, consumed with ``yield from``) a structured **observation** dict
that the loop folds into the next coder turn.

Markers (sourced from ``prompts_default.json`` -> ``"markers"``, same pattern
as ``execution/parsing.py`` so user customization stays in sync):

  ``### FILE: <path>`` + fenced body   -> write file              (kind="file")
  ``### DELETE: <path>``               -> delete file             (kind="delete")
  ``### READ: <path>``                 -> return on-disk content  (kind="read")
  ``### READ_RANGE: <path>:<a>-<b>``   -> exact line range        (kind="range")
  ``### SEARCH: <text> [:: <glob>]``   -> literal search          (kind="search")
  ``### GREP: <regex> [:: <glob>]``    -> regex search            (kind="grep")
  ``### GLOB: <pattern>``              -> filename search         (kind="glob")
  ``### WEB_FETCH: <url>``             -> guarded HTTP GET        (kind="web_fetch")
  ``### WEB_SEARCH: <query>``          -> provider web search     (kind="web_search")
  ``### INSPECT: <cmd>``               -> run read-only shell diag (kind="inspect")
  ``### VERIFY: <cmd>`` / ``### RUN:`` -> run a guarded command   (kind="command")
  ``### DONE``                         -> signal loop completion  (parsed separately)

No new shell-exec surface: ``### VERIFY`` / ``### RUN`` reuse the *exact* safety
gates (``_is_safe_verification_command``, ``CommandRegistry.is_toolchain_mismatch``,
``_missing_command_paths``) and ``_run_verification_command`` already used by the
stepwise/legacy coder paths in ``core/agent.py`` — commands that fail any gate are
never executed, only reported back as a skip-observation.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Generator

from ..verification import (
    VerificationCommand,
    CommandRegistry,
    ProvenWork,
    _clean_command_line,
    _is_safe_inspect_command,
    _is_safe_verification_command,
    _is_test_verification_command,
    _missing_command_paths,
    _command_toolchain,
)
from .parsing import _parse_file_changes, _parse_file_deletions
from .edits import parse_edit_blocks, apply_search_replace
from .search import (
    extract_keywords,
    glob_files,
    read_file_range,
    resolve_project_path,
    search_regex,
    search_text,
    smart_search,
)
from ..web import scrape_text_from_html, web_fetch, web_search

_READ_CAP = 2500       # chars — matches the stepwise pre-step disk-read excerpt cap
_OUTPUT_TAIL = 600     # chars of command output folded into the observation


# ── marker discovery (read from prompts JSON at call time, tolerant parsing) ──

def _marker_pattern(marker: str) -> str:
    marker = (marker or "").strip()
    if marker.endswith(":"):
        marker = marker[:-1].rstrip()
    escaped = re.escape(marker).replace(r"\ ", r"\s+")
    # (?![\w-]) word boundary right after the marker word: prevents "### READ"
    # from swallowing "### READ_RANGE: ..." (seen in real session logs).
    return rf"{escaped}(?![\w-])\s*:?"


def _marker_text(pm, key: str, fallback: str) -> str:
    """Resolve a marker string from the prompt config.

    PromptManager.get_marker returns the KEY ITSELF when the key is missing
    from the JSON markers section — treat that exactly like "not configured"
    and use the code fallback (this silently broke READ_RANGE/SEARCH/GREP/...
    parsing in real runs before).
    """
    raw = pm.get_marker(key) if pm is not None else None
    if not raw or raw == key:
        return fallback
    return raw


def _marker_re(pm, key: str, fallback: str) -> "re.Pattern[str]":
    marker = _marker_pattern(_marker_text(pm, key, fallback))
    return re.compile(rf'^[ \t]*{marker}\s*(.*?)\s*$', re.M | re.I)


def _strip_md_decor(arg: str) -> str:
    """Strip markdown decoration models wrap around marker arguments:
    backticks and bold `**...**` (without breaking `**/` glob prefixes)."""
    s = (arg or "").strip().strip('`').strip()
    if s.startswith("**") and not s.startswith("**/"):
        s = s[2:]
    if s.endswith("**") and not s.endswith("/**"):
        s = s[:-2]
    return s.strip().strip('`').strip()


def _parse_read_markers(text: str, pm) -> list[str]:
    paths: list[str] = []
    for m in _marker_re(pm, "read", "### READ:").finditer(text):
        p = _strip_md_decor(m.group(1))
        if p and not any(ch in p for ch in "\r\n<>"):
            paths.append(p)
    return list(dict.fromkeys(paths))


def _parse_command_markers(text: str, pm) -> list[str]:
    cmds: list[str] = []
    markers = [
        _marker_pattern(_marker_text(pm, "verify", "### VERIFY:")),
        _marker_pattern(_marker_text(pm, "run", "### RUN:")),
    ]
    combined = re.compile(rf'^[ \t]*(?:{"|".join(markers)})\s*(.*?)\s*$', re.M | re.I)
    for m in combined.finditer(text):
        c = _clean_command_line(_strip_md_decor(m.group(1)))
        if c:
            cmds.append(c)
    return list(dict.fromkeys(cmds))


def _parse_inspect_markers(text: str, pm) -> list[str]:
    cmds: list[str] = []
    marker = _marker_pattern(_marker_text(pm, "inspect", "### INSPECT:"))
    combined = re.compile(rf'^[ \t]*{marker}\s*(.*?)\s*$', re.M | re.I)
    for m in combined.finditer(text):
        c = _clean_command_line(_strip_md_decor(m.group(1)))
        if c:
            cmds.append(c)
    return list(dict.fromkeys(cmds))


def _has_done_marker(text: str, pm) -> bool:
    return _marker_re(pm, "done", "### DONE").search(text) is not None


def _parse_simple_markers(text: str, pm, key: str, fallback: str) -> list[str]:
    """Generic one-line marker parser: collects the trimmed argument of every
    occurrence, deduplicated, order-preserving."""
    args: list[str] = []
    for m in _marker_re(pm, key, fallback).finditer(text):
        a = m.group(1).strip().strip("`")
        if a:
            args.append(a)
    return list(dict.fromkeys(args))


def _split_pattern_glob(arg: str) -> tuple[str, str | None]:
    """Split '<pattern> :: <glob>' — the optional '::' separator scopes a
    search to a file glob. Tolerates the bracketed form the models copy from
    the docs ('pattern [:: glob]') and markdown decoration on either side."""
    if "::" in arg:
        pattern, _, glob = arg.partition("::")
        pattern = _strip_md_decor(pattern.rstrip().rstrip("[").rstrip())
        glob = glob.strip().rstrip("]").strip()
        return pattern, (_strip_md_decor(glob) or None)
    return _strip_md_decor(arg), None


_GREP_STYLE_RE = re.compile(
    r'^(?:(?:rg|grep|findstr|select-string)\s+)?(?:-[a-zA-Z]+\s+)*"(?P<pat>[^"]+)"\s*(?P<path>\S+)?\s*$'
)


def _normalize_grep_arg(arg: str) -> tuple[str, str | None]:
    """Models often emit rg/grep-style arguments ('-n "pattern" path') instead
    of the plain '<pattern> [:: <glob>]' form. Detect and translate that so
    the search still works instead of matching the flag text literally."""
    pattern, glob = _split_pattern_glob(arg)
    m = _GREP_STYLE_RE.match(pattern)
    if m:
        pattern = m.group("pat")
        if not glob and m.group("path"):
            glob = m.group("path").replace("\\", "/")
    return pattern, glob


_RANGE_ARG_RE = re.compile(r"^(?P<path>.+?)[\s:]+(?P<start>\d+)\s*[-:,]\s*(?P<end>\d+)\s*$")


def _looks_like_path(arg: str) -> bool:
    """True when a (cleaned) inspect/search argument is really a file path or
    glob, not a shell command — e.g. '**/scraper.py' or 'src/cli.py'."""
    s = arg.strip()
    return bool(s) and " " not in s and ("/" in s or "\\" in s or "." in s)


def parse_actions(text: str, pm) -> dict:
    """Parse every action marker from one coder turn.

    Returns ``{"reads": [path...], "ranges": [arg...], "searches": [arg...],
    "greps": [arg...], "globs": [pattern...], "web_fetches": [url...],
    "web_searches": [query...], "inspects": [cmd...],
    "files": [(path, content)...], "deletes": [path...], "commands": [cmd...],
    "done": bool}``.

    The loop applies categories in a fixed safe order — search/read first,
    then write, then delete, then verify/run — regardless of how the model
    interleaved the markers in its text, because writes must land on disk
    before a verify command can usefully run against them.
    """
    return {
        "reads":        _parse_read_markers(text, pm),
        "ranges":       _parse_simple_markers(text, pm, "read_range", "### READ_RANGE:"),
        "searches":     _parse_simple_markers(text, pm, "search", "### SEARCH:"),
        "greps":        _parse_simple_markers(text, pm, "grep", "### GREP:"),
        "globs":        _parse_simple_markers(text, pm, "glob", "### GLOB:"),
        "web_fetches":  _parse_simple_markers(text, pm, "web_fetch", "### WEB_FETCH:"),
        "web_searches": _parse_simple_markers(text, pm, "web_search", "### WEB_SEARCH:"),
        "inspects":     _parse_inspect_markers(text, pm),
        "files":        _parse_file_changes(text),
        "deletes":      _parse_file_deletions(text),
        "edits":        parse_edit_blocks(text),
        "commands":     _parse_command_markers(text, pm),
        "done":         _has_done_marker(text, pm),
    }


def has_any_action(actions: dict) -> bool:
    """True when the coder turn contained at least one executable action."""
    return any((
        actions.get("reads"), actions.get("ranges"), actions.get("searches"),
        actions.get("greps"), actions.get("globs"), actions.get("web_fetches"),
        actions.get("web_searches"), actions.get("inspects"), actions.get("files"),
        actions.get("deletes"), actions.get("edits"),
        actions.get("commands"), actions.get("done"),
    ))


# ── handlers — each yields UI events, returns an observation dict ────────────

def do_read(project_root: Path, rel_path: str) -> Generator[tuple, None, dict]:
    """``### READ: <path>`` — always allowed; returns a capped excerpt.

    Glob-style arguments (e.g. ``**/scraper.py``) are resolved to the first
    matching project file instead of failing with 'not found'."""
    rel_path = _strip_md_decor(rel_path)
    if any(ch in rel_path for ch in "*?[") :
        matches = glob_files(project_root, rel_path, max_results=5)
        if matches:
            yield ("tool", f"glob_resolve({rel_path!r}) -> {matches[0]}", "tool")
            rel_path = matches[0]
        else:
            yield ("tool", f"read_file({rel_path!r}) -> no file matches that pattern", "tool")
            return {"kind": "read", "path": rel_path, "ok": False,
                    "detail": "no file matches that glob pattern"}
    else:
        # Tolerant resolution: nested layouts make models emit subdir-relative
        # paths (fixtures/x.html instead of starter_project/fixtures/x.html).
        resolved, note = resolve_project_path(project_root, rel_path)
        if resolved is not None:
            if note:
                yield ("tool", f"path_resolve: {note}", "tool")
            rel_path = resolved
    target = (project_root / rel_path).resolve()
    try:
        target.relative_to(project_root.resolve())
    except ValueError:
        yield ("log", json.dumps({
            "type": "file_read", "path": rel_path, "ok": False,
            "detail": "path escapes project root",
        }, ensure_ascii=False), "log")
        yield ("warn", f"read_file({rel_path!r}) -> refused (escapes project root)", "warn")
        return {"kind": "read", "path": rel_path, "ok": False, "detail": "path escapes project root"}
    if not target.exists():
        yield ("log", json.dumps({
            "type": "file_read", "path": rel_path, "ok": False,
            "detail": "file does not exist",
        }, ensure_ascii=False), "log")
        yield ("tool", f"read_file({rel_path!r}) -> not found", "tool")
        return {"kind": "read", "path": rel_path, "ok": False, "detail": "file does not exist"}
    try:
        raw = target.read_text("utf-8", errors="replace")
    except OSError as exc:
        yield ("log", json.dumps({
            "type": "file_read", "path": rel_path, "ok": False,
            "detail": str(exc),
        }, ensure_ascii=False), "log")
        yield ("warn", f"read_file({rel_path!r}) -> {exc}", "warn")
        return {"kind": "read", "path": rel_path, "ok": False, "detail": str(exc)}

    lines = raw.splitlines(keepends=True)
    total_lines = len(lines)

    # Produce a reasonable excerpt (prefer recent lines for "after search" use case) and report exact line range.
    # Keep spirit of old char cap but work in lines so we can report "lines X-Y".
    max_lines = 80
    if total_lines > max_lines:
        start_idx = total_lines - max_lines
        excerpt_lines = lines[start_idx:]
        start_line = start_idx + 1
        end_line = total_lines
        excerpt = "".join(excerpt_lines)
    else:
        start_line = 1
        end_line = total_lines
        excerpt = raw

    # Also respect a soft char cap on the returned content for the obs (old behavior).
    if len(excerpt) > _READ_CAP:
        excerpt = excerpt[-_READ_CAP:]
        # If we truncated the excerpt further, adjust the reported end (best effort).
        # For simplicity we keep the logical line range of the intended excerpt.

    yield ("tool", f"read_file({rel_path!r}) -> {len(raw)} chars (lines {start_line}-{end_line} of {total_lines})", "tool")
    yield ("log", json.dumps({
        "type": "file_read",
        "path": rel_path,
        "ok": True,
        "content": raw,
        "chars": len(raw),
        "sha256": hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest(),
        "excerpt_returned": excerpt,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
    }, ensure_ascii=False), "log")
    return {
        "kind": "read",
        "path": rel_path,
        "ok": True,
        "content": excerpt,
        "chars": len(raw),
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
    }


def do_read_range(project_root: Path, arg: str) -> Generator[tuple, None, dict]:
    """``### READ_RANGE: <path>:<start>-<end>`` — exact line-range read."""
    m = _RANGE_ARG_RE.match(_strip_md_decor(arg))
    if not m:
        yield ("warn", f"read_range({arg!r}) -> bad format (use path:start-end)", "warn")
        return {"kind": "range", "arg": arg, "ok": False,
                "detail": "bad format — use `path:start-end`, e.g. src/app.py:40-90"}
    # Models routinely wrap the path in backticks (`path`) — strip them, plus any
    # stray quotes, so READ_RANGE doesn't fail with a bogus "file does not exist"
    # (session 2026-06-13_20-55-49: every read_range failed on a trailing backtick).
    rel = m.group("path").strip().strip("`\"'").strip()
    start, end = int(m.group("start")), int(m.group("end"))
    rr = read_file_range(project_root, rel, start, end)
    if not rr.ok:
        yield ("log", json.dumps({
            "type": "file_read_range", "path": rel, "ok": False,
            "requested_start": start, "requested_end": end,
            "detail": rr.error,
        }, ensure_ascii=False), "log")
        yield ("warn", f"read_range({rel!r}, {start}-{end}) -> {rr.error}", "warn")
        return {"kind": "range", "path": rel, "ok": False, "detail": rr.error}
    yield ("tool", f"read_range({rel!r}) -> lines {rr.start_line}-{rr.end_line} of {rr.total_lines}", "tool")
    yield ("log", json.dumps({
        "type": "file_read_range",
        "path": rel,
        "ok": True,
        "content": rr.content,
        "requested_start": start,
        "requested_end": end,
        "start_line": rr.start_line,
        "end_line": rr.end_line,
        "total_lines": rr.total_lines,
    }, ensure_ascii=False), "log")
    return {"kind": "range", "path": rel, "ok": True, "content": rr.content,
            "start_line": rr.start_line, "end_line": rr.end_line, "total_lines": rr.total_lines}


def _matches_to_obs(kind: str, query: str, matches, glob: str | None) -> dict:
    listed = [
        {"path": m.rel_path, "line": m.line, "column": m.column, "text": m.text}
        for m in matches
    ]
    return {"kind": kind, "query": query, "glob": glob, "ok": True,
            "count": len(matches), "matches": listed}


def do_search(project_root: Path, arg: str) -> Generator[tuple, None, dict]:
    """``### SEARCH: <text> [:: <glob>]`` — literal, case-insensitive."""
    pattern, glob = _normalize_grep_arg(arg)
    if not pattern:
        return {"kind": "search", "query": arg, "ok": False, "detail": "empty pattern"}
    # A bare path/glob argument means "find this file", not "find this text".
    if glob is None and _looks_like_path(pattern) and any(ch in pattern for ch in "*?[/\\"):
        files = glob_files(project_root, pattern)
        yield ("tool", f"glob_files({pattern!r}) -> {len(files)} files [search arg was a path]", "tool")
        obs = {"kind": "glob", "query": pattern, "ok": True, "count": len(files),
               "files": files}
        yield ("log", json.dumps({"type": "tool_observation", "tool": "glob_files", "observation": obs},
                                  ensure_ascii=False), "log")
        return obs
    # ONE search command that handles literal text, `a|b` alternation, regex,
    # and prose descriptions (keyword fallback). Real runs proved small models
    # will not learn the difference — the runtime finds something useful for
    # every query instead of returning a useless 0.
    matches, strategy = smart_search(project_root, pattern, glob=glob)
    yield ("tool",
           f"smart_search({pattern[:60]!r}{f', glob={glob!r}' if glob else ''}) "
           f"-> {len(matches)} hits [{strategy}]", "tool")
    obs = _matches_to_obs("search", pattern, matches, glob)
    if matches and strategy != "literal":
        obs["detail"] = f"strategy: {strategy}"
    if not matches:
        obs["detail"] = (
            f"{strategy}. SEARCH accepts literal text, `a|b` alternation, regex, "
            "or descriptions (keywords are auto-extracted). Try a concrete code "
            "token, optionally scoped: `### SEARCH: Feldkirch :: **/*.html`"
        )
    yield ("log", json.dumps({"type": "tool_observation", "tool": "smart_search",
                              "strategy": strategy, "observation": obs},
                              ensure_ascii=False), "log")
    return obs


def do_grep(project_root: Path, arg: str) -> Generator[tuple, None, dict]:
    """``### GREP: <regex> [:: <glob>]`` — regex-first search (rg-style
    arguments like ``-n "pattern" path`` are translated automatically);
    falls back to the same smart pipeline as SEARCH on 0 hits/bad regex."""
    pattern, glob = _normalize_grep_arg(arg)
    if not pattern:
        return {"kind": "grep", "query": arg, "ok": False, "detail": "empty regex"}
    result = search_regex(project_root, pattern, glob=glob, case_sensitive=False)
    if isinstance(result, str) or not result:
        # invalid regex OR no hits — run the full smart pipeline instead
        reason = result if isinstance(result, str) else "0 regex hits"
        matches, strategy = smart_search(project_root, pattern, glob=glob)
        yield ("tool", f"smart_search({pattern[:60]!r}) -> {len(matches)} hits [{reason}; {strategy}]", "tool")
        obs = _matches_to_obs("grep", pattern, matches, glob)
        obs["detail"] = f"{reason}; used smart search ({strategy})"
        yield ("log", json.dumps({"type": "tool_observation", "tool": "smart_search",
                                  "strategy": strategy, "observation": obs},
                                  ensure_ascii=False), "log")
        return obs
    yield ("tool", f"search_regex({pattern!r}{f', glob={glob!r}' if glob else ''}) -> {len(result)} hits", "tool")
    obs = _matches_to_obs("grep", pattern, result, glob)
    yield ("log", json.dumps({"type": "tool_observation", "tool": "search_regex", "observation": obs},
                              ensure_ascii=False), "log")
    return obs


def do_glob(project_root: Path, pattern: str) -> Generator[tuple, None, dict]:
    """``### GLOB: <pattern>`` — filename search (e.g. `**/*.test.ts`)."""
    files = glob_files(project_root, pattern)
    yield ("tool", f"glob_files({pattern!r}) -> {len(files)} files", "tool")
    obs = {"kind": "glob", "query": pattern, "ok": True, "count": len(files),
           "files": files}
    yield ("log", json.dumps({"type": "tool_observation", "tool": "glob_files", "observation": obs},
                              ensure_ascii=False), "log")
    return obs


def do_web_fetch(project_root: Path, url: str) -> Generator[tuple, None, dict]:
    """``### WEB_FETCH: <url>`` — guarded HTTP GET + HTML text extraction."""
    res = web_fetch(url, project_dir=project_root / ".kratos")
    if not res.ok:
        yield ("warn", f"web_fetch({url!r}) -> {res.error}", "warn")
        return {"kind": "web_fetch", "url": url, "ok": False, "detail": res.error}
    text = res.text
    if res.content_type.startswith("text/html"):
        text = scrape_text_from_html(text)
    yield ("tool", f"web_fetch({url!r}) -> HTTP {res.status}, {len(text)} chars extracted", "tool")
    obs = {"kind": "web_fetch", "url": url, "ok": True, "status": res.status,
           "content": text, "content_type": res.content_type}
    yield ("log", json.dumps({"type": "tool_observation", "tool": "web_fetch", "observation": obs},
                              ensure_ascii=False), "log")
    return obs


def do_web_search(project_root: Path, query: str) -> Generator[tuple, None, dict]:
    """``### WEB_SEARCH: <query>`` — provider-adapter search with honest errors."""
    results, error = web_search(query, project_dir=project_root / ".kratos")
    if error:
        obs = {"kind": "web_search", "query": query, "ok": False, "detail": error}
        yield ("log", json.dumps({"type": "tool_observation", "tool": "web_search", "observation": obs},
                                  ensure_ascii=False), "log")
        yield ("warn", f"web_search({query!r}) -> {error}", "warn")
        return obs
    yield ("tool", f"web_search({query!r}) -> {len(results)} results", "tool")
    obs = {"kind": "web_search", "query": query, "ok": True,
           "results": [r.to_dict() for r in results]}
    yield ("log", json.dumps({"type": "tool_observation", "tool": "web_search", "observation": obs},
                              ensure_ascii=False), "log")
    return obs


def do_write(
    project_root: Path, rel_path: str, content: str,
    proof: ProvenWork, attempt: int, original_snapshots: dict,
) -> Generator[tuple, None, dict]:
    """``### FILE: <path>`` — gated by ``config.can_write()`` at the call site.

    Nested layouts: when the given path does not exist but EXACTLY ONE
    existing project file matches it as a suffix (fixtures/x.html →
    starter_project/fixtures/x.html), the write goes to that file — models
    routinely emit subproject-relative paths, and silently creating a second
    copy at the wrong root breaks both tests and verification."""
    rel_path = _strip_md_decor(rel_path)
    if not (project_root / rel_path).exists():
        resolved, note = resolve_project_path(project_root, rel_path)
        if resolved is not None and resolved != rel_path:
            yield ("tool", f"path_resolve: {note}", "tool")
            rel_path = resolved
    target = (project_root / rel_path).resolve()
    try:
        target.relative_to(project_root.resolve())
        if rel_path not in original_snapshots:
            original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None

        # Capture old line count for delta reporting (user-visible edit stats).
        old_text = original_snapshots.get(rel_path)
        if old_text is None and target.exists():
            try:
                old_text = target.read_text("utf-8")
            except Exception:
                old_text = None
        old_lines = len(old_text.splitlines()) if old_text else 0

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        file_bytes = target.stat().st_size
        new_sha = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        old_sha = (
            hashlib.sha256(old_text.encode("utf-8", "replace")).hexdigest()
            if old_text is not None else None
        )

        new_lines = len(content.splitlines())
        lines_added = max(0, new_lines - old_lines)
        lines_removed = max(0, old_lines - new_lines)

        if rel_path not in proof.files_changed:
            proof.files_changed.append(rel_path)
        proof.file_checks.append({
            "path": rel_path,
            "operation": "write",
            "ok": True,
            "bytes": file_bytes,
            "old_line_count": old_lines,
            "new_line_count": new_lines,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
        })

        delta_str = f" (-{lines_removed} +{lines_added} lines)" if (lines_removed or lines_added) else ""
        yield ("tool", f"write_file({rel_path!r}) -> {file_bytes} bytes{delta_str}", "tool")
        yield ("log", json.dumps({
            "type": "file_write",
            "path": rel_path,
            "ok": True,
            "content": content,
            "previous_content": old_text,
            "size_bytes": file_bytes,
            "old_line_count": old_lines,
            "new_line_count": new_lines,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "sha256": new_sha,
            "previous_sha256": old_sha,
            "attempt": attempt,
        }, ensure_ascii=False), "log")
        return {
            "kind": "file",
            "path": rel_path,
            "ok": True,
            "bytes": file_bytes,
            "old_line_count": old_lines,
            "new_line_count": new_lines,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
        }
    except (ValueError, OSError) as exc:
        proof.file_checks.append({"path": rel_path, "operation": "write", "ok": False, "error": str(exc)})
        yield ("log", json.dumps({
            "type": "file_write",
            "path": rel_path,
            "ok": False,
            "content": content,
            "detail": str(exc),
            "attempt": attempt,
        }, ensure_ascii=False), "log")
        yield ("error", f"File write failed for {rel_path}: {exc}", "error")
        return {"kind": "file", "path": rel_path, "ok": False, "detail": str(exc)}


def do_delete(
    project_root: Path, rel_path: str,
    proof: ProvenWork, attempt: int, original_snapshots: dict,
) -> Generator[tuple, None, dict]:
    """``### DELETE: <path>`` — gated by ``config.can_delete()`` at the call site.

    Uses the same path resolution as ``do_write`` so a model that emits an
    extra leading directory (e.g. the repeated project-root name) still deletes
    the real file instead of silently no-op'ing on a non-existent nested path."""
    rel_path = _strip_md_decor(rel_path)
    if not (project_root / rel_path).exists():
        resolved, note = resolve_project_path(project_root, rel_path)
        if resolved is not None and resolved != rel_path:
            yield ("tool", f"path_resolve: {note}", "tool")
            rel_path = resolved
    target = (project_root / rel_path).resolve()
    try:
        target.relative_to(project_root.resolve())
        if rel_path not in original_snapshots:
            original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None
        previous_content = original_snapshots.get(rel_path)
        previous_sha = (
            hashlib.sha256(previous_content.encode("utf-8", "replace")).hexdigest()
            if previous_content is not None else None
        )
        existed = target.exists()
        if target.exists():
            target.unlink()
        ok = not target.exists()
        if rel_path not in proof.files_changed:
            proof.files_changed.append(rel_path)
        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": ok})
        yield ("tool", f"delete_file({rel_path!r}) -> {'ok' if ok else 'FAILED'}", "tool")
        yield ("log", json.dumps({
            "type": "file_delete",
            "path": rel_path,
            "ok": ok,
            "existed": existed,
            "previous_content": previous_content,
            "previous_sha256": previous_sha,
            "attempt": attempt,
        }, ensure_ascii=False), "log")
        return {"kind": "delete", "path": rel_path, "ok": ok}
    except (ValueError, OSError) as exc:
        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": False, "error": str(exc)})
        yield ("log", json.dumps({
            "type": "file_delete",
            "path": rel_path,
            "ok": False,
            "detail": str(exc),
            "attempt": attempt,
        }, ensure_ascii=False), "log")
        yield ("error", f"File delete failed for {rel_path}: {exc}", "error")
        return {"kind": "delete", "path": rel_path, "ok": False, "detail": str(exc)}


def do_edit(
    project_root: Path, rel_path: str, search: str, replace: str,
    proof: ProvenWork, attempt: int, original_snapshots: dict,
) -> Generator[tuple, None, dict]:
    """``### EDIT: <path>`` with a SEARCH/REPLACE block — a TARGETED change to an
    existing file. Far safer for weak models than re-emitting the whole file
    (full rewrites routinely regress code the previous turn fixed). The SEARCH
    text must match the current on-disk content (exactly, or modulo trailing
    whitespace / CRLF); otherwise the file is left untouched and the model is
    told to re-READ instead of silently corrupting it."""
    rel_path = _strip_md_decor(rel_path)
    if not (project_root / rel_path).exists():
        resolved, note = resolve_project_path(project_root, rel_path)
        if resolved is not None and resolved != rel_path:
            yield ("tool", f"path_resolve: {note}", "tool")
            rel_path = resolved
    target = (project_root / rel_path).resolve()
    try:
        target.relative_to(project_root.resolve())
    except ValueError:
        yield ("warn", f"edit_file({rel_path!r}) -> refused (escapes project root)", "warn")
        return {"kind": "edit", "path": rel_path, "ok": False, "detail": "path escapes project root"}
    if not target.is_file():
        yield ("warn", f"edit_file({rel_path!r}) -> file does not exist (use ### FILE to create)", "warn")
        return {"kind": "edit", "path": rel_path, "ok": False, "skipped": True,
                "detail": "file does not exist — use ### FILE to create it"}
    try:
        before = target.read_text("utf-8", errors="replace")
    except OSError as exc:
        yield ("warn", f"edit_file({rel_path!r}) -> {exc}", "warn")
        return {"kind": "edit", "path": rel_path, "ok": False, "detail": str(exc)}

    new_content, status = apply_search_replace(before, search, replace)
    if new_content == before and status in ("not_found", "empty_search", "ambiguous"):
        detail = {
            "not_found": ("SEARCH block not found verbatim — ### READ the file and copy the "
                          "exact current text (with indentation) into the SEARCH block."),
            "empty_search": "empty SEARCH block.",
            "ambiguous": "SEARCH matched multiple places — add surrounding lines to make it unique.",
        }[status]
        yield ("warn", f"edit_file({rel_path!r}) -> {status}", "warn")
        return {"kind": "edit", "path": rel_path, "ok": False, "skipped": True,
                "detail": detail, "status": status}
    if new_content == before:
        yield ("tool", f"edit_file({rel_path!r}) -> no change (replacement already present)", "tool")
        return {"kind": "edit", "path": rel_path, "ok": True, "status": "noop",
                "detail": "no change (content already matched replacement)"}

    if rel_path not in original_snapshots:
        original_snapshots[rel_path] = before
    try:
        target.write_text(new_content, encoding="utf-8")
    except OSError as exc:
        yield ("error", f"edit_file({rel_path!r}) -> {exc}", "error")
        return {"kind": "edit", "path": rel_path, "ok": False, "detail": str(exc)}

    from .diffing import diff_stats
    st = diff_stats(before, new_content)
    if rel_path not in proof.files_changed:
        proof.files_changed.append(rel_path)
    file_bytes = target.stat().st_size
    new_sha = hashlib.sha256(new_content.encode("utf-8", "replace")).hexdigest()
    proof.file_checks.append({
        "path": rel_path, "operation": "edit", "ok": True, "bytes": file_bytes,
        "lines_added": st.added, "lines_removed": st.removed, "match": status,
    })
    if status == "ambiguous":
        yield ("warn", f"edit_file({rel_path!r}) -> ambiguous; edited the FIRST occurrence", "warn")
    yield ("tool", f"edit_file({rel_path!r}) -> {st.as_suffix()} lines [{status}]", "tool")
    yield ("log", json.dumps({
        "type": "file_write", "path": rel_path, "ok": True, "operation": "edit",
        "content": new_content, "previous_content": before, "size_bytes": file_bytes,
        "lines_added": st.added, "lines_removed": st.removed, "match": status,
        "sha256": new_sha, "attempt": attempt,
    }, ensure_ascii=False), "log")
    return {"kind": "edit", "path": rel_path, "ok": True, "status": status,
            "bytes": file_bytes, "content": new_content,
            "lines_added": st.added, "lines_removed": st.removed}


def do_command(
    agent, project_root: Path, cmd_registry: CommandRegistry, raw_cmd: str,
    proof: ProvenWork,
) -> Generator[tuple, None, dict]:
    """``### VERIFY: <cmd>`` / ``### RUN: <cmd>`` — guarded by the *exact* same
    safety gates the stepwise coder used. Anything that fails a gate is
    reported back as a skip-observation and is **never executed**."""
    cmd = _clean_command_line(raw_cmd)
    if not cmd:
        return {"kind": "command", "cmd": raw_cmd, "ok": False, "skipped": True, "detail": "empty command"}

    if not _is_safe_verification_command(cmd, project_root):
        yield ("warn", f"skipped `{cmd}` — not a recognized safe build/test command", "warn")
        return {"kind": "command", "cmd": cmd, "ok": False, "skipped": True,
                "detail": "not a recognized safe build/test command"}

    if cmd_registry.is_toolchain_mismatch(cmd):
        tc = _command_toolchain(cmd)
        yield ("warn",
               f"skipped `{cmd}` (toolchain `{tc}` not in project: "
               f"{', '.join(sorted(cmd_registry.toolchains))})", "warn")
        return {"kind": "command", "cmd": cmd, "ok": False, "skipped": True,
                "detail": f"toolchain `{tc}` not used by this project"}

    # Nested layouts: commands of a toolchain that lives in a subdir run there
    # (otherwise `python -m pytest` in the parent root fails with import
    # errors the model cannot fix — the root cause of a real failed run).
    rel_cwd = cmd_registry.cwd_for(cmd)
    check_root = (project_root / rel_cwd) if rel_cwd else project_root
    missing = _missing_command_paths(cmd, check_root)
    if missing:
        yield ("warn", f"skipped `{cmd}` — file(s) not found: {', '.join(missing)}", "warn")
        return {"kind": "command", "cmd": cmd, "ok": False, "skipped": True,
                "detail": f"references missing file(s): {', '.join(missing)}"}

    is_test = _is_test_verification_command(cmd)
    vcmd = VerificationCommand(cmd=cmd, purpose="coder-requested verification",
                               source="coder-loop", is_test=is_test, cwd=rel_cwd)
    # Emit a clear, visible command echo so the actual Befehl being executed is printed
    # (not just the dim tool meta). Handled in cli.py / tui.py for non-dim "$ cmd" style.
    yield ("command", cmd, "command_echo")
    yield ("tool", f"run_command({cmd!r}) -> {vcmd.purpose}", "tool")
    res = agent._run_verification_command(vcmd)
    proof.commands.append(res)
    proof.commands_planned.append({"cmd": cmd, "purpose": vcmd.purpose,
                                   "source": vcmd.source, "is_test": is_test})
    status = "ok" if res["exit_code"] == 0 else "FAILED"
    yield ("tool", f"verify_command({cmd!r}) -> {status} exit={res['exit_code']}", "tool")
    yield ("log", json.dumps({
        "type": "build_test", "cmd": vcmd.cmd, "purpose": vcmd.purpose,
        "source": vcmd.source, "is_test": vcmd.is_test,
        "exit_code": res["exit_code"],
        "duration_seconds": res.get("duration_seconds"),
        "output": res.get("output", ""),
        "stdout": res.get("stdout", ""),
        "stderr": res.get("stderr", ""),
        "cwd": res.get("cwd"),
        "shell": res.get("shell"),
        "timeout_seconds": res.get("timeout_seconds"),
        "timed_out": res.get("timed_out"),
        "blocked": res.get("blocked"),
        "block_reason": res.get("block_reason"),
        "result": res,
    }, ensure_ascii=False), "log")
    return {
        "kind": "command", "cmd": cmd, "ok": res["exit_code"] == 0, "skipped": False,
        "exit_code": res["exit_code"], "is_test": is_test,
        "duration_seconds": res.get("duration_seconds"),
        "output": res.get("output", "")[-_OUTPUT_TAIL:],
    }


def do_inspect(
    agent, project_root: Path, raw_cmd: str,
) -> Generator[tuple, None, dict]:
    """``### INSPECT: <cmd>`` — read-only shell diagnostics only.

    This runs through the agent's dedicated read-only shell helper and rejects
    anything that looks write-capable or shell-escape oriented.
    """
    cmd = _clean_command_line(_strip_md_decor(raw_cmd))
    if not cmd:
        return {"kind": "inspect", "cmd": raw_cmd, "ok": False, "skipped": True, "detail": "empty command"}

    # Models frequently put a PATH or GLOB after ### INSPECT (e.g.
    # "### INSPECT: **/scraper.py**"). Redirect that to a file read instead of
    # rejecting the turn — the intent is obviously "show me that file".
    if _looks_like_path(cmd):
        yield ("tool", f"inspect_redirect({cmd!r}) -> treating as file read", "tool")
        obs = yield from do_read(project_root, cmd)
        return obs

    if not _is_safe_inspect_command(cmd, project_root):
        # Prose descriptions ("Search for the exact HTML structure", "Check if
        # the module can be imported") are the single most common INSPECT
        # misuse in real runs. Instead of burning the turn with a rejection,
        # redirect the description to the smart search pipeline.
        # NEVER redirect write-capable/blocked commands — those stay skipped.
        from ..verification import _INSPECT_BLOCKED_RE
        is_blocked = bool(_INSPECT_BLOCKED_RE.search(cmd))
        if not is_blocked and " " in cmd and len(cmd.split()) >= 3:
            yield ("tool", f"inspect_redirect({cmd[:60]!r}) -> smart search (prose argument)", "tool")
            obs = yield from do_search(project_root, cmd)
            if isinstance(obs, dict):
                obs.setdefault("detail", "")
                obs["detail"] = (
                    "NOTE: INSPECT expects a read-only shell command; your description "
                    "was run through smart search instead. " + str(obs.get("detail") or "")
                ).strip()
            return obs
        yield ("warn", f"skipped inspect `{cmd}` — not a recognized read-only diagnostic command", "warn")
        return {
            "kind": "inspect", "cmd": cmd, "ok": False, "skipped": True,
            "detail": ("not a recognized read-only diagnostic command — use "
                       "### SEARCH/### GREP for text search or ### READ for files"),
        }

    if not hasattr(agent, "_run_readonly_command"):
        yield ("error", "inspect helper missing on agent: _run_readonly_command", "error")
        return {
            "kind": "inspect", "cmd": cmd, "ok": False, "skipped": True,
            "detail": "agent has no readonly command runner",
        }

    # Clear visible echo for INSPECT commands too (read-only diagnostics the coder requests).
    yield ("command", cmd, "command_echo")
    yield ("tool", f"inspect_command({cmd!r}) -> readonly shell", "tool")
    res = agent._run_readonly_command(cmd, project_root)
    status = "ok" if res["exit_code"] == 0 else "FAILED"
    yield ("tool", f"inspect_result({cmd!r}) -> {status} exit={res['exit_code']}", "tool")
    yield ("log", json.dumps({
        "type": "inspect",
        "cmd": cmd,
        "exit_code": res["exit_code"],
        "duration_seconds": res.get("duration_seconds"),
        "output": res.get("output", ""),
        "stdout": res.get("stdout", ""),
        "stderr": res.get("stderr", ""),
        "cwd": res.get("cwd"),
        "shell": res.get("shell"),
        "timeout_seconds": res.get("timeout_seconds"),
        "timed_out": res.get("timed_out"),
        "blocked": res.get("blocked"),
        "block_reason": res.get("block_reason"),
        "result": res,
    }, ensure_ascii=False), "log")
    return {
        "kind": "inspect", "cmd": cmd, "ok": res["exit_code"] == 0, "skipped": False,
        "exit_code": res["exit_code"],
        "duration_seconds": res.get("duration_seconds"),
        "output": res.get("output", "")[-_OUTPUT_TAIL:],
    }


# ── observation rendering — folds results back into the next coder turn ─────

def format_observation(observations: list[dict], pm) -> str:
    """Render one iteration's observation dicts into the text turn fed back
    to the coder as its next user message. Templates come from the
    ``observation_*`` snippets in ``prompts_default.json`` (overridable)."""
    lines = [pm.get_snippet("observation_header") or "OBSERVATION:"]
    for obs in observations:
        kind = obs.get("kind")
        if kind == "read":
            if obs.get("ok"):
                tmpl = pm.get_snippet("observation_file_read") or "  --- current content of {path} (lines {start_line}-{end_line} of {total_lines}) ---\n{content}"
                lines.append(tmpl.format(
                    path=obs["path"],
                    content=obs.get("content", ""),
                    start_line=obs.get("start_line", "?"),
                    end_line=obs.get("end_line", "?"),
                    total_lines=obs.get("total_lines", "?"),
                ))
            else:
                tmpl = pm.get_snippet("observation_file_read_missing") or "  {path} does not exist on disk yet."
                lines.append(tmpl.format(path=obs["path"]))
        elif kind == "file":
            if obs.get("ok"):
                delta = ""
                if "lines_added" in obs or "lines_removed" in obs:
                    delta = f" (-{obs.get('lines_removed', 0)} +{obs.get('lines_added', 0)} lines)"
                tmpl = pm.get_snippet("observation_file_write") or "  wrote {path} -> {bytes} bytes{delta}"
                lines.append(tmpl.format(
                    path=obs["path"],
                    bytes=obs.get("bytes", 0),
                    delta=delta,
                ))
            else:
                tmpl = pm.get_snippet("observation_file_write_failed") or "  FAILED to write {path}: {error}"
                lines.append(tmpl.format(path=obs["path"], error=obs.get("detail", "?")))
        elif kind == "delete":
            tmpl = pm.get_snippet("observation_file_delete") or "  deleted {path} -> {status}"
            lines.append(tmpl.format(path=obs["path"], status="ok" if obs.get("ok") else "FAILED"))
        elif kind == "command":
            if obs.get("skipped"):
                tmpl = pm.get_snippet("observation_command_skipped") or "  skipped `{cmd}` -> {reason}"
                lines.append(tmpl.format(cmd=obs["cmd"], reason=obs.get("detail", "")))
            else:
                tmpl = pm.get_snippet("observation_command") or "  ran `{cmd}` -> exit={exit_code}{test_tag}\n{output}"
                # When a diagnosis is attached (failed command), surface it ABOVE
                # the raw output so weak models act on the concrete fix, not the
                # traceback. This is the per-item feedback in the structured loop.
                _out = obs.get("output", "")
                _detail = obs.get("detail")
                if _detail:
                    _out = f"{_detail}\n\n--- raw output (tail) ---\n{_out}"
                lines.append(tmpl.format(
                    cmd=obs["cmd"], exit_code=obs.get("exit_code"),
                    test_tag=" [TEST]" if obs.get("is_test") else "",
                    output=_out,
                ))
        elif kind == "inspect":
            if obs.get("skipped"):
                tmpl = pm.get_snippet("observation_inspect_skipped") or "  skipped inspect `{cmd}` -> {reason}"
                lines.append(tmpl.format(cmd=obs["cmd"], reason=obs.get("detail", "")))
            else:
                tmpl = pm.get_snippet("observation_inspect") or "  inspected `{cmd}` -> exit={exit_code}\n{output}"
                lines.append(tmpl.format(
                    cmd=obs["cmd"], exit_code=obs.get("exit_code"),
                    output=obs.get("output", ""),
                ))
        elif kind == "range":
            if obs.get("ok"):
                lines.append(
                    f"  --- {obs['path']} lines {obs.get('start_line')}-{obs.get('end_line')} "
                    f"of {obs.get('total_lines')} ---\n{obs.get('content', '')}"
                )
            else:
                lines.append(f"  READ_RANGE failed: {obs.get('detail', '?')}")
        elif kind in ("search", "grep"):
            if not obs.get("ok"):
                lines.append(f"  {kind.upper()} failed: {obs.get('detail', '?')}")
            elif not obs.get("count"):
                hint = obs.get("detail") or ""
                lines.append(f"  {kind.upper()} `{obs.get('query')}` -> 0 hits" + (f" — {hint}" if hint else ""))
            else:
                lines.append(f"  {kind.upper()} `{obs.get('query')}` -> {obs['count']} hits:")
                for m in obs.get("matches", [])[:20]:
                    lines.append(f"    {m['path']}:{m['line']}:{m['column']}  {m['text']}")
        elif kind == "glob":
            files = obs.get("files", [])
            lines.append(f"  GLOB `{obs.get('query')}` -> {obs.get('count', 0)} files:")
            lines.extend(f"    {f}" for f in files[:40])
        elif kind == "web_fetch":
            if obs.get("ok"):
                lines.append(f"  WEB_FETCH {obs.get('url')} -> HTTP {obs.get('status')}\n{obs.get('content', '')[:4000]}")
            else:
                lines.append(f"  WEB_FETCH {obs.get('url')} failed: {obs.get('detail', '?')}")
        elif kind == "web_search":
            if obs.get("ok"):
                lines.append(f"  WEB_SEARCH `{obs.get('query')}` results:")
                for r in obs.get("results", [])[:8]:
                    lines.append(f"    - {r.get('title')}\n      {r.get('url')}\n      {r.get('snippet')}")
            else:
                lines.append(f"  WEB_SEARCH `{obs.get('query')}` failed: {obs.get('detail', '?')}")
        elif kind == "edit":
            if obs.get("ok"):
                lines.append(f"  EDIT {obs.get('path')} -> +{obs.get('lines_added',0)}/-{obs.get('lines_removed',0)} [{obs.get('status')}]")
            else:
                lines.append(f"  EDIT {obs.get('path')} FAILED: {obs.get('detail','?')}")
        elif kind == "note":
            detail = obs.get("detail") or obs.get("output") or ""
            if detail:
                lines.append(f"  {detail}")
        else:
            detail = obs.get("detail") or obs.get("output") or ""
            if detail:
                lines.append(f"  {detail}")
    return "\n".join(lines)
