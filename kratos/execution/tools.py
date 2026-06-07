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
import re
from pathlib import Path
from typing import Generator

from ..verification import (
    VerificationCommand,
    CommandRegistry,
    ProvenWork,
    _clean_command_line,
    _is_safe_verification_command,
    _is_test_verification_command,
    _missing_command_paths,
    _command_toolchain,
)
from .parsing import _parse_file_changes, _parse_file_deletions

_READ_CAP = 2500       # chars — matches the stepwise pre-step disk-read excerpt cap
_OUTPUT_TAIL = 600     # chars of command output folded into the observation


# ── marker discovery (read from prompts JSON at call time, tolerant parsing) ──

def _marker_pattern(marker: str) -> str:
    marker = (marker or "").strip()
    if marker.endswith(":"):
        marker = marker[:-1].rstrip()
    escaped = re.escape(marker).replace(r"\ ", r"\s+")
    return rf"{escaped}\s*:?"


def _marker_re(pm, key: str, fallback: str) -> "re.Pattern[str]":
    marker = _marker_pattern(pm.get_marker(key) or fallback)
    return re.compile(rf'^[ \t]*{marker}\s*(.*?)\s*$', re.M | re.I)


def _parse_read_markers(text: str, pm) -> list[str]:
    paths: list[str] = []
    for m in _marker_re(pm, "read", "### READ:").finditer(text):
        p = m.group(1).strip().strip("`")
        if p and not any(ch in p for ch in "\r\n<>"):
            paths.append(p)
    return list(dict.fromkeys(paths))


def _parse_command_markers(text: str, pm) -> list[str]:
    cmds: list[str] = []
    markers = [
        _marker_pattern(pm.get_marker("verify") or "### VERIFY:"),
        _marker_pattern(pm.get_marker("run") or "### RUN:"),
    ]
    combined = re.compile(rf'^[ \t]*(?:{"|".join(markers)})\s*(.*?)\s*$', re.M | re.I)
    for m in combined.finditer(text):
        c = _clean_command_line(m.group(1))
        if c:
            cmds.append(c)
    return list(dict.fromkeys(cmds))


def _has_done_marker(text: str, pm) -> bool:
    return _marker_re(pm, "done", "### DONE").search(text) is not None


def parse_actions(text: str, pm) -> dict:
    """Parse every action marker from one coder turn.

    Returns ``{"reads": [path...], "files": [(path, content)...],
    "deletes": [path...], "commands": [cmd...], "done": bool}``.

    The loop applies categories in a fixed safe order — read, then write, then
    delete, then verify/run — regardless of how the model interleaved the
    markers in its text, because writes must land on disk before a verify
    command can usefully run against them.
    """
    return {
        "reads":    _parse_read_markers(text, pm),
        "files":    _parse_file_changes(text),
        "deletes":  _parse_file_deletions(text),
        "commands": _parse_command_markers(text, pm),
        "done":     _has_done_marker(text, pm),
    }


# ── handlers — each yields UI events, returns an observation dict ────────────

def do_read(project_root: Path, rel_path: str) -> Generator[tuple, None, dict]:
    """``### READ: <path>`` — always allowed; returns a capped excerpt."""
    target = (project_root / rel_path).resolve()
    try:
        target.relative_to(project_root.resolve())
    except ValueError:
        yield ("warn", f"read_file({rel_path!r}) -> refused (escapes project root)", "warn")
        return {"kind": "read", "path": rel_path, "ok": False, "detail": "path escapes project root"}
    if not target.exists():
        yield ("tool", f"read_file({rel_path!r}) -> not found", "tool")
        return {"kind": "read", "path": rel_path, "ok": False, "detail": "file does not exist"}
    try:
        raw = target.read_text("utf-8", errors="replace")
    except OSError as exc:
        yield ("warn", f"read_file({rel_path!r}) -> {exc}", "warn")
        return {"kind": "read", "path": rel_path, "ok": False, "detail": str(exc)}
    excerpt = raw[-_READ_CAP:] if len(raw) > _READ_CAP else raw
    yield ("tool", f"read_file({rel_path!r}) -> {len(raw)} chars", "tool")
    return {"kind": "read", "path": rel_path, "ok": True, "content": excerpt, "chars": len(raw)}


def do_write(
    project_root: Path, rel_path: str, content: str,
    proof: ProvenWork, attempt: int, original_snapshots: dict,
) -> Generator[tuple, None, dict]:
    """``### FILE: <path>`` — gated by ``config.can_write()`` at the call site."""
    target = (project_root / rel_path).resolve()
    try:
        target.relative_to(project_root.resolve())
        if attempt == 0 and rel_path not in original_snapshots:
            original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        file_bytes = target.stat().st_size
        if rel_path not in proof.files_changed:
            proof.files_changed.append(rel_path)
        proof.file_checks.append({"path": rel_path, "operation": "write", "ok": True, "bytes": file_bytes})
        yield ("tool", f"write_file({rel_path!r}) -> {file_bytes} bytes", "tool")
        return {"kind": "file", "path": rel_path, "ok": True, "bytes": file_bytes}
    except (ValueError, OSError) as exc:
        proof.file_checks.append({"path": rel_path, "operation": "write", "ok": False, "error": str(exc)})
        yield ("error", f"File write failed for {rel_path}: {exc}", "error")
        return {"kind": "file", "path": rel_path, "ok": False, "detail": str(exc)}


def do_delete(
    project_root: Path, rel_path: str,
    proof: ProvenWork, attempt: int, original_snapshots: dict,
) -> Generator[tuple, None, dict]:
    """``### DELETE: <path>`` — gated by ``config.can_delete()`` at the call site."""
    target = (project_root / rel_path).resolve()
    try:
        target.relative_to(project_root.resolve())
        if attempt == 0 and rel_path not in original_snapshots:
            original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None
        if target.exists():
            target.unlink()
        ok = not target.exists()
        if rel_path not in proof.files_changed:
            proof.files_changed.append(rel_path)
        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": ok})
        yield ("tool", f"delete_file({rel_path!r}) -> {'ok' if ok else 'FAILED'}", "tool")
        return {"kind": "delete", "path": rel_path, "ok": ok}
    except (ValueError, OSError) as exc:
        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": False, "error": str(exc)})
        yield ("error", f"File delete failed for {rel_path}: {exc}", "error")
        return {"kind": "delete", "path": rel_path, "ok": False, "detail": str(exc)}


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

    if not _is_safe_verification_command(cmd):
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

    missing = _missing_command_paths(cmd, project_root)
    if missing:
        yield ("warn", f"skipped `{cmd}` — file(s) not found: {', '.join(missing)}", "warn")
        return {"kind": "command", "cmd": cmd, "ok": False, "skipped": True,
                "detail": f"references missing file(s): {', '.join(missing)}"}

    is_test = _is_test_verification_command(cmd)
    vcmd = VerificationCommand(cmd=cmd, purpose="coder-requested verification",
                               source="coder-loop", is_test=is_test)
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
        "output": res.get("output", "")[-3000:],
    }), "log")
    return {
        "kind": "command", "cmd": cmd, "ok": res["exit_code"] == 0, "skipped": False,
        "exit_code": res["exit_code"], "is_test": is_test,
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
                tmpl = pm.get_snippet("observation_file_read") or "  --- current content of {path} ---\n{content}"
                lines.append(tmpl.format(path=obs["path"], content=obs.get("content", "")))
            else:
                tmpl = pm.get_snippet("observation_file_read_missing") or "  {path} does not exist on disk yet."
                lines.append(tmpl.format(path=obs["path"]))
        elif kind == "file":
            if obs.get("ok"):
                tmpl = pm.get_snippet("observation_file_write") or "  wrote {path} -> {bytes} bytes"
                lines.append(tmpl.format(path=obs["path"], bytes=obs.get("bytes", 0)))
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
                lines.append(tmpl.format(
                    cmd=obs["cmd"], exit_code=obs.get("exit_code"),
                    test_tag=" [TEST]" if obs.get("is_test") else "",
                    output=obs.get("output", ""),
                ))
        elif kind == "done":
            if not obs.get("ok"):
                lines.append(pm.get_snippet("observation_done_premature") or
                             "You emitted ### DONE before a test command passed. Keep working.")
        elif kind == "note":
            lines.append(str(obs.get("detail", "")))
    return "\n".join(lines)
