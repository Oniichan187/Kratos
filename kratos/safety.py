"""SafetyGuard — central gate for dangerous commands and out-of-project paths.

Every shell-executing surface in Kratos (ShellRunner, verification runner,
inspect runner) must consult :func:`check_command` before executing anything.
Every file-writing surface must consult :func:`check_path`.

Design goals:
  - Windows-first (PowerShell + CMD patterns), but bash patterns included.
  - Block destructive operations (format, recursive delete outside a single
    explicit file, registry edits, shutdown, credential access).
  - Block download-and-execute and shell-escape patterns.
  - Never raise on weird input — unknown == not explicitly dangerous, the
    allowlist gates elsewhere still apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SafetyVerdict",
    "check_command",
    "check_path",
    "is_dangerous_command",
]


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # truthy == allowed
        return self.allowed


# ── dangerous command patterns ────────────────────────────────────────────────
# Each entry: (compiled regex, human-readable reason). Case-insensitive.
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Drive / disk destruction
    (re.compile(r"\bformat(?:\.com)?\s+[a-z]:", re.I), "drive formatting"),
    (re.compile(r"\bmkfs(\.\w+)?\b", re.I), "filesystem creation"),
    (re.compile(r"\bdd\b.*\bof=/dev/", re.I), "raw device write"),
    (re.compile(r"\bdiskpart\b", re.I), "disk partitioning"),
    # Recursive / forced deletion
    (re.compile(r"\bdel\b.*(/s\b.*/q\b|/q\b.*/s\b)", re.I), "recursive forced delete (del /s /q)"),
    (re.compile(r"\brd\b\s+/s|\brmdir\b\s+/s", re.I), "recursive directory removal"),
    (re.compile(r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)[a-z]*\b", re.I), "rm -rf"),
    (re.compile(r"\brm\s+-rf?\s+[/~]", re.I), "rm targeting root/home"),
    (re.compile(r"remove-item\b(?=.*(-recurse|-force))(?!.*-whatif)", re.I),
     "Remove-Item -Recurse/-Force"),
    # System state
    (re.compile(r"\bshutdown\b|\brestart-computer\b|\bstop-computer\b|\breboot\b", re.I), "system shutdown/reboot"),
    (re.compile(r"\breg(?:\.exe)?\s+(delete|add)\b|\bremove-itemproperty\b.*\bhk(lm|cu)", re.I), "registry modification"),
    (re.compile(r"\bschtasks\b|\bnew-scheduledtask", re.I), "scheduled task persistence"),
    (re.compile(r"\bnet\s+user\b|\bnet\s+localgroup\b", re.I), "account manipulation"),
    (re.compile(r"\bbcdedit\b|\bvssadmin\b\s+delete", re.I), "boot config / shadow copy tampering"),
    # Download-and-execute / shell escape
    (re.compile(r"invoke-expression|\biex\b", re.I), "Invoke-Expression on dynamic input"),
    (re.compile(r"(downloadstring|downloadfile)\s*\(", re.I), "WebClient download"),
    (re.compile(r"\b(curl|wget|iwr|invoke-webrequest|invoke-restmethod|irm)\b.*\|\s*"
                r"(sh|bash|powershell|pwsh|iex|python|cmd)\b", re.I),
     "download piped to interpreter"),
    (re.compile(r"\bstart-process\b.*(-windowstyle\s+hidden|-verb\s+runas)", re.I),
     "hidden/elevated Start-Process"),
    (re.compile(r"-encodedcommand\b|\bfrombase64string\b", re.I), "encoded command execution"),
    # Credentials / exfiltration
    (re.compile(r"\bmimikatz\b|\blsass\b|\bsekurlsa\b", re.I), "credential theft tooling"),
    (re.compile(r"\bget-credential\b|\bcmdkey\b\s+/list", re.I), "credential access"),
    (re.compile(r"(\$env:|%)(\w*(token|secret|password|api_?key)\w*)(%|\b)", re.I),
     "secret environment variable access"),
    # Privilege / ownership
    (re.compile(r"\btakeown\b|\bicacls\b.*\bgrant\b", re.I), "ownership/ACL change"),
    (re.compile(r"\bsudo\b\s+rm|\bchmod\s+777\s+/", re.I), "privileged destructive op"),
    # Fork bombs / obvious junk
    (re.compile(r":\(\)\s*\{\s*:\|\:&\s*\};", re.I), "fork bomb"),
]

# Paths that must never be written/deleted even when inside a project tree
_FORBIDDEN_PATH_PARTS = {".git/objects", ".git/refs", ".git/HEAD"}


def is_dangerous_command(cmd: str) -> str | None:
    """Return the human-readable reason if *cmd* matches a dangerous pattern,
    else None."""
    if not cmd or not cmd.strip():
        return None
    normalized = " ".join(cmd.split())
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(normalized):
            return reason
    return None


def check_command(cmd: str) -> SafetyVerdict:
    """Gate a shell command. Blocked commands are NEVER executed."""
    reason = is_dangerous_command(cmd)
    if reason is not None:
        return SafetyVerdict(False, f"blocked dangerous command ({reason})")
    return SafetyVerdict(True)


def check_path(path: Path | str, project_root: Path) -> SafetyVerdict:
    """Gate a file write/delete target: must stay inside *project_root* and
    must not touch git internals."""
    try:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (project_root / resolved).resolve()
        else:
            resolved = resolved.resolve()
        resolved.relative_to(project_root.resolve())
    except (ValueError, OSError):
        return SafetyVerdict(False, "path escapes project root")
    rel = resolved.relative_to(project_root.resolve()).as_posix()
    for forbidden in _FORBIDDEN_PATH_PARTS:
        if rel.startswith(forbidden):
            return SafetyVerdict(False, f"path touches protected location ({forbidden})")
    return SafetyVerdict(True)
