"""ShellRunner — safe, observable command execution for Kratos.

Windows-first: PowerShell (pwsh > powershell) and CMD are first-class;
bash is used as fallback on POSIX hosts (Kratos also runs under WSL).

Every command:
  - passes through :func:`kratos.safety.check_command` (blocked == never run),
  - runs with a hard timeout and explicit working directory,
  - captures stdout and stderr SEPARATELY plus the exit code,
  - returns a plain dict (``CommandResult``) so callers/loggers/reporters can
    serialize it without ceremony.

No shell=True for powershell/cmd/bash invocations — the interpreter binary is
invoked directly with the command as an argument, which avoids a second layer
of quoting/injection surprises.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

from ..safety import check_command

__all__ = ["ShellRunner", "CommandResult"]

CommandResult = dict  # {"cmd", "shell", "exit_code", "stdout", "stderr", "duration_seconds", "blocked", "block_reason", "timed_out"}

ShellType = Literal["powershell", "cmd", "bash", "auto"]


def _cap(text: str) -> str:
    return text or ""


class ShellRunner:
    """Executes guarded shell commands with timeout, cwd and split streams."""

    def __init__(self, project_root: Path, default_timeout: int = 120, logger=None) -> None:
        self.project_root = Path(project_root)
        self.default_timeout = default_timeout
        self._logger = logger  # optional SessionLogger

    # ── interpreter discovery ────────────────────────────────────────────────

    @staticmethod
    def _find_powershell() -> str | None:
        return shutil.which("pwsh") or shutil.which("powershell")

    @staticmethod
    def _find_cmd() -> str | None:
        return shutil.which("cmd") or (os.environ.get("ComSpec") if os.name == "nt" else None)

    @staticmethod
    def _find_bash() -> str | None:
        return shutil.which("bash") or shutil.which("sh")

    def _resolve_shell(self, shell_type: ShellType) -> tuple[str, list[str]] | None:
        """Return (shell_name, argv_prefix) or None if unavailable."""
        if shell_type in ("powershell", "auto"):
            ps = self._find_powershell()
            if ps:
                return ("powershell", [ps, "-NoProfile", "-NonInteractive",
                                       "-ExecutionPolicy", "Bypass", "-Command"])
            if shell_type == "powershell":
                return None
        if shell_type in ("cmd", "auto") and os.name == "nt":
            cmd = self._find_cmd()
            if cmd:
                return ("cmd", [cmd, "/d", "/s", "/c"])
            if shell_type == "cmd":
                return None
        if shell_type == "cmd":
            return None  # CMD only exists on Windows
        bash = self._find_bash()
        if bash:
            return ("bash", [bash, "-c"])
        return None

    # ── execution ────────────────────────────────────────────────────────────

    def run(self, command: str, shell_type: ShellType = "auto",
            timeout_seconds: int | None = None, cwd: Path | None = None) -> CommandResult:
        """Run *command*; never raises. Blocked/dangerous commands return a
        result with ``blocked=True`` and exit_code 126 without executing."""
        timeout = timeout_seconds or self.default_timeout
        workdir = Path(cwd) if cwd else self.project_root
        base: CommandResult = {
            "cmd": command, "shell": shell_type, "exit_code": 126,
            "stdout": "", "stderr": "", "duration_seconds": 0.0,
            "blocked": False, "block_reason": "", "timed_out": False,
            "cwd": str(workdir), "timeout_seconds": timeout,
        }

        verdict = check_command(command)
        if not verdict:
            base.update(blocked=True, block_reason=verdict.reason,
                        stderr=f"SafetyGuard: {verdict.reason}")
            self._log(base)
            return base

        resolved = self._resolve_shell(shell_type)
        if resolved is None:
            base.update(exit_code=127,
                        stderr=f"no interpreter available for shell_type={shell_type!r}")
            self._log(base)
            return base
        shell_name, argv_prefix = resolved
        base["shell"] = shell_name

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [*argv_prefix, command],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=timeout, cwd=str(workdir),
            )
            base.update(
                exit_code=int(proc.returncode),
                stdout=_cap(proc.stdout), stderr=_cap(proc.stderr),
            )
        except subprocess.TimeoutExpired as exc:
            base.update(
                exit_code=124, timed_out=True,
                stdout=_cap(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
                stderr=f"timeout after {timeout}s",
            )
        except OSError as exc:
            base.update(exit_code=127, stderr=str(exc))
        base["duration_seconds"] = round(time.monotonic() - started, 3)
        self._log(base)
        return base

    # ── convenience wrappers (mirror the tool-spec names) ────────────────────

    def run_powershell(self, command: str, timeout_seconds: int | None = None) -> CommandResult:
        return self.run(command, "powershell", timeout_seconds)

    def run_cmd(self, command: str, timeout_seconds: int | None = None) -> CommandResult:
        return self.run(command, "cmd", timeout_seconds)

    def run_command(self, command: str, shell_type: ShellType = "auto",
                    timeout_seconds: int | None = None) -> CommandResult:
        return self.run(command, shell_type, timeout_seconds)

    # ── logging ──────────────────────────────────────────────────────────────

    def _log(self, result: CommandResult) -> None:
        if self._logger is None:
            return
        try:
            self._logger.log_build_test(
                cmd=result["cmd"], exit_code=result["exit_code"],
                output=result["stdout"] + ("\n" + result["stderr"] if result["stderr"] else ""),
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                shell=result.get("shell"),
                cwd=result.get("cwd"),
                duration_seconds=result.get("duration_seconds"),
                timeout_seconds=result.get("timeout_seconds"),
                timed_out=result.get("timed_out"),
                blocked=result.get("blocked"),
                block_reason=result.get("block_reason"),
            )
        except Exception:
            pass
