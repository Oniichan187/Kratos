"""Build/test command runner mixin — discovers and executes verification
commands (configured ``/build``/``/test``, README-extracted, auto-inferred)
and exposes the single-shot ``_run_build_test`` helper used by the
diagnostic-loop route.

Split out of ``KratosAgent`` (mixed in via
``class KratosAgent(_RoleRunnerMixin, _RetryMixin, _VerificationRunnerMixin)``);
these methods operate on the same ``self`` (config, indexer).
"""

from __future__ import annotations

import shutil
import subprocess
import time

from ..router import Route
from ..verification import (
    VerificationCommand,
    _dedupe_verification_commands,
    _is_safe_verification_command,
    _is_test_verification_command,
    _extract_readme_verification_commands,
    _infer_project_verification_commands,
)


class _VerificationRunnerMixin:
    """Provides ``_verification_commands``, ``_run_verification_command``,
    ``_run_readonly_command``, and ``_run_build_test``."""

    # ── build/test runner ─────────────────────────────────────────────────────

    def _verification_commands(self, route: Route) -> list[VerificationCommand]:
        commands: list[VerificationCommand] = []

        if self.config.build_cmd:
            commands.append(VerificationCommand(
                cmd=self.config.build_cmd,
                purpose="configured build verification",
                source="/build",
                is_test=_is_test_verification_command(self.config.build_cmd),
            ))
        if self.config.test_cmd:
            commands.append(VerificationCommand(
                cmd=self.config.test_cmd,
                purpose="configured test verification",
                source="/test",
                is_test=True,
            ))

        if commands:
            return _dedupe_verification_commands(commands)

        if not self.config.auto_discover_verification:
            return []

        root = self._indexer.root
        commands.extend(_extract_readme_verification_commands(root))
        commands.extend(_infer_project_verification_commands(root))

        safe = [item for item in commands if _is_safe_verification_command(item.cmd)]
        return _dedupe_verification_commands(safe)

    def _run_verification_command(self, command: VerificationCommand) -> dict:
        started = time.monotonic()
        try:
            result = subprocess.run(
                command.cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.verification_timeout_seconds,
                cwd=str(self._indexer.root),
            )
            output = (result.stdout or "") + (result.stderr or "")
            exit_code = int(result.returncode)
        except Exception as exc:
            output = str(exc)
            exit_code = 1

        return {
            "cmd": command.cmd,
            "purpose": command.purpose,
            "source": command.source,
            "is_test": command.is_test,
            "exit_code": exit_code,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output": output,
        }

    def _run_readonly_command(self, cmd: str, root=None) -> dict:
        started = time.monotonic()
        shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        try:
            result = subprocess.run(
                [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", cmd],
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.verification_timeout_seconds,
                cwd=str(root or self._indexer.root),
            )
            output = (result.stdout or "") + (result.stderr or "")
            exit_code = int(result.returncode)
        except Exception as exc:
            output = str(exc)
            exit_code = 1

        return {
            "cmd": cmd,
            "purpose": "readonly inspection",
            "source": "coder-inspect",
            "is_test": False,
            "exit_code": exit_code,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output": output,
        }

    def _run_build_test(self) -> tuple[bool, str, str] | None:
        commands = self._verification_commands(Route.DIAGNOSTIC_LOOP)
        if not commands:
            return None
        result = self._run_verification_command(commands[0])
        return (result["exit_code"] == 0, result["output"], result["cmd"])
