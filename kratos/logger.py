"""Session logger — JSONL log in .kratos/ when /logging on is active.

When enabled, logs EVERYTHING without truncation:
  - Full user input
  - Context builder: files scanned, files loaded with full content, full context package
  - Planner: full system prompt, full message sent, full thinking tokens, full response
  - Coder: full system prompt, full message sent, full thinking tokens, full response
  - Verifier: full message sent, full response, decision
  - File operations: op, path, full content written (≤ 100 KB)
  - Build/test: full command + full output
  - Routing decisions, tool calls, errors

Log format: one JSON object per line (JSONL).
Each entry has: {"ts": "ISO-8601", "type": "event_type", ...fields}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class SessionLogger:
    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._path: Path | None = None
        self._fh = None
        self._enabled = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def enable(self) -> Path:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = self._log_dir / f"session_{ts}.jsonl"
        self._fh = open(self._path, "a", encoding="utf-8")
        self._enabled = True
        self._write("session_start", log_path=str(self._path))
        return self._path

    def disable(self) -> None:
        if self._enabled:
            self._write("session_end")
        self._enabled = False
        if self._fh:
            self._fh.close()
            self._fh = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def log_path(self) -> Path | None:
        return self._path

    # ── core writer — no truncation ───────────────────────────────────────────

    def _write(self, type_: str, **fields) -> None:
        if not self._enabled or not self._fh:
            return
        entry = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": type_,
            **fields,
        }
        try:
            self._fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except Exception:
            pass

    # ── pipeline stage log methods ────────────────────────────────────────────

    def log_input(self, text: str) -> None:
        self._write("user_input", text=text, char_count=len(text), word_count=len(text.split()))

    def log_route(self, intent: str, route: str) -> None:
        self._write("route_decision", intent=intent, route=route)

    def log_index(self, project_name: str, files: list[dict]) -> None:
        """files: list of {rel_path, size, priority}"""
        self._write("index_project",
                    project=project_name,
                    file_count=len(files),
                    files=files)

    def log_context_package(self, intent: str, route: str, scope: str,
                             memory_summary: str, files: list[dict],
                             full_context_prompt: str) -> None:
        """Log the full context package that will be given to the planner."""
        self._write("context_package",
                    intent=intent, route=route, scope=scope,
                    memory_summary=memory_summary,
                    files_loaded=files,
                    full_context_prompt=full_context_prompt,
                    token_estimate=len(full_context_prompt.split()))

    def log_model_input(self, role: str, model: str,
                        system_prompt: str, message: str) -> None:
        """Full prompt sent to a model (system + user message)."""
        self._write("model_input",
                    role=role, model=model,
                    system_prompt=system_prompt,
                    message=message,
                    message_tokens_estimate=len(message.split()))

    def log_model_thinking(self, role: str, thinking_text: str) -> None:
        """Full chain-of-thought / thinking tokens from a model."""
        if thinking_text.strip():
            self._write("model_thinking", role=role,
                        text=thinking_text,
                        char_count=len(thinking_text))

    def log_model_output(self, role: str, model: str, text: str) -> None:
        """Full model response text."""
        self._write("model_output", role=role, model=model,
                    text=text, char_count=len(text))

    def log_verify_decision(self, decision: str, feedback: str, iteration: int) -> None:
        self._write("verify_decision",
                    decision=decision,
                    feedback=feedback,
                    iteration=iteration)

    def log_tool(self, name: str, args: dict, result: str = "") -> None:
        self._write("tool_call", name=name, args=args, result=result)

    def log_file_write(self, rel_path: str, content: str) -> None:
        self._write("file_write",
                    path=rel_path,
                    size_bytes=len(content.encode()),
                    content=content[:102400])   # max 100 KB in log

    def log_file_delete(self, rel_path: str) -> None:
        self._write("file_delete", path=rel_path)

    def log_build_test(self, cmd: str, exit_code: int, output: str) -> None:
        self._write("build_test",
                    command=cmd,
                    exit_code=exit_code,
                    output=output)

    def log_info(self, msg: str) -> None:
        self._write("info", msg=msg)

    def log_warn(self, msg: str) -> None:
        self._write("warn", msg=msg)

    def log_error(self, msg: str) -> None:
        self._write("error", msg=msg)
