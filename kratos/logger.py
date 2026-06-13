"""Session logger - JSONL diagnostics in .kratos/ when /logging on is active.

When enabled, this logger is intentionally exhaustive and untruncated. UI and
prompt paths may still shorten data for display/model stability, but anything
sent here should preserve the full underlying value for debugging.

Log format: one JSON object per line. Every entry has at least:
{"ts": "ISO-8601", "seq": int, "type": "event_type", ...fields}
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionLogger:
    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._path: Path | None = None
        self._fh = None
        self._enabled = False
        self._lock = threading.RLock()
        self._seq = 0

    # -- lifecycle ---------------------------------------------------------

    def enable(self) -> Path:
        with self._lock:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._path = self._log_dir / f"session_{ts}.jsonl"
            self._fh = open(self._path, "a", encoding="utf-8")
            self._enabled = True
            self._write(
                "session_start",
                log_path=str(self._path),
                cwd=str(Path.cwd()),
                pid=os.getpid(),
            )
            return self._path

    def disable(self) -> None:
        with self._lock:
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

    # -- core writer -------------------------------------------------------

    def _write(self, type_: str, **fields: Any) -> None:
        if not self._enabled or not self._fh:
            return
        with self._lock:
            if not self._enabled or not self._fh:
                return
            self._seq += 1
            entry = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "seq": self._seq,
                "type": type_,
                **fields,
            }
            try:
                self._fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                self._fh.flush()
            except Exception as exc:
                print(f"Kratos logging failed: {exc}", file=sys.stderr)

    def log_event(self, type_: str, **fields: Any) -> None:
        self._write(type_, **fields)

    def log_agent_event(self, source: str, content: str, kind: str) -> None:
        self._write(
            "agent_event",
            source=source,
            kind=kind,
            content=content,
            char_count=len(content or ""),
        )

    # -- pipeline stage log methods ---------------------------------------

    def log_input(self, text: str) -> None:
        self._write("user_input", text=text, char_count=len(text), word_count=len(text.split()))

    def log_route(self, intent: str, route: str) -> None:
        self._write("route_decision", intent=intent, route=route)

    def log_index(self, project_name: str, files: list[dict]) -> None:
        self._write("index_project", project=project_name, file_count=len(files), files=files)

    def log_context_package(
        self,
        intent: str,
        route: str,
        scope: str,
        memory_summary: str,
        files: list[dict],
        full_context_prompt: str,
    ) -> None:
        self._write(
            "context_package",
            intent=intent,
            route=route,
            scope=scope,
            memory_summary=memory_summary,
            files_loaded=files,
            full_context_prompt=full_context_prompt,
            token_estimate=len(full_context_prompt.split()),
        )

    def log_model_input(
        self,
        role: str,
        model: str,
        system_prompt: str,
        message: str,
        *,
        messages: list[dict] | None = None,
        **metadata: Any,
    ) -> None:
        self._write(
            "model_input",
            role=role,
            model=model,
            system_prompt=system_prompt,
            message=message,
            messages=messages,
            message_tokens_estimate=len(message.split()),
            **metadata,
        )

    def log_model_stream(self, role: str, model: str, token: str, kind: str) -> None:
        self._write(
            "model_stream",
            role=role,
            model=model,
            kind=kind,
            token=token,
            char_count=len(token or ""),
        )

    def log_model_thinking(self, role: str, thinking_text: str, **metadata: Any) -> None:
        if thinking_text.strip():
            self._write(
                "model_thinking",
                role=role,
                text=thinking_text,
                char_count=len(thinking_text),
                **metadata,
            )

    def log_model_output(self, role: str, model: str, text: str, **metadata: Any) -> None:
        self._write(
            "model_output",
            role=role,
            model=model,
            text=text,
            char_count=len(text),
            **metadata,
        )

    def log_verify_decision(self, decision: str, feedback: str, iteration: int) -> None:
        self._write("verify_decision", decision=decision, feedback=feedback, iteration=iteration)

    def log_tool(self, name: str, args: dict, result: str = "") -> None:
        self._write("tool_call", name=name, args=args, result=result)

    def log_file_write(
        self,
        rel_path: str,
        content: str,
        *,
        previous_content: str | None = None,
        sha256: str | None = None,
        ok: bool | None = None,
        **metadata: Any,
    ) -> None:
        self._write(
            "file_write",
            path=rel_path,
            size_bytes=len(content.encode("utf-8", "replace")),
            content=content,
            previous_content=previous_content,
            sha256=sha256,
            ok=ok,
            **metadata,
        )

    def log_file_delete(
        self,
        rel_path: str,
        *,
        previous_content: str | None = None,
        existed: bool | None = None,
        **metadata: Any,
    ) -> None:
        self._write(
            "file_delete",
            path=rel_path,
            previous_content=previous_content,
            existed=existed,
            **metadata,
        )

    def log_build_test(
        self,
        cmd: str,
        exit_code: int,
        output: str,
        *,
        stdout: str | None = None,
        stderr: str | None = None,
        **metadata: Any,
    ) -> None:
        self._write(
            "build_test",
            command=cmd,
            exit_code=exit_code,
            output=output,
            stdout=stdout,
            stderr=stderr,
            **metadata,
        )

    def log_info(self, msg: str) -> None:
        self._write("info", msg=msg)

    def log_warn(self, msg: str) -> None:
        self._write("warn", msg=msg)

    def log_error(self, msg: str) -> None:
        self._write("error", msg=msg)
