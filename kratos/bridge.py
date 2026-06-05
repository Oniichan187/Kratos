"""Ollama bridge — supports native Windows Ollama and WSL Ollama.

Detects automatically which variant is running. On this machine Ollama runs
as a native Windows process (ollama.exe / ollama app.exe), so Windows paths
are used directly for model creation and the start() method uses the Windows
Ollama CLI. WSL-based startup is kept as a fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Generator

import requests

_PROBE_TIMEOUT = 3
_START_TIMEOUT = 45


# ── Ollama runtime detection ──────────────────────────────────────────────────

def _is_native_windows_ollama() -> bool:
    """Return True if ollama.exe is available on Windows PATH."""
    return sys.platform == "win32" and shutil.which("ollama") is not None


def _wsl_exe() -> str | None:
    return shutil.which("wsl.exe") or shutil.which("wsl")


def _run_wsl(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    wsl = _wsl_exe()
    if not wsl:
        raise RuntimeError("wsl.exe not found on PATH")
    return subprocess.run(
        [wsl, *args],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout,
    )


def win_to_wsl(win_path: str | Path) -> str:
    """Convert C:\\... to /mnt/c/... for WSL-side Ollama (fallback only)."""
    p = Path(win_path)
    try:
        p = p.resolve()
    except OSError:
        p = p.absolute()
    drive = p.drive
    if len(drive) == 2 and drive[1] == ":":
        letter = drive[0].lower()
        rest = p.as_posix()[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{letter}{rest}"
    forward = str(win_path).replace("\\", "/")
    proc = _run_wsl(["wslpath", "-a", forward], timeout=15)
    if proc.returncode != 0:
        raise RuntimeError(f"wslpath failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


# ── Bridge ────────────────────────────────────────────────────────────────────

class OllamaBridge:
    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=_PROBE_TIMEOUT)
            return r.status_code == 200
        except Exception:
            return False

    def start(self) -> bool:
        """Start Ollama. Prefers native Windows ollama.exe, falls back to WSL."""
        if self.is_running():
            return True

        if _is_native_windows_ollama():
            # Native Windows — start as background process
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
                )
            except Exception:
                pass
        else:
            # WSL fallback
            try:
                _run_wsl(
                    ["-e", "bash", "-c",
                     "nohup env OLLAMA_HOST=0.0.0.0 CUDA_VISIBLE_DEVICES=0 "
                     "ollama serve > /tmp/kratos_ollama.log 2>&1 &"],
                    timeout=10,
                )
            except Exception:
                pass

        deadline = time.monotonic() + _START_TIMEOUT
        while time.monotonic() < deadline:
            if self.is_running():
                return True
            time.sleep(1.5)
        return False

    # ── model management ─────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def model_exists(self, name: str) -> bool:
        try:
            models = self.list_models()
            base = name.split(":")[0]
            return any(m == name or m.split(":")[0] == base for m in models)
        except Exception:
            return False

    def create_from_gguf(
        self,
        model_name: str,
        gguf_win_path: str | Path,
        system_prompt: str = "",
        gpu_layers: int = 50,
        ctx: int = 4096,
        temp: float = 0.7,
    ) -> Generator[str, None, None]:
        """Create an Ollama model from a local GGUF file.

        Uses the Windows-native ``ollama create`` CLI when available
        (the HTTP /api/create endpoint has trouble parsing Windows paths).
        Falls back to the HTTP API with the WSL path for WSL-side Ollama.
        """
        if _is_native_windows_ollama():
            yield from self._create_from_gguf_cli(
                model_name, gguf_win_path, system_prompt, gpu_layers, ctx, temp
            )
        else:
            yield from self._create_from_gguf_api(
                model_name, win_to_wsl(gguf_win_path), system_prompt, gpu_layers, ctx, temp
            )

    def _create_from_gguf_cli(
        self,
        model_name: str,
        gguf_win_path: str | Path,
        system_prompt: str,
        gpu_layers: int,
        ctx: int,
        temp: float,
    ) -> Generator[str, None, None]:
        """Create via ``ollama create`` subprocess — handles Windows paths correctly."""
        lines = [f"FROM {gguf_win_path}"]
        if system_prompt:
            escaped = system_prompt.replace('"""', '\\"\\"\\"')
            lines.append(f'\nSYSTEM """{escaped}"""')
        lines += [
            f"\nPARAMETER num_gpu {gpu_layers}",
            f"PARAMETER num_ctx {ctx}",
            f"PARAMETER temperature {temp}",
            "PARAMETER top_p 0.9",
            "PARAMETER repeat_penalty 1.1",
        ]
        modelfile_content = "\n".join(lines)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".modelfile", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(modelfile_content)
            tmp.close()
            proc = subprocess.run(
                ["ollama", "create", model_name, "-f", tmp.name],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    if line.strip():
                        yield line.strip()
            if proc.returncode != 0:
                err = proc.stderr or proc.stdout or "unknown error"
                yield f"ERROR: {err.strip()[-200:]}"
            else:
                yield "success"
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _create_from_gguf_api(
        self,
        model_name: str,
        gguf_path: str,
        system_prompt: str,
        gpu_layers: int,
        ctx: int,
        temp: float,
    ) -> Generator[str, None, None]:
        """Create via HTTP API (WSL Ollama with /mnt/c/... paths)."""
        lines = [f"FROM {gguf_path}"]
        if system_prompt:
            escaped = system_prompt.replace('"""', '\\"\\"\\"')
            lines.append(f'\nSYSTEM """{escaped}"""')
        lines += [
            f"\nPARAMETER num_gpu {gpu_layers}",
            f"PARAMETER num_ctx {ctx}",
            f"PARAMETER temperature {temp}",
        ]
        r = requests.post(
            f"{self.host}/api/create",
            json={"name": model_name, "modelfile": "\n".join(lines), "stream": True},
            stream=True,
            timeout=300,
        )
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if status:
                        yield status
                except json.JSONDecodeError:
                    pass

    def pull_model(self, name: str) -> Generator[str, None, None]:
        """Pull a model from Ollama hub."""
        r = requests.post(
            f"{self.host}/api/pull",
            json={"name": name, "stream": True},
            stream=True,
            timeout=None,
        )
        last = ""
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    completed = data.get("completed", 0)
                    total = data.get("total", 0)
                    msg = f"{status} {int(completed/total*100)}%" if total else status
                    if msg != last and msg:
                        yield msg
                        last = msg
                except json.JSONDecodeError:
                    pass

    # ── inference ─────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        num_predict: int = 4096,
        num_ctx: int | None = None,
        think: bool | None = None,
        keep_alive: str = "0",
    ) -> Generator[tuple[str, str], None, None]:
        """Stream chat response tokens via /api/chat.

        Yields ``(token, kind)`` tuples where ``kind`` is:
          - ``"think"``  internal reasoning tokens (Qwen3 / thinking models)
          - ``"text"``   final response tokens
        """
        options: dict = {"temperature": temperature, "num_predict": num_predict}
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep_alive,   # "0" = unload immediately after response
            "options": options,
        }
        if think is not None:
            payload["think"] = think   # top-level field, not in options
        r = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            stream=True,
            timeout=None,
        )
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    # Surface Ollama errors immediately instead of silently skipping
                    if "error" in data:
                        raise RuntimeError(f"Ollama: {data['error']}")
                    if not data.get("done", False):
                        msg = data.get("message", {})
                        think = msg.get("thinking", "")
                        text = msg.get("content", "")
                        if think:
                            yield (think, "think")
                        if text:
                            yield (text, "text")
                except json.JSONDecodeError:
                    pass
