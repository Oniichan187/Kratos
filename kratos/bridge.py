"""Ollama bridge — native Windows + WSL fallback.

chat() yields (token, kind) where kind is "think" | "text".
After the stream it emits one final ("usage", json_str, "usage") tuple
containing prompt_eval_count + eval_count from Ollama's done-message,
so callers can track real token consumption.
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
    """Convert C:\\... to /mnt/c/... for WSL-side Ollama."""
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
        """Start Ollama — prefers native Windows, falls back to WSL."""
        if self.is_running():
            return True

        if _is_native_windows_ollama():
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
                capture_output=True, encoding="utf-8", errors="replace", timeout=120,
            )
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    if line.strip():
                        yield line.strip()
            if proc.returncode != 0:
                yield f"ERROR: {(proc.stderr or proc.stdout or 'unknown').strip()[-200:]}"
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
            stream=True, timeout=300,
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
        r = requests.post(
            f"{self.host}/api/pull",
            json={"name": name, "stream": True},
            stream=True, timeout=None,
        )
        last = ""
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    completed = data.get("completed", 0)
                    total = data.get("total", 0)
                    msg = f"{status} {int(completed / total * 100)}%" if total else status
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
        """Stream chat response.

        Yields (token, kind) where kind is "think" | "text".
        After the last token, yields one final ("", "usage") tuple whose
        token field is a JSON string:
            {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}
        Callers that don't care about usage can ignore tuples where kind=="usage".
        """
        options: dict = {"temperature": temperature, "num_predict": num_predict}
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep_alive,
            "options": options,
        }
        if think is not None:
            payload["think"] = think

        r = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            stream=True,
            timeout=None,
        )

        prompt_tokens = 0
        completion_tokens = 0

        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in data:
                raise RuntimeError(f"Ollama: {data['error']}")

            if data.get("done", False):
                # Harvest real token counts from the done-message
                prompt_tokens     = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                break

            msg   = data.get("message", {})
            think_tok = msg.get("thinking", "")
            text_tok  = msg.get("content", "")
            if think_tok:
                yield (think_tok, "think")
            if text_tok:
                yield (text_tok, "text")

        # Always emit a usage tuple so the agent can track consumption
        usage = json.dumps({
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        })
        yield (usage, "usage")
