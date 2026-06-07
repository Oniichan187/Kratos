"""LLM role-runner mixin — prompt assembly, auto-compression, and the
planner/coder/verifier model-call generators.

Split out of ``KratosAgent`` so the orchestration class (``core/agent.py``)
stays focused on the pipeline shape; these methods are mixed back in via
``class KratosAgent(_RoleRunnerMixin, ...)`` and operate on the same
``self`` (config, bridge, histories, compressor, cancel event).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Generator

from ..llm.tokens import estimate, estimate_messages, fit_to_budget
from ..prompts import get_system, load_prompts
from ..roles import _needs_thinking
from ..router import Route
from ..context import ScopeType


class _RoleRunnerMixin:
    """Provides ``_run_planner``/``_run_coder``/``_run_verifier`` and the
    shared prompt-preparation/auto-compression helpers they rely on."""

    # ── model runners ─────────────────────────────────────────────────────────

    def _role_num_ctx(
        self,
        role: str,
        model: str,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> int:
        from ..llm.tokens import effective_num_ctx
        configured = {
            "planner": self.config.planner_num_ctx,
            "coder": self.config.coder_num_ctx,
            "verifier": self.config.verifier_num_ctx,
        }[role]
        return effective_num_ctx(
            model=model,
            configured_num_ctx=configured,
            vram_ctx_ceiling=self.config.vram_ctx_ceiling,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            force_max_context=getattr(self.config, "always_max_ctx", True),
        )

    @staticmethod
    def _messages(system: str, history: list[dict], msg: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": msg},
        ]

    def _prepare_model_prompt(
        self,
        role: str,
        model: str,
        system: str,
        history: list[dict],
        msg: str,
        num_predict: int,
    ) -> tuple[list[dict], int, int, str, list[tuple[str, str, str]]]:
        events: list[tuple[str, str, str]] = []
        prompt_msgs = self._messages(system, history, msg)
        prompt_tok = estimate_messages(prompt_msgs)
        num_ctx = self._role_num_ctx(role, model, prompt_tok, num_predict)

        compressed = self._auto_compress_if_needed(
            history, model, num_ctx,
            prompt_tokens=prompt_tok,
            role=role,
            _pending_events=events,
        )
        if compressed:
            prompt_msgs = self._messages(system, history, msg)
            prompt_tok = estimate_messages(prompt_msgs)

        if prompt_tok > num_ctx and history:
            removed_pairs = 0
            while history and prompt_tok > num_ctx:
                remove_n = 2 if len(history) >= 2 else 1
                del history[:remove_n]
                removed_pairs += 1
                prompt_msgs = self._messages(system, history, msg)
                prompt_tok = estimate_messages(prompt_msgs)
            events.append((
                "compress",
                f"Trimmed {role} history by {removed_pairs} old pair(s) "
                f"to fit real {num_ctx:,}-token window",
                "info",
            ))

        if prompt_tok > num_ctx:
            reserved = estimate_messages(self._messages(system, history, ""))
            msg_budget = max(256, num_ctx - reserved - 64)
            fitted_msg = fit_to_budget(msg, msg_budget)
            if fitted_msg != msg:
                msg = fitted_msg
                prompt_msgs = self._messages(system, history, msg)
                prompt_tok = estimate_messages(prompt_msgs)
                events.append((
                    "warn",
                    f"{role} prompt exceeded real {num_ctx:,}-token window; "
                    "trimmed current input before model call.",
                    "warn",
                ))

        if prompt_tok > num_ctx:
            history.clear()
            msg_budget = max(256, num_ctx - estimate(system) - 128)
            msg = fit_to_budget(msg, msg_budget)
            prompt_msgs = self._messages(system, history, msg)
            prompt_tok = estimate_messages(prompt_msgs)
            events.append((
                "warn",
                f"{role} prompt still exceeded context after compression; "
                "dropped role history for this call.",
                "warn",
            ))

        return prompt_msgs, prompt_tok, num_ctx, msg, events

    def _auto_compress_if_needed(
        self, history: list[dict], model: str, num_ctx: int,
        prompt_tokens: int | None = None,
        role: str = "",
        _pending_events: list | None = None,
    ) -> bool:
        """Compress history in-place if it's approaching the context limit.

        If _pending_events is provided, appends a (source, content, kind) tuple
        so the caller can yield it — making compression visible in the UI.
        """
        if not self.config.auto_compress:
            return False
        history_tok = estimate_messages(history)
        tok = prompt_tokens if prompt_tokens is not None else history_tok
        threshold = int(num_ctx * self.config.compress_threshold)
        if tok > threshold or history_tok > threshold or len(history) > self.config.max_history_pairs * 2:
            before = [dict(item) for item in history]
            compressed = self._compressor.compress_history(history, keep_pairs=4)
            if compressed and _pending_events is not None:
                short_model = model.split("/")[-1].split(":")[0]
                _pending_events.append(
                    ("compress",
                     f"Auto-compressed {short_model} history  {tok:,} tokens → keep 4 pairs",
                     "info")
                )
            if compressed:
                self._persist_compression_artifact(
                    role=role or "unknown",
                    model=model,
                    history_before=before,
                    history_after=list(history),
                    prompt_tokens=tok,
                    num_ctx=num_ctx,
                    threshold=threshold,
                )
            return compressed
        return False

    def _persist_compression_artifact(
        self,
        *,
        role: str,
        model: str,
        history_before: list[dict],
        history_after: list[dict],
        prompt_tokens: int,
        num_ctx: int,
        threshold: int,
    ) -> Path | None:
        root = getattr(self._indexer, "root", None)
        if root is None:
            return None

        artifact_dir = Path(root) / ".kratos" / "knowledge" / "compressions"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        short_model = model.split("/")[-1].split(":")[0]
        safe_role = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (role or "role"))
        path = artifact_dir / f"{stamp}_{safe_role}_{short_model}.md"

        removed = history_before[: max(0, len(history_before) - len(history_after))]
        removed_notes: list[str] = []
        for item in removed[-8:]:
            role_name = str(item.get("role", "unknown"))
            content = " ".join(str(item.get("content", "")).split())
            if not content:
                continue
            removed_notes.append(f"- `{role_name}`: {content[:220]}")

        compressed_note = ""
        if history_after:
            compressed_note = str(history_after[0].get("content", "")).strip()

        artifact = [
            "# Auto-Compression Artifact",
            "",
            f"- Time: {stamp}",
            f"- Role: {role}",
            f"- Model: {model}",
            f"- Prompt tokens: {prompt_tokens}",
            f"- Context window: {num_ctx}",
            f"- Compression threshold: {threshold}",
            f"- History messages before: {len(history_before)}",
            f"- History messages after: {len(history_after)}",
            "",
            "## Compressed Summary",
            compressed_note or "_No compressed summary available._",
            "",
            "## Removed Context Signals",
        ]
        artifact.extend(removed_notes or ["- _No removed context captured._"])
        path.write_text("\n".join(artifact).rstrip() + "\n", encoding="utf-8")

        knowledge = getattr(self, "_knowledge", None)
        if knowledge is not None and hasattr(knowledge, "ingest_markdown_artifact"):
            try:
                knowledge.ingest_markdown_artifact(
                    path,
                    metadata={
                        "kind": "compression_artifact",
                        "role": role,
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "num_ctx": num_ctx,
                        "threshold": threshold,
                    },
                )
            except Exception:
                pass
        return path

    def _run_planner(
        self, msg: str, route: Route, keep_alive: str = "0",
        scope: ScopeType = "targeted", task: str = "", is_retry: bool = False,
    ) -> Generator:
        needs_thinking = _needs_thinking(task, scope, route, is_retry, 0)
        planner_think: bool | None = None if needs_thinking else False
        p = load_prompts()

        # Larger budget on retry+CoT so thinking doesn't crowd out the actual plan
        if needs_thinking and is_retry:
            num_predict = p.get_predict("plan_retry")
        elif needs_thinking:
            num_predict = p.get_predict("plan_heavy")
        else:
            num_predict = p.get_predict("plan")

        prompt_msgs, prompt_tok, num_ctx, stored_msg, _compress_events = self._prepare_model_prompt(
            "planner",
            self.config.planner_model,
            get_system("planner"),
            self._planner_history,
            msg,
            num_predict,
        )
        for _ev in _compress_events:
            yield _ev
        yield ("ctx_info", f"planner|{prompt_tok}|{num_ctx}", "info")

        yield ("log", json.dumps({
            "type": "model_input", "role": "planner",
            "model": self.config.planner_model,
            "num_ctx": num_ctx, "prompt_tokens_est": prompt_tok,
            "think": needs_thinking,
        }), "log")

        thinking = ""
        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.planner_model,
                messages=prompt_msgs,
                temperature=self.config.planner_temp,
                num_predict=num_predict,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                think=planner_think,
                cancel_event=self._cancel_event,
            ):
                if kind == "think":
                    thinking += token
                    yield ("planner", token, "think")   # stream so user sees progress
                elif kind == "usage":
                    self._record_usage(token)
                    yield ("usage", token, "usage")
                else:
                    full += token
                    yield ("planner", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Planner interrupted.", "warn")
            return ""
        except Exception as exc:
            yield ("error", f"Planner failed: {exc}", "error")

        # Guard: if CoT exhausted the token budget before producing any output,
        # re-run immediately with think=False using the same prompt.
        if not full.strip() and needs_thinking:
            yield ("warn", "CoT used all token budget — retrying without chain-of-thought", "warn")
            try:
                for token, kind in self.bridge.chat(
                    model=self.config.planner_model,
                    messages=prompt_msgs,
                    temperature=self.config.planner_temp,
                    num_predict=p.get_predict("plan"),
                    num_ctx=num_ctx,
                    keep_alive=keep_alive,
                    think=False,
                    cancel_event=self._cancel_event,
                ):
                    if kind == "usage":
                        self._record_usage(token)
                        yield ("usage", token, "usage")
                    elif kind != "think":
                        full += token
                        yield ("planner", token, kind)
            except Exception as exc:
                yield ("error", f"Planner no-think retry failed: {exc}", "error")

        if thinking:
            yield ("log", json.dumps({"type": "model_thinking", "role": "planner",
                                       "chars": len(thinking)}), "log")
        yield ("log", json.dumps({"type": "model_output", "role": "planner",
                                   "chars": len(full)}), "log")

        if not self._is_cancelled():
            self._planner_history.append({"role": "user",      "content": stored_msg})
            self._planner_history.append({"role": "assistant", "content": full})
        return full

    def _run_coder(self, msg: str, model_override: str | None = None) -> Generator:
        coder_model = model_override or self.config.coder_model
        p = load_prompts()
        prompt_msgs, prompt_tok, num_ctx, stored_msg, _compress_events = self._prepare_model_prompt(
            "coder",
            coder_model,
            get_system("coder"),
            self._coder_history,
            msg,
            p.get_predict("code"),
        )
        for _ev in _compress_events:
            yield _ev
        yield ("ctx_info", f"coder|{prompt_tok}|{num_ctx}", "info")

        yield ("log", json.dumps({
            "type": "model_input", "role": "coder",
            "model": coder_model,
            "num_ctx": num_ctx, "prompt_tokens_est": prompt_tok,
        }), "log")

        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=coder_model,
                messages=prompt_msgs,
                temperature=self.config.coder_temp,
                num_predict=p.get_predict("code"),
                num_ctx=num_ctx,
                think=False,
                cancel_event=self._cancel_event,
            ):
                if kind == "usage":
                    self._record_usage(token)
                    yield ("usage", token, "usage")
                elif kind != "think":
                    full += token
                    yield ("coder", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Coder interrupted.", "warn")
            return ""
        except Exception as exc:
            yield ("error", f"Coder failed: {exc}", "error")

        yield ("log", json.dumps({"type": "model_output", "role": "coder",
                                   "chars": len(full)}), "log")

        if not self._is_cancelled():
            self._coder_history.append({"role": "user",      "content": stored_msg})
            self._coder_history.append({"role": "assistant", "content": full})
        return full

    def _run_verifier(self, msg: str) -> Generator:
        p = load_prompts()
        prompt_msgs, prompt_tok, num_ctx, _stored_msg, _compress_events = self._prepare_model_prompt(
            "verifier",
            self.config.verifier_model,
            get_system("verifier"),
            [],
            msg,
            p.get_predict("verify"),
        )
        for _ev in _compress_events:
            yield _ev

        yield ("ctx_info", f"verifier|{prompt_tok}|{num_ctx}", "info")
        yield ("log", json.dumps({
            "type": "model_input", "role": "verifier",
            "model": self.config.verifier_model, "num_ctx": num_ctx,
            "prompt_tokens_est": prompt_tok,
        }), "log")

        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.verifier_model,
                messages=prompt_msgs,
                temperature=self.config.verifier_temp,
                num_predict=p.get_predict("verify"),
                num_ctx=num_ctx,
                keep_alive="0",
                think=False,
                cancel_event=self._cancel_event,
            ):
                if kind == "usage":
                    self._record_usage(token)
                elif kind != "think":
                    full += token
                    yield ("verify", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Verifier interrupted.", "warn")
            return ""
        except Exception as exc:
            yield ("error", f"Verifier failed: {exc}", "error")
            full = "NEEDS_REVISION: verifier error — treat as unverified"

        yield ("log", json.dumps({"type": "model_output", "role": "verifier",
                                   "chars": len(full)}), "log")
        return full
