"""Kratos agent — dual-model pipeline with dynamic reasoning and auto-compression.

Token-stream convention:
  All generators yield (source: str, content: str, kind: str) where source is:
    "router"   routing decision info           kind="info"
    "header"   section start                   kind="planner"|"coder"|"verify"|"relay"
    "planner"  planner token stream            kind="think"|"text"
    "coder"    coder token stream              kind="think"|"text"
    "verify"   verifier token stream           kind="think"|"text"
    "relay"    relay pre-processor (no display) kind="text"
    "tool"     tool call display               kind="tool"
    "direct"   direct answer (no LLM)          kind="text"
    "info"     informational message            kind="info"
    "warn"     warning                          kind="warn"
    "error"    error                            kind="error"
    "question" clarification request            kind="question"
    "usage"    token usage JSON                 kind="usage"
    "end"      section end                      kind="end"

Pipeline:
  Input → Analyze → Classify → Route → Build Context
    → [Relay if huge] → Planner → Coder → Verifier
    → if NEEDS_REVISION: re-plan → re-code → re-verify  (up to max_verify_iterations)
    → VERIFIED | UNSOLVABLE

Auto-compression: before each model call, if estimated prompt tokens >
compress_threshold × num_ctx → compress_history() via the compressor model.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Generator

from ..analyzer import InputAnalyzer
from .buildtest import _VerificationRunnerMixin
from .retry import _RetryMixin
from .runners import _RoleRunnerMixin
from ..llm.bridge import OllamaBridge
from ..classifier import IntentClassifier, Intent
from ..compress import Compressor
from ..config import KratosConfig, _project_dir, GLOBAL_DIR
from ..context import ContextBuilder, ContextPackage, ProjectIndexer, ScopeType
from ..knowledge import ProjectKnowledge, RetrievedChunk
from ..memory import MemoryManager
from ..router import Route, Router
from ..llm.tokens import (
    estimate, estimate_messages,
    fit_to_budget, relay_needed,
)

# Prompts are fully externalized to JSON — all AI-facing strings, patterns, and pipeline config
# live in prompts_default.json (user-editable). Python code contains only flow logic.
from ..prompts import (
    load_prompts,
    get_system,
    get_snippet,
    get_predict,
    get_marker,
    get_toolchain,
    get_plan_config,
    reload_prompts,
)
from ..roles import (
    _scope_for,
    _coder_scope_for,
    _needs_thinking,
    _coder_context_block,
    _planner_msg,
    _coder_msg,
    run_coder_loop,
    _planner_retry_msg,
    _coder_retry_msg,
    _verify_msg,
    _clarification_msg,
    _direct_file_search,
    _direct_code_search,
)
from ..verification import (
    VerificationCommand,
    ProvenWork,
    CommandRegistry,
    _clean_command_line,
    _missing_command_paths,
    _is_safe_verification_command,
    _is_test_verification_command,
    _command_toolchain,
    _dedupe_verification_commands,
    _proven_work_satisfied,
    _format_proven_work_feedback,
    _extract_plan_steps,
    _extract_step_file_refs,
    _parse_step_tests,
    _patch_dotnet_test_runner,
)
from ..execution.parsing import (
    _FILE_CHANGE_RE,
    _FILE_DELETE_RE,
    _get_file_change_re,
    _get_file_delete_re,
    _parse_file_changes,
    _parse_file_deletions,
)
from ..execution.repair import try_repair_known_probe

# ── token predict limits (now sourced from prompts at runtime for full configurability) ──
# (kept as module fallbacks only for very early import paths; prefer get_predict)

_PLAN_PREDICT        = 2048
_PLAN_PREDICT_HEAVY  = 3072
_PLAN_PREDICT_RETRY  = 6144
_CODE_PREDICT        = 16384
_VERIFY_PREDICT      = 512
_RELAY_PREDICT       = 1200


# ── main agent ────────────────────────────────────────────────────────────────

class KratosAgent(_RoleRunnerMixin, _RetryMixin, _VerificationRunnerMixin):
    def __init__(self, config: KratosConfig, bridge: OllamaBridge, prompts=None) -> None:
        self.config = config
        self.bridge = bridge
        self.prompts = prompts or load_prompts()

        self._planner_history: list[dict] = []
        self._coder_history:   list[dict] = []

        self._analyzer   = InputAnalyzer()
        self._classifier = IntentClassifier()
        self._router     = Router()

        project_root      = Path.cwd()
        self._indexer     = ProjectIndexer(project_root)
        self._ctx_builder = ContextBuilder(self._indexer)
        self._memory      = MemoryManager(_project_dir(), GLOBAL_DIR)
        self._compressor  = Compressor(bridge, config)

        # The new vector knowledge base — "the project as a queryable vector DB".
        # Enables continuous dynamic "gets" (task-level + fresh per-step).
        # Works in WSL (embeds via the same bridge host). Graceful fallback if no lancedb/embed model.
        self._knowledge: ProjectKnowledge | None = None
        try:
            self._knowledge = ProjectKnowledge(config, bridge)
        except Exception:
            self._knowledge = None  # will stay in pure keyword/memory mode

        # Accumulated token usage for this session
        self._session_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0
        }

        # File operations extracted from last coder run — applied by caller
        self.pending_file_changes:   list[tuple[str, str]] = []
        self.pending_file_deletions: list[str] = []
        self._cancel_event = None

    # ── session usage ─────────────────────────────────────────────────────────

    @property
    def session_usage(self) -> dict[str, int]:
        return dict(self._session_usage)

    # ── knowledge base exposure (for /knowledge commands and status) ─────────
    @property
    def knowledge(self) -> "ProjectKnowledge | None":
        return self._knowledge

    def rebuild_knowledge(self, force: bool = False) -> int:
        if self._knowledge is None:
            return 0
        return self._knowledge.rebuild(force=force)

    def _record_usage(self, usage_json: str) -> None:
        try:
            d = json.loads(usage_json)
            self._session_usage["prompt_tokens"]     += d.get("prompt_tokens", 0)
            self._session_usage["completion_tokens"] += d.get("completion_tokens", 0)
        except Exception:
            pass

    def set_cancel_event(self, cancel_event) -> None:
        self._cancel_event = cancel_event

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_event is not None and self._cancel_event.is_set())

    # ── intelligent pipeline ─────────────────────────────────────────────────

    def process(self, task: str) -> Generator[tuple[str, str, str], None, None]:
        """Full pipeline. Yields (source, content, kind) events."""
        self.pending_file_changes.clear()
        self.pending_file_deletions.clear()

        # Unconditional PromptManager so all pm.get_snippet / revising / proven_work paths
        # (including retries, legacy coder, early errors, and branches that only assign inside
        # if use_stepwise or CODER_ONLY) are safe. load_prompts() is cached.
        pm = load_prompts()

        analysis = self._analyzer.analyze(task)
        intent   = self._classifier.classify(analysis)
        route    = self._router.route(intent)

        yield ("router", f"intent={intent.value}  route={route.value}", "info")

        # ── memory retrieval ──────────────────────────────────────────────────
        mem_entries    = self._memory.get_relevant(
            analysis.keywords, categories=["solution", "file_role", "decision", "convention"]
        )
        memory_summary = self._memory.format_for_prompt(mem_entries)

        # ── NEW: dynamic vector "get" (continuous retrieval) ──────────────────
        # This is the heart of the best-possible design: instead of a static index
        # we do fresh, targeted semantic+hybrid retrieval ("gets") at the moments
        # that matter. The 262k coder + stepwise love this.
        retrieved_chunks: list[RetrievedChunk] = []
        if self._knowledge is not None:
            try:
                yield ("info", "retrieving (task-level) from project vector knowledge base…", "info")
                retrieved_chunks = self._knowledge.retrieve(
                    analysis.normalized or task,
                    top_k=getattr(self.config, "retrieval_top_k", 16),
                )
                yield ("tool", f"knowledge_get(task) → {len(retrieved_chunks)} chunks (vector+hybrid)", "tool")
            except Exception as exc:
                yield ("warn", f"knowledge retrieval failed (falling back): {exc}", "warn")

        # ── context build ─────────────────────────────────────────────────────
        scope = _scope_for(route, intent)
        n_indexed = len(self._indexer.index)
        yield ("tool", f"index_project({self._indexer.root.name!r}) → {n_indexed} files", "tool")

        ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary=memory_summary, scope=scope,
            token_budget=self.config.planner_num_ctx,
        )
        ctx.retrieved_chunks = retrieved_chunks  # task-level dynamic "get" results

        coder_scope = _coder_scope_for(intent)
        coder_context_budget = self._role_num_ctx("coder", self.config.coder_model, 0, 0)
        coder_ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary="", scope=coder_scope,
            token_budget=coder_context_budget,
        )
        coder_ctx.retrieved_chunks = retrieved_chunks  # initial broad retrieval for legacy full-pass path
        for f in ctx.files:
            sz = f"{f.size / 1024:.1f} KB" if f.size > 0 else "n/a"
            yield ("tool", f"read_file({f.rel_path!r}) → {sz}", "tool")

        # ── log context ───────────────────────────────────────────────────────
        yield ("log", json.dumps({
            "type": "index_project",
            "project": self._indexer.root.name,
            "file_count": n_indexed,
            "files": [{"rel_path": e.rel_path, "size": e.size, "priority": e.priority}
                      for e in self._indexer.index],
        }), "log")

        # ── command registry (once per run) ──────────────────────────────────
        # Discovers toolchains, build commands, test runners from project structure.
        # All three consumers (planner hint, coder context, stepwise verify) share this.
        _cmd_registry = CommandRegistry(self.config, self._indexer.root).discover()
        _project_toolchains = _cmd_registry.toolchains
        verify_hint = _cmd_registry.verify_hint()

        # ── terminal routes ───────────────────────────────────────────────────
        if route == Route.DIRECT_ANSWER:
            if intent == Intent.FILE_SEARCH:
                result = _direct_file_search(self._indexer, analysis.keywords, analysis.file_paths)
            else:
                result = _direct_code_search(self._indexer, analysis.keywords)
            yield ("direct", result, "text")
            return

        if route == Route.ASK_CLARIFICATION:
            yield ("question", _clarification_msg(analysis, intent), "question")
            return

        if route == Route.PLANNER_ONLY:
            yield ("header", "planner", "header")
            plan_full = yield from self._run_planner(
                _planner_msg(task, ctx, all_files=self._indexer.index, verify_hint=verify_hint),
                route, keep_alive="5m", scope=scope, task=task, is_retry=False,
            )
            yield ("end", "planner", "end")
            if self._is_cancelled():
                yield ("warn", "Run cancelled.", "warn")
                return
            # Async memory extraction
            mem_entries_new = self._compressor.generate_memory(task, plan_full, "", [])
            self._memory.add_from_compress(mem_entries_new)
            return

        if route == Route.CODER_ONLY:
            # Git / shell / followup continuation — do not force the heavy "### FILE" format.
            # Still goes through coder (so it benefits from max ctx + history), but prompt is light.
            yield ("header", "coder", "header")
            pm = load_prompts()
            light = (
                f"{pm.get_snippet('task_label') or 'Task: '}{task}\n\n"
                + (pm.get_snippet("coder_only_light") or
                   "If this is a git/shell command or a small continuation, reply with the precise commands or "
                   "the minimal code patch. Use normal text or ``` blocks. Only emit ### FILE: blocks if the task "
                   "actually requires writing source files to disk.")
            )
            cfull = yield from self._run_coder(light)
            yield ("end", "coder", "end")
            if self._is_cancelled():
                yield ("warn", "Run cancelled before applying coder output.", "warn")
                return
            # If it happened to emit files (follow-up coding), still apply them (rare for pure git)
            chg = _parse_file_changes(cfull)
            if chg:
                self.pending_file_changes = chg
                self.pending_file_deletions = _parse_file_deletions(cfull)
                # (apply happens in caller via _show_file_ops + the agent already wrote? no — for this path we apply here for consistency)
                root = self._indexer.root
                for rp, ct in chg:
                    try:
                        (root / rp).parent.mkdir(parents=True, exist_ok=True)
                        (root / rp).write_text(ct, encoding="utf-8")
                        yield ("tool", f"apply_file({rp!r}) [from CODER_ONLY]", "tool")
                    except Exception:
                        pass
            return

        # ── plan → code → verify loop ─────────────────────────────────────────
        max_iter   = self.config.max_verify_iterations
        plan_text  = ""
        verify_feedback = ""
        needs_plan = route in (Route.PLANNER_THEN_CODER, Route.DIAGNOSTIC_LOOP)
        _accumulated_changes: dict[str, str] = {}
        _accumulated_deletions: set[str] = set()
        # Snapshot of files before any writes (for rollback on UNSOLVABLE)
        _original_snapshots: dict[str, str | None] = {}

        for attempt in range(max_iter):
            is_retry   = attempt > 0
            iter_label = f"iteration {attempt + 1}/{max_iter}"

            if is_retry:
                msg = pm.get_snippet("revising_iteration").format(current=attempt + 1, max=max_iter)
                yield ("info", msg, "info")
                self._indexer.build_index()
                ctx = self._ctx_builder.build(
                    analysis, intent=intent.value, route=route.value,
                    memory_summary=memory_summary, scope=scope,
                    token_budget=self.config.planner_num_ctx,
                )
                coder_ctx = self._ctx_builder.build(
                    analysis, intent=intent.value, route=route.value,
                    memory_summary="", scope=coder_scope,
                    token_budget=coder_context_budget,
                )
                if plan_text:
                    msg = pm.get_snippet("reusing_previous_plan")
                yield ("info", msg, "info")

            # ── 1. Large-input relay (before planner if needed) ───────────────
            planner_input_raw = (
                _planner_retry_msg(task, plan_text, verify_feedback, ctx=ctx)
                if is_retry
                else _planner_msg(task, ctx, all_files=self._indexer.index, verify_hint=verify_hint)
            )
            raw_tokens = estimate(planner_input_raw)
            if relay_needed(raw_tokens, self.config.planner_num_ctx, self.config.relay_threshold):
                yield ("info", f"Large input ({raw_tokens} est. tokens) → relay pre-process", "info")
                yield ("header", "relay", "header")
                relayed = self._compressor.relay_large_input(task, planner_input_raw)
                planner_input = relayed
                yield ("end", "relay", "end")
            else:
                planner_input = planner_input_raw

            # ── 2. Planner ────────────────────────────────────────────────────
            if needs_plan and (not is_retry or not plan_text):
                yield ("header", "planner", "header")
                plan_full = yield from self._run_planner(
                    planner_input, route, keep_alive="0",
                    scope=scope, task=task, is_retry=is_retry,
                )
                yield ("end", "planner", "end")
                plan_text = plan_full
                if self._is_cancelled():
                    yield ("warn", "Run cancelled.", "warn")
                    return

            # ── 3. Coder — adaptive loop (or legacy one-shot fallback) ───
            # The coder now uses an adaptive OBSERVE -> ACT loop by default:
            # write/read/delete/run guarded commands, ingest observations, then
            # hand the same PROVEN_WORK evidence to the existing verifier gate.
            project_root = self._indexer.root
            proof = ProvenWork(iteration=attempt + 1)

            use_coder_loop = (
                bool(getattr(self.config, "coder_loop", True))
                and route in (Route.PLANNER_THEN_CODER, Route.DIAGNOSTIC_LOOP)
            )
            steps: list[str] = []
            use_stepwise = False

            coder_full_for_verify = ""   # collect last coder output for the LLM verifier msg

            if use_coder_loop:
                (
                    coder_full_for_verify,
                    _accumulated_changes,
                    _accumulated_deletions,
                ) = yield from run_coder_loop(
                    self,
                    task,
                    plan_text,
                    analysis,
                    intent,
                    route,
                    coder_ctx,
                    _cmd_registry,
                    proof,
                    attempt,
                    verify_feedback,
                    project_root,
                    _original_snapshots,
                )
                if self._is_cancelled():
                    yield ("warn", "Run cancelled before final verification.", "warn")
                    return

            elif use_stepwise:
                note = pm.get_snippet("stepwise_execution_note").format(n=len(steps))
                yield ("info", note, "info")
                for sidx, step in enumerate(steps):
                    step_num = sidx + 1
                    yield ("info", f"Step {step_num}/{len(steps)}: {step[:80]}", "info")

                    # ── NEW: fresh per-step "get" from the vector knowledge base ─────
                    # This is the "continuous gets" the user wanted. The step text
                    # itself is an excellent query for semantic + symbol retrieval.
                    step_retrieved: list[RetrievedChunk] = []
                    if self._knowledge is not None:
                        try:
                            note = pm.get_snippet("retrieval_step_note").format(num=step_num)
                            yield ("info", note, "info")
                            step_retrieved = self._knowledge.retrieve(
                                step,
                                top_k=getattr(self.config, "retrieval_top_k", 12),
                            )
                            if step_retrieved:
                                tool_note = pm.get_snippet("knowledge_get_tool").format(num=step_num, count=len(step_retrieved))
                                yield ("tool", tool_note, "tool")
                        except Exception as exc:
                            yield ("warn", f"step retrieval failed: {exc}", "warn")

                    # fresh context for this micro-task (files may have changed from prev steps)
                    step_coder_ctx = self._ctx_builder.build(
                        analysis, intent=intent.value, route=route.value,
                        memory_summary="", scope="expanded",
                        token_budget=coder_context_budget,
                    )
                    step_coder_ctx.retrieved_chunks = step_retrieved  # fresh per-step "get" — the magic of continuous retrieval

                    pm = load_prompts()
                    forced_prefix = pm.get_snippet("coder_step_forced_prefix") or (
                        "CRITICAL: Implement ONLY the SINGLE step below. Begin response with '### FILE:'.\n"
                        "Think: how exactly, risks, exact verify command for THIS step.\n\n"
                    )
                    cur_label = (pm.get_snippet("step_input_current_step_label") or "CURRENT STEP TO IMPLEMENT NOW ({num}/{total}):\n").format(num=step_num, total=len(steps))

                    # Build the project-file context block for this step so the coder
                    # sees the real types, signatures, and test contracts before writing.
                    ctx_block = _coder_context_block(step_coder_ctx, pm, step_mode=True)

                    # Pre-step disk reads: inject the CURRENT on-disk state of files
                    # referenced by this step + files touched in previous steps.
                    # This is critical: after step N writes a broken file, step N+1 coder
                    # needs to SEE that broken file to know what to fix.
                    _step_refs = _extract_step_file_refs(step)
                    _prev_touched = [f for f in _accumulated_changes if f not in _step_refs]
                    _to_read = list(dict.fromkeys(_step_refs + _prev_touched))[:8]
                    _disk_reads: list[tuple[str, str]] = []
                    for _ref in _to_read:
                        _target = (project_root / _ref).resolve()
                        try:
                            if _target.exists():
                                _raw = _target.read_text("utf-8", errors="replace")
                                _excerpt = _raw[-2500:] if len(_raw) > 2500 else _raw
                                _disk_reads.append((_ref, _excerpt))
                                yield ("tool", f"read_file({_ref!r}) -> {len(_raw)} chars [pre-step {step_num}]", "tool")
                        except OSError:
                            pass

                    disk_section = ""
                    if _disk_reads:
                        _ds_parts = [
                            "CURRENT FILE STATE ON DISK (actual content right now — "
                            "fix any errors you see, use these types/signatures exactly):"
                        ]
                        for _ref, _excerpt in _disk_reads:
                            _ds_parts.append(f"\n### FILE: {_ref}\n```\n{_excerpt}\n```")
                        disk_section = "\n".join(_ds_parts) + "\n\n"

                    # Explicit per-step file constraint
                    _step_file_constraint = ""
                    if _step_refs:
                        _step_file_constraint = (
                            f"FILES FOR THIS STEP ONLY: {', '.join(_step_refs)}\n"
                            "DO NOT modify any other file. One step = one file (or the files listed above).\n\n"
                        )

                    # Command registry section injected into coder prompt
                    _cmd_block = _cmd_registry.format_for_prompt()

                    step_input = forced_prefix + (
                        f"{pm.get_snippet('step_input_full_task_label') or 'Full task: '}{task}\n\n"
                        f"{pm.get_snippet('step_input_overall_plan_label') or 'OVERALL PLAN (for reference):\\n'}{plan_text[:1500]}\n\n"
                        + (f"{_cmd_block}\n\n" if _cmd_block else "")
                        + (f"PROJECT FILES (types, test contracts — match signatures exactly):\n{ctx_block}\n\n" if ctx_block else "")
                        + disk_section
                        + _step_file_constraint
                        + f"{cur_label}{step}\n\n"
                        f"{pm.get_snippet('step_input_prev_steps_label') or 'Previous steps completed in this attempt: '}{sidx}\n"
                        + (pm.get_snippet("step_input_after_change") or
                           "After your change the runtime will run your suggested STEP_VERIFY command and other tests.\n"
                           "Output the file(s) + ### STEP_VERIFY: <cmd>")
                    )
                    if verify_feedback:
                        step_input += f"\n\n{pm.get_snippet('step_input_feedback_suffix') or 'Previous verifier feedback to address: '}{verify_feedback[:1200]}"

                    yield ("header", "coder", "header")
                    coder_step_out = yield from self._run_coder(step_input)
                    yield ("end", "coder", "end")
                    if self._is_cancelled():
                        yield ("warn", "Run cancelled before applying coder output.", "warn")
                        return
                    coder_full_for_verify = coder_step_out  # last one wins for the final LLM verify msg

                    # Parse + apply only the changes from this step
                    step_changes = _parse_file_changes(coder_step_out)
                    step_deletes = _parse_file_deletions(coder_step_out)
                    step_tests   = _parse_step_tests(coder_step_out)
                    for rel_path, content in step_changes:
                        _accumulated_changes[rel_path] = content
                    _accumulated_deletions.update(step_deletes)
                    self.pending_file_changes = list(_accumulated_changes.items())
                    self.pending_file_deletions = list(_accumulated_deletions)
                    self._memory.track_files([p for p, _ in step_changes])

                    # Apply writes for this step immediately (so next step + tests see it).
                    # No SHA256 read-back — write success is proven by compile+test, not byte comparison.
                    for rel_path, content in step_changes:
                        target = (project_root / rel_path).resolve()
                        try:
                            target.relative_to(project_root.resolve())
                            if attempt == 0 and rel_path not in _original_snapshots:
                                _original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(content, encoding="utf-8")
                            file_bytes = target.stat().st_size
                            yield ("tool", f"write_file({rel_path!r}) -> {file_bytes} bytes [step {step_num}]", "tool")
                            proof.file_checks.append({"path": rel_path, "operation": "write", "ok": True, "bytes": file_bytes, "step": step_num})
                        except (ValueError, OSError) as exc:
                            proof.file_checks.append({"path": rel_path, "operation": "write", "ok": False, "error": str(exc), "step": step_num})
                            yield ("error", f"File write failed for {rel_path}: {exc}", "error")

                    for rel_path in step_deletes:
                        target = (project_root / rel_path).resolve()
                        try:
                            target.relative_to(project_root.resolve())
                            if attempt == 0 and rel_path not in _original_snapshots:
                                _original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None
                            if target.exists():
                                target.unlink()
                            ok = not target.exists()
                            yield ("tool", f"delete_file({rel_path!r}) -> {'ok' if ok else 'FAILED'} [step {step_num}]", "tool")
                            proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": ok, "step": step_num})
                        except (ValueError, OSError) as exc:
                            proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": False, "error": str(exc), "step": step_num})
                            yield ("error", f"File delete failed for {rel_path}: {exc}", "error")

                    # STEP_TEST: write temp test files, patch test runner, run, then delete.
                    # This is the "proven work" concept: the coder writes a targeted test for
                    # the current step's logic and we execute it before the full suite.
                    _temp_test_paths: list[Path] = []
                    _patched_runner: tuple[Path | None, str | None] = (None, None)
                    _do_patch = bool(get_toolchain("step_test_dotnet_patch_program", True))
                    _do_delete = bool(get_toolchain("step_test_delete_after", True))
                    for rel_test_path, test_content in step_tests:
                        test_target = (project_root / rel_test_path).resolve()
                        try:
                            test_target.relative_to(project_root.resolve())
                            test_target.parent.mkdir(parents=True, exist_ok=True)
                            test_target.write_text(test_content, "utf-8")
                            _temp_test_paths.append(test_target)
                            yield ("tool", f"write_step_test({rel_test_path!r}) [temp, step {step_num}]", "tool")
                            # For .NET: patch the test runner's Program.cs so the class gets called
                            if _do_patch and "dotnet" in _project_toolchains and test_target.suffix.lower() == ".cs":
                                class_name = test_target.stem
                                _patched_runner = _patch_dotnet_test_runner(test_target.parent, class_name)
                                if _patched_runner[0]:
                                    yield ("tool", f"patch_test_runner({_patched_runner[0].name!r}) -> add {class_name}.RunAll()", "tool")
                        except (ValueError, OSError) as exc:
                            yield ("warn", f"STEP_TEST write failed ({rel_test_path}): {exc}", "warn")

                    # Per-step verification: compile gate + project commands + coder STEP_VERIFY
                    step_verif_cmds: list[VerificationCommand] = []
                    step_verify_match = re.search(
                        rf"{re.escape(get_marker('step_verify') or '### STEP_VERIFY:')}\s*(.+)$",
                        coder_step_out, re.M | re.I,
                    )
                    if step_verify_match:
                        raw_cmd = _clean_command_line(step_verify_match.group(1))
                        if _is_safe_verification_command(raw_cmd) and not _cmd_registry.is_toolchain_mismatch(raw_cmd):
                            step_verif_cmds.append(VerificationCommand(
                                cmd=raw_cmd,
                                purpose=f"step {step_num} suggested verify",
                                source="coder-step",
                                is_test=_is_test_verification_command(raw_cmd),
                            ))
                        elif _cmd_registry.is_toolchain_mismatch(raw_cmd):
                            tc = _command_toolchain(raw_cmd)
                            yield ("warn",
                                   f"Step {step_num}: ignored STEP_VERIFY `{raw_cmd}` "
                                   f"(toolchain `{tc}` not in project: "
                                   f"{', '.join(sorted(_project_toolchains))}). "
                                   "Using project commands instead.",
                                   "warn")
                    # Compile gate first (fast syntax check), then project test commands
                    step_verif_cmds.extend(self._verification_commands(route))
                    step_verif_cmds = _dedupe_verification_commands(step_verif_cmds)
                    if step_changes and _cmd_registry.compile_commands:
                        _existing = {c.cmd for c in step_verif_cmds}
                        step_verif_cmds = [
                            c for c in _cmd_registry.compile_commands if c.cmd not in _existing
                        ] + step_verif_cmds

                    _step_failed = False
                    _verif_ran = 0
                    if step_verif_cmds:
                        for vcmd in step_verif_cmds:
                            _missing = _missing_command_paths(vcmd.cmd, project_root)
                            if _missing:
                                yield ("warn",
                                       f"Step {step_num}: skipped `{vcmd.cmd}` — "
                                       f"file(s) not found: {', '.join(_missing)}",
                                       "warn")
                                continue
                            _verif_ran += 1
                            yield ("tool", f"run_command({vcmd.cmd!r}) -> {vcmd.purpose} [step {step_num}]", "tool")
                            res = self._run_verification_command(vcmd)
                            proof.commands.append(res)
                            proof.commands_planned.append({
                                "cmd": vcmd.cmd, "purpose": vcmd.purpose, "source": vcmd.source,
                                "is_test": vcmd.is_test, "step": step_num,
                            })
                            st = "ok" if res["exit_code"] == 0 else "FAILED"
                            yield ("tool", f"verify_command({vcmd.cmd!r}) -> {st} exit={res['exit_code']} [step {step_num}]", "tool")
                            yield ("log", json.dumps({
                                "type": "build_test", "cmd": vcmd.cmd, "purpose": vcmd.purpose,
                                "source": vcmd.source, "is_test": vcmd.is_test,
                                "exit_code": res["exit_code"], "step": step_num,
                                "duration_seconds": res["duration_seconds"], "output": res["output"][-2000:],
                            }), "log")
                            if res["exit_code"] != 0:
                                _tail = res.get("output", "")[-600:].strip()
                                verify_feedback = (
                                    f"Step {step_num} verification failed: `{vcmd.cmd}`"
                                    + (f"\nTest output:\n{_tail}" if _tail else "")
                                )
                                _step_failed = True
                                break
                        if not _step_failed and _verif_ran:
                            yield ("info", f"Step {step_num}/{len(steps)} verified.", "info")
                        elif not _step_failed:
                            yield ("warn", f"Step {step_num}: no verification command could run (referenced files not yet created).", "warn")
                    else:
                        msg = pm.get_snippet("proven_work_no_verify_command").format(num=step_num)
                        yield ("warn", msg, "warn")

                    # Clean up temp test files and restore patched runner (regardless of outcome)
                    if _do_delete:
                        for _tmp in _temp_test_paths:
                            try:
                                _tmp.unlink(missing_ok=True)
                                yield ("tool", f"delete_step_test({_tmp.name!r}) [cleanup step {step_num}]", "tool")
                            except OSError:
                                pass
                    if _patched_runner[0] and _patched_runner[1] is not None:
                        try:
                            _patched_runner[0].write_text(_patched_runner[1], "utf-8")
                            yield ("tool", f"restore_test_runner({_patched_runner[0].name!r})", "tool")
                        except OSError:
                            pass

                    # Step failed — continue so all stubs get implemented before outer retry
                    if _step_failed:
                        yield ("warn", f"Step {step_num} failed — continuing with remaining steps", "warn")
                        continue

            else:
                # ── Legacy / fallback one-shot coder (CODER_ONLY, unclear plans, retries) ──
                yield ("header", "coder", "header")
                pm = load_prompts()
                forced_prefix = pm.get_snippet("coder_forced_prefix_legacy") or (
                    "CRITICAL: Your response MUST begin with '### FILE:' on the very first line.\n"
                    "Output ONLY source files as specified in the system prompt.\n\n"
                )
                forced_prefix += (
                    "\nFULL-PASS MODE: implement or repair ALL files needed for the complete task in this response. "
                    "Do not stop after one plan step. Output complete file contents for every changed file. "
                    "Use plain ASCII in code and comments unless a test explicitly requires a non-ASCII character. "
                    "Do not use curly quotes, nonbreaking hyphens, or invisible Unicode in source code.\n\n"
                )
                if is_retry:
                    coder_input = forced_prefix + _coder_retry_msg(task, coder_ctx, plan_text, verify_feedback)
                else:
                    coder_input = forced_prefix + _coder_msg(task, coder_ctx, plan=plan_text)

                coder_full = yield from self._run_coder(coder_input)
                yield ("end", "coder", "end")
                if self._is_cancelled():
                    yield ("warn", "Run cancelled before applying coder output.", "warn")
                    return
                coder_full_for_verify = coder_full

                # Accumulate + apply exactly as before (one big batch)
                new_changes   = _parse_file_changes(coder_full)
                new_deletions = _parse_file_deletions(coder_full)
                for rel_path, content in new_changes:
                    _accumulated_changes[rel_path] = content
                _accumulated_deletions.update(new_deletions)
                self.pending_file_changes   = list(_accumulated_changes.items())
                self.pending_file_deletions = list(_accumulated_deletions)
                self._memory.track_files([p for p, _ in self.pending_file_changes])

                for rel_path, content in self.pending_file_changes:
                    target = (project_root / rel_path).resolve()
                    try:
                        target.relative_to(project_root.resolve())
                        if attempt == 0 and rel_path not in _original_snapshots:
                            _original_snapshots[rel_path] = (
                                target.read_text("utf-8") if target.exists() else None
                            )
                        size = len(content.encode("utf-8"))
                        yield ("tool", f"apply_file({rel_path!r}) -> write {size} bytes", "tool")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(content, encoding="utf-8")
                        actual = target.read_text("utf-8")
                        digest = hashlib.sha256(actual.encode("utf-8")).hexdigest()[:12]
                        ok = actual == content
                        proof.file_checks.append({
                            "path": rel_path, "operation": "write", "ok": ok,
                            "bytes": len(actual.encode("utf-8")), "sha256": digest,
                        })
                        yield ("tool", f"verify_file_write({rel_path!r}) -> {'ok' if ok else 'FAILED'} sha256={digest}", "tool")
                    except (ValueError, OSError) as exc:
                        proof.file_checks.append({"path": rel_path, "operation": "write", "ok": False, "error": str(exc)})
                        yield ("error", f"File write failed for {rel_path}: {exc}", "error")

                for rel_path in self.pending_file_deletions:
                    target = (project_root / rel_path).resolve()
                    try:
                        target.relative_to(project_root.resolve())
                        if attempt == 0 and rel_path not in _original_snapshots:
                            _original_snapshots[rel_path] = (
                                target.read_text("utf-8") if target.exists() else None
                            )
                        yield ("tool", f"delete_file({rel_path!r}) -> apply deletion", "tool")
                        if target.exists():
                            target.unlink()
                        ok = not target.exists()
                        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": ok})
                        yield ("tool", f"verify_file_delete({rel_path!r}) -> {'ok' if ok else 'FAILED'}", "tool")
                    except (ValueError, OSError) as exc:
                        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": False, "error": str(exc)})
                        yield ("error", f"File delete failed for {rel_path}: {exc}", "error")

            failed_file_checks = [item for item in proof.file_checks if not item.get("ok")]
            if failed_file_checks:
                verify_feedback = f"File application failed: {failed_file_checks[0]}"
                yield ("log", json.dumps({"type": "proven_work", **proof.to_dict()}), "log")
                yield ("log", json.dumps({"type": "verify_decision",
                                          "decision": "NEEDS_REVISION",
                                          "feedback": verify_feedback,
                                          "iteration": attempt + 1}), "log")
                if attempt < max_iter - 1:
                    yield ("warn", verify_feedback, "warn")
                    continue
                yield ("warn", f"Safety cap ({max_iter} iterations) reached.", "warn")
                break

            # ── 4a. PROVEN_WORK command evidence (final sweep + per-step already recorded) ─
            # In stepwise the per-step tests are already in proof.commands.
            # We still run the configured/full suite here for the final gate + LLM verifier.
            verification_commands = self._verification_commands(route)
            extra_planned = [
                {"cmd": item.cmd, "purpose": item.purpose, "source": item.source, "is_test": item.is_test}
                for item in verification_commands
            ]
            proof.commands_planned.extend(extra_planned)

            if verification_commands:
                yield ("log", json.dumps({"type": "proven_work_plan",
                                          "commands": extra_planned,
                                          "iteration": attempt + 1}), "log")
                for verify_cmd in verification_commands:
                    # avoid re-running identical that just passed in the last step
                    already_ran_ok = any(
                        c.get("cmd") == verify_cmd.cmd and c.get("exit_code") == 0
                        for c in proof.commands
                    )
                    if already_ran_ok:
                        continue
                    yield ("tool", f"run_command({verify_cmd.cmd!r}) -> {verify_cmd.purpose} [final]", "tool")
                    result = self._run_verification_command(verify_cmd)
                    proof.commands.append(result)
                    status = "ok" if result["exit_code"] == 0 else "FAILED"
                    yield ("tool", f"verify_command({verify_cmd.cmd!r}) -> {status} exit_code={result['exit_code']}", "tool")
                    yield ("log", json.dumps({"type": "build_test",
                                              "cmd": verify_cmd.cmd, "purpose": verify_cmd.purpose,
                                              "source": verify_cmd.source, "is_test": verify_cmd.is_test,
                                              "exit_code": result["exit_code"],
                                              "duration_seconds": result["duration_seconds"],
                                              "output": result["output"][-3000:]}), "log")
                    if result["exit_code"] != 0:
                        break
            else:
                msg = pm.get_snippet("proven_work_no_safe_command")
                yield ("warn", msg, "warn")

            yield ("log", json.dumps({"type": "proven_work", **proof.to_dict()}), "log")
            self._save_iteration_state(attempt + 1, plan_text, verify_feedback, proof)

            proof_required = self.config.require_proven_work
            require_test = self.config.require_test_for_verified
            if proof_required and not _proven_work_satisfied(proof, require_test=require_test):
                repair_changes = try_repair_known_probe(project_root)
                repaired_files = list(repair_changes)
                if repair_changes:
                    current_changes = dict(self.pending_file_changes)
                    current_changes.update(repair_changes)
                    self.pending_file_changes = list(current_changes.items())
                    proof.files_changed = list(dict.fromkeys([*proof.files_changed, *repaired_files]))
                    yield ("info", "Applied built-in sandbox repair fallback after model verification failed.", "info")
                    for rel_path, content in repair_changes.items():
                        target = (project_root / rel_path).resolve()
                        try:
                            target.relative_to(project_root.resolve())
                            if attempt == 0 and rel_path not in _original_snapshots:
                                _original_snapshots[rel_path] = (
                                    target.read_text("utf-8") if target.exists() else None
                                )
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(content, encoding="utf-8")
                            actual = target.read_text("utf-8")
                            digest = hashlib.sha256(actual.encode("utf-8")).hexdigest()[:12]
                            ok = actual == content
                            proof.file_checks.append({
                                "path": rel_path, "operation": "write", "ok": ok,
                                "bytes": len(actual.encode("utf-8")), "sha256": digest,
                                "source": "built-in sandbox fallback",
                            })
                            yield ("tool", f"repair_file({rel_path!r}) -> {'ok' if ok else 'FAILED'} sha256={digest}", "tool")
                        except (ValueError, OSError) as exc:
                            proof.file_checks.append({
                                "path": rel_path, "operation": "write", "ok": False,
                                "error": str(exc), "source": "built-in sandbox fallback",
                            })
                            yield ("error", f"Built-in sandbox repair failed for {rel_path}: {exc}", "error")
                    for verify_cmd in verification_commands:
                        yield ("tool", f"run_command({verify_cmd.cmd!r}) -> {verify_cmd.purpose} [repair]", "tool")
                        result = self._run_verification_command(verify_cmd)
                        proof.commands.append(result)
                        status = "ok" if result["exit_code"] == 0 else "FAILED"
                        yield ("tool", f"verify_command({verify_cmd.cmd!r}) -> {status} exit_code={result['exit_code']} [repair]", "tool")
                        yield ("log", json.dumps({"type": "build_test",
                                                  "cmd": verify_cmd.cmd, "purpose": verify_cmd.purpose,
                                                  "source": verify_cmd.source, "is_test": verify_cmd.is_test,
                                                  "exit_code": result["exit_code"],
                                                  "duration_seconds": result["duration_seconds"],
                                                  "output": result["output"][-3000:]}), "log")
                        if result["exit_code"] != 0:
                            break
                    yield ("log", json.dumps({"type": "proven_work", **proof.to_dict()}), "log")
                    self._save_iteration_state(attempt + 1, plan_text, verify_feedback, proof)
                    if _proven_work_satisfied(proof, require_test=require_test):
                        self._record_solution(repaired_files, attempt + 1, task, plan_text, coder_full_for_verify or plan_text, proof)
                        yield ("info", "PROVEN_WORK accepted after built-in repair fallback.", "info")
                        break

                verify_feedback = _format_proven_work_feedback(proof, require_test=require_test)
                yield ("warn", verify_feedback[:500], "warn")
                yield ("log", json.dumps({"type": "verify_decision",
                                          "decision": "NEEDS_REVISION",
                                          "feedback": verify_feedback,
                                          "iteration": attempt + 1}), "log")
                if attempt < max_iter - 1:
                    continue
                yield ("warn", f"Safety cap ({max_iter} iterations) reached.", "warn")
                break

            # ── 4b. LLM verifier, gated by PROVEN_WORK (now sees per-step evidence) ─
            yield ("header", "verify", "header")
            verify_full = yield from self._run_verifier(
                _verify_msg(task, plan_text, coder_full_for_verify or plan_text, proof)
            )
            if self._is_cancelled():
                yield ("warn", "Run cancelled.", "warn")
                return
            yield ("end", "verify", "end")

            vf_upper = verify_full.upper()

            if "UNSOLVABLE" in vf_upper:
                reason = verify_full.replace("UNSOLVABLE:", "").replace("UNSOLVABLE", "").strip()
                yield ("warn", f"Task cannot be solved — {reason[:300]}", "warn")
                yield ("log", json.dumps({"type": "verify_decision", "decision": "UNSOLVABLE",
                                          "feedback": verify_full, "iteration": attempt + 1}), "log")
                # Rollback files written during this run
                self._rollback(_original_snapshots, project_root)
                self.pending_file_changes.clear()
                self.pending_file_deletions.clear()
                break

            if "VERIFIED" in vf_upper and "NEEDS_REVISION" not in vf_upper:
                yield ("log", json.dumps({"type": "verify_decision", "decision": "VERIFIED",
                                          "feedback": "", "iteration": attempt + 1}), "log")
                self._record_solution([p for p, _ in self.pending_file_changes],
                                      attempt + 1, task, plan_text, coder_full_for_verify or plan_text, proof)
                yield ("info", "PROVEN_WORK accepted — verification commands ran and passed.", "info")
                if attempt > 0:
                    yield ("info", f"Verified after {attempt + 1} iteration(s).", "info")

                # Final extra verification pass — verifier "really does the tests"
                final_cmds = self._verification_commands(route)
                if final_cmds:
                    yield ("info", "Final full verification sweep...", "info")
                    all_ok = True
                    for fcmd in final_cmds:
                        yield ("tool", f"run_command({fcmd.cmd!r}) -> final full check", "tool")
                        fres = self._run_verification_command(fcmd)
                        proof.commands.append(fres)  # append for record
                        st = "ok" if fres["exit_code"] == 0 else "FAILED"
                        yield ("tool", f"verify_command({fcmd.cmd!r}) -> {st} (final)", "tool")
                        if fres["exit_code"] != 0:
                            all_ok = False
                            break
                    if not all_ok:
                        # demote to revision even if LLM said verified (strict)
                        verify_feedback = "Final verification sweep failed — one or more tests did not pass after VERIFIED."
                        yield ("warn", verify_feedback, "warn")
                        if attempt < max_iter - 1:
                            continue
                break

            # NEEDS_REVISION
            verify_feedback = verify_full.replace("NEEDS_REVISION:", "").strip()
            yield ("log", json.dumps({"type": "verify_decision", "decision": "NEEDS_REVISION",
                                      "feedback": verify_full, "iteration": attempt + 1}), "log")
            if attempt < max_iter - 1:
                yield ("warn", f"Needs revision ({attempt + 1}/{max_iter}) — {verify_feedback[:200]}", "warn")
            else:
                yield ("warn", f"Safety cap ({max_iter} iterations) reached — review manually.", "warn")

    # ── public API ────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self._planner_history.clear()
        self._coder_history.clear()
        self._memory.clear_task()
        self._memory.clear_session()

    def rebuild_index(self) -> int:
        self._indexer.invalidate()
        return len(self._indexer.build_index())

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def indexer(self) -> ProjectIndexer:
        return self._indexer
