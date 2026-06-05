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

import json
import re
import subprocess
from pathlib import Path
from typing import Generator

from .analyzer import InputAnalyzer
from .bridge import OllamaBridge
from .classifier import IntentClassifier, Intent
from .compress import Compressor
from .config import KratosConfig, _project_dir, GLOBAL_DIR
from .context import ContextBuilder, ContextPackage, ProjectIndexer, ScopeType
from .memory import MemoryEntry, MemoryManager
from .router import Route, Router
from .tokens import (
    estimate, estimate_messages, choose_num_ctx,
    fit_to_budget, relay_needed, model_max_ctx,
)

# ── system prompts (hardcoded, generic, caveman-short) ────────────────────────

PLANNER_SYSTEM = """\
You are Kratos Planner. Analyze the task and produce a clear, practical plan.

Answer in plain text:
- What exactly needs to be done?
- Which files are relevant or need to change?
- Step-by-step: what should the coder implement?
- Potential risks (circular imports, missing deps, breaking changes)?

Rules:
- Do NOT write code or modify files yourself.
- If test files are in context, read them first — they define the exact API.
- If the task is a question, just answer it directly.
- Be concise. Skip anything that does not apply."""

PLANNER_VERIFY_SYSTEM = """\
You are Kratos Verifier. Review whether the implementation correctly fulfills the task.

Check:
1. Does it implement ALL requirements?
2. Is the code complete — no placeholders, no "...", no unimplemented stubs?
3. Are there obvious bugs, syntax errors, or logical mistakes?
4. Does it follow the plan?

Respond ONLY with one of:

VERIFIED
(implementation is complete and correct)

NEEDS_REVISION:
<specific list of what is missing or wrong — reference exact file names>

UNSOLVABLE:
<explanation — only when requirements are contradictory or dependencies missing>"""

CODER_SYSTEM = """\
You are Kratos Coder. Output ONLY file changes in the exact format below.

OUTPUT RULES:
1. Start with "### FILE:" on the very first character — no greeting, no preamble.
2. Write EVERY source file listed in the plan that still contains stubs or NotImplementedError.
3. Write COMPLETE file content — every function fully implemented, no "...", no pass-with-TODO.
4. Do NOT create undocumented extra files (no extra READMEs, no helper files not in the plan).
5. Do NOT rewrite a complete file unless the plan explicitly says to fix it.
6. Before each import ask: does that module already import from THIS module?
   If yes, do NOT import it — define what you need locally instead.

FORMAT:
### FILE: relative/path/to/file.ext
```language
complete file content
```

### FILE: another/file.ext
```language
complete file content
```

To delete: ### DELETE: relative/path/to/file.ext

After all files:
### SUMMARY
Changed: file1.py, file2.py — one-line description of what was done"""

RELAY_SYSTEM = """\
Extract essential information from the large project context below for a planner.
Output a compact structured summary (max 600 words):
1. Project structure (key files and their roles)
2. Most relevant code (only what matters for the task)
3. Errors or constraints
4. What the planner needs to answer the task
Output ONLY the summary. No preamble."""

# ── token predict limits ──────────────────────────────────────────────────────

_PLAN_PREDICT       = 2048    # concise plan for simple tasks
_PLAN_PREDICT_HEAVY = 4096    # complex multi-file / architecture tasks
_CODE_PREDICT       = 16384   # up to ~2 500 lines of code
_VERIFY_PREDICT     = 512     # short VERIFIED / NEEDS_REVISION response
_RELAY_PREDICT      = 1200    # compact extract


# ── output parsers ────────────────────────────────────────────────────────────

_FILE_CHANGE_RE = re.compile(
    r'###\s+FILE:\s*(.+?)\s*\n```(?:\w+)?\n(.*?)```',
    re.S,
)
_FILE_DELETE_RE = re.compile(r'###\s+DELETE:\s*(.+?)\s*$', re.M)


def _parse_file_changes(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).strip(), m.group(2)) for m in _FILE_CHANGE_RE.finditer(text)]


def _parse_file_deletions(text: str) -> list[str]:
    return [m.group(1).strip() for m in _FILE_DELETE_RE.finditer(text)]


# ── scope / num_ctx selection ─────────────────────────────────────────────────

def _scope_for(route: Route, intent: Intent) -> ScopeType:
    if route == Route.DIRECT_ANSWER:
        return "none"
    if route == Route.PLANNER_ONLY:
        if intent in (Intent.QUESTION, Intent.EXPLAIN):
            return "minimal"
        return "architecture"
    if route == Route.CODER_ONLY:
        if intent == Intent.FOLLOWUP:
            return "patch_context"
        if intent == Intent.SHELL_GIT:
            return "none"
        return "targeted"
    if route == Route.DIAGNOSTIC_LOOP:
        return "diagnostic"
    if route == Route.PLANNER_THEN_CODER:
        return "architecture"
    return "minimal"


def _coder_scope_for(intent: Intent) -> ScopeType:
    if intent == Intent.FOLLOWUP:
        return "patch_context"
    return "expanded"


# ── dynamic reasoning ─────────────────────────────────────────────────────────

def _needs_thinking(
    task: str, scope: ScopeType, route: Route, is_retry: bool, n_files: int
) -> bool:
    """Use chain-of-thought only when the task genuinely needs it.

    Signals: complex scope, retry, multi-file context, long task description,
    error/diagnostic route. Saves VRAM time on simple tasks.
    """
    if is_retry:
        return True
    if scope in ("architecture", "diagnostic"):
        return True
    if route in (Route.DIAGNOSTIC_LOOP,):
        return True
    if n_files > 5:
        return True
    if len(task.split()) > 40:
        return True
    return False


# ── message builders ──────────────────────────────────────────────────────────

def _planner_msg(task: str, ctx: ContextPackage, all_files: list | None = None) -> str:
    parts: list[str] = []
    if ctx.project_description:
        parts.append(ctx.project_description)
    if all_files and not ctx.project_description:
        listing = "\n".join(f"  {e.rel_path}" for e in all_files[:100])
        parts.append(f"All project files:\n{listing}")
    if ctx.memory_summary:
        parts.append(ctx.memory_summary)
    if ctx.files:
        parts.append("File contents (most relevant):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1500)}")
    if ctx.error_lines:
        parts.append("Errors/logs:\n" + "\n".join(ctx.error_lines[:15]))
    parts.append(f"Task: {task}")
    return "\n\n".join(p for p in parts if p)


def _coder_msg(task: str, ctx: ContextPackage, plan: str) -> str:
    parts: list[str] = []
    if plan:
        parts.append(f"Plan:\n{plan}")
    if ctx.files:
        noise = ("README", "ARCH", "REQUIRE", "docs/", "CHANGELOG", "LICENSE")
        test_files = [f for f in ctx.files
                      if any(x in f.rel_path for x in ("test_", "_test.", ".spec.", ".test."))]
        src_files  = [f for f in ctx.files
                      if f not in test_files and not any(x in f.rel_path for x in noise)]
        stub_files = [f for f in src_files if "NotImplementedError" in (f.content or "")]
        done_files = [f for f in src_files if f not in stub_files]

        if test_files:
            parts.append("TEST FILES — these define the exact API. Match every signature exactly:")
            for f in test_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1200)}")
        if stub_files:
            names = ", ".join(f.rel_path for f in stub_files)
            parts.append(f"STUB FILES — IMPLEMENT ALL: {names}")
            for f in stub_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1000)}")
        if done_files:
            parts.append("Already-implemented (reference only — do NOT rewrite unless plan says to):")
            for f in done_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(600)}")
    if ctx.error_lines:
        parts.append("Errors:\n" + "\n".join(ctx.error_lines[:10]))
    parts.append(f"Task: {task}")
    return "\n\n".join(p for p in parts if p)


def _planner_retry_msg(
    task: str, prev_plan: str, verify_feedback: str, ctx: ContextPackage | None = None
) -> str:
    parts = [f"Task: {task}"]
    if ctx and ctx.files:
        parts.append("Current file state (updated since last iteration):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(800)}")
    parts.append(f"Previous plan:\n{prev_plan[:600]}")
    parts.append(
        f"Verifier / test feedback — what still needs to be fixed:\n{verify_feedback[:2000]}\n\n"
        "Produce a precise plan for what the coder must implement to fix all issues. "
        "List each file and function. For circular imports name the exact import line to remove."
    )
    return "\n\n".join(parts)


def _coder_retry_msg(
    task: str, ctx: ContextPackage, plan: str, verify_feedback: str
) -> str:
    parts: list[str] = []
    if ctx.files:
        noise = ("README", "ARCH", "REQUIRE", "docs/", "CHANGELOG", "LICENSE")
        test_files = [f for f in ctx.files
                      if any(x in f.rel_path for x in ("test_", "_test.", ".spec.", ".test."))]
        src_files  = [f for f in ctx.files
                      if f not in test_files and not any(x in f.rel_path for x in noise)]
        if test_files:
            parts.append("TEST FILES — exact API you must satisfy:")
            for f in test_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1000)}")
        parts.append("Current source files:")
        for f in src_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1000)}")
    parts.append(f"Revised plan:\n{plan}")
    parts.append(
        f"Required fixes:\n{verify_feedback[:2000]}\n\n"
        "Fix ALL issues. Every NotImplementedError must be replaced."
    )
    parts.append(f"Task: {task}")
    return "\n\n".join(p for p in parts if p)


def _verify_msg(task: str, plan: str, coder_output: str) -> str:
    return (
        f"Task: {task}\n\n"
        f"Plan given to coder:\n{plan[:600]}\n\n"
        f"Coder output:\n{coder_output[:2000]}"
    )


def _clarification_msg(analysis, intent: Intent) -> str:
    kw_str = ", ".join(analysis.keywords[:6]) if analysis.keywords else "none"
    return (
        f"Your request is unclear. Detected keywords: [{kw_str}]. "
        "Please specify: what should change, which file(s), and what the expected result is."
    )


# ── direct search (no LLM) ────────────────────────────────────────────────────

def _direct_file_search(indexer: ProjectIndexer, keywords: list[str], file_paths: list[str]) -> str:
    results: list[str] = []
    index = indexer.index
    if not index:
        return "No project files indexed."
    for fp in file_paths:
        fp_l = fp.lower().replace("\\", "/")
        for e in index:
            if fp_l in e.rel_path.lower():
                results.append(f"  {e.rel_path}  [pri={e.priority}]")
    for kw in keywords[:5]:
        for e in index:
            line = f"  {e.rel_path}  [pri={e.priority}]"
            if kw.lower() in e.rel_path.lower() and line not in results:
                results.append(line)
    if not results:
        top = [f"  {e.rel_path}" for e in index[:20]]
        return "No direct matches. Top project files:\n" + "\n".join(top)
    return "Matching files:\n" + "\n".join(results[:20])


def _direct_code_search(indexer: ProjectIndexer, keywords: list[str]) -> str:
    lines: list[str] = []
    for kw in keywords[:3]:
        hits = indexer.search_content(kw, max_results=8)
        if hits:
            lines.append(f"\nResults for '{kw}':")
            for entry, lineno, text in hits:
                lines.append(f"  {entry.rel_path}:{lineno}  {text}")
    return "\n".join(lines) if lines else "No matches found in project files."


# ── main agent ────────────────────────────────────────────────────────────────

class KratosAgent:
    def __init__(self, config: KratosConfig, bridge: OllamaBridge) -> None:
        self.config = config
        self.bridge = bridge

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

        # Accumulated token usage for this session
        self._session_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0
        }

        # File operations extracted from last coder run — applied by caller
        self.pending_file_changes:   list[tuple[str, str]] = []
        self.pending_file_deletions: list[str] = []

    # ── session usage ─────────────────────────────────────────────────────────

    @property
    def session_usage(self) -> dict[str, int]:
        return dict(self._session_usage)

    def _record_usage(self, usage_json: str) -> None:
        try:
            d = json.loads(usage_json)
            self._session_usage["prompt_tokens"]     += d.get("prompt_tokens", 0)
            self._session_usage["completion_tokens"] += d.get("completion_tokens", 0)
        except Exception:
            pass

    # ── intelligent pipeline ─────────────────────────────────────────────────

    def process(self, task: str) -> Generator[tuple[str, str, str], None, None]:
        """Full pipeline. Yields (source, content, kind) events."""
        self.pending_file_changes.clear()
        self.pending_file_deletions.clear()

        analysis = self._analyzer.analyze(task)
        intent   = self._classifier.classify(analysis)
        route    = self._router.route(intent)

        yield ("router", f"intent={intent.value}  route={route.value}", "info")

        # ── memory retrieval ──────────────────────────────────────────────────
        mem_entries    = self._memory.get_relevant(
            analysis.keywords, categories=["solution", "file_role", "decision", "convention"]
        )
        memory_summary = self._memory.format_for_prompt(mem_entries)

        # ── context build ─────────────────────────────────────────────────────
        scope = _scope_for(route, intent)
        n_indexed = len(self._indexer.index)
        yield ("tool", f"index_project({self._indexer.root.name!r}) → {n_indexed} files", "tool")

        ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary=memory_summary, scope=scope,
            token_budget=self.config.planner_num_ctx,
        )
        coder_scope = _coder_scope_for(intent)
        coder_ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary="", scope=coder_scope,
            token_budget=self.config.coder_num_ctx,
        )
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
                _planner_msg(task, ctx, all_files=self._indexer.index),
                route, keep_alive="5m", scope=scope, task=task, is_retry=False,
            )
            yield ("end", "planner", "end")
            # Async memory extraction
            mem_entries_new = self._compressor.generate_memory(task, plan_full, "", [])
            self._memory.add_from_compress(mem_entries_new)
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
                yield ("info", f"Revising — {iter_label}", "info")
                self._indexer.build_index()
                ctx = self._ctx_builder.build(
                    analysis, intent=intent.value, route=route.value,
                    memory_summary=memory_summary, scope=scope,
                    token_budget=self.config.planner_num_ctx,
                )
                coder_ctx = self._ctx_builder.build(
                    analysis, intent=intent.value, route=route.value,
                    memory_summary="", scope=coder_scope,
                    token_budget=self.config.coder_num_ctx,
                )

            # ── 1. Large-input relay (before planner if needed) ───────────────
            planner_input_raw = (
                _planner_retry_msg(task, plan_text, verify_feedback, ctx=ctx)
                if is_retry
                else _planner_msg(task, ctx, all_files=self._indexer.index)
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
            if needs_plan or is_retry:
                yield ("header", "planner", "header")
                plan_full = yield from self._run_planner(
                    planner_input, route, keep_alive="0",
                    scope=scope, task=task, is_retry=is_retry,
                )
                yield ("end", "planner", "end")
                plan_text = plan_full

            # ── 3. Coder ──────────────────────────────────────────────────────
            yield ("header", "coder", "header")
            forced_prefix = (
                "CRITICAL: Your response MUST begin with '### FILE:' on the very first line.\n"
                "Output ONLY source files as specified in the system prompt.\n\n"
            )
            if is_retry:
                coder_input = forced_prefix + _coder_retry_msg(task, coder_ctx, plan_text, verify_feedback)
            else:
                coder_input = forced_prefix + _coder_msg(task, coder_ctx, plan=plan_text)

            coder_full = yield from self._run_coder(coder_input)
            yield ("end", "coder", "end")

            # Accumulate file ops — later iterations override earlier for same path
            new_changes   = _parse_file_changes(coder_full)
            new_deletions = _parse_file_deletions(coder_full)
            for rel_path, content in new_changes:
                _accumulated_changes[rel_path] = content
            _accumulated_deletions.update(new_deletions)
            self.pending_file_changes   = list(_accumulated_changes.items())
            self.pending_file_deletions = list(_accumulated_deletions)

            # Track files in session memory
            self._memory.track_files([p for p, _ in self.pending_file_changes])

            # Write files to disk so the verifier + next planner see updated state.
            # Snapshot originals on first attempt for potential rollback.
            project_root = self._indexer.root
            for rel_path, content in self.pending_file_changes:
                target = (project_root / rel_path).resolve()
                try:
                    target.relative_to(project_root.resolve())
                    if attempt == 0 and rel_path not in _original_snapshots:
                        _original_snapshots[rel_path] = (
                            target.read_text("utf-8") if target.exists() else None
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                except (ValueError, OSError):
                    pass
            for rel_path in self.pending_file_deletions:
                target = (project_root / rel_path).resolve()
                try:
                    target.relative_to(project_root.resolve())
                    if attempt == 0 and rel_path not in _original_snapshots:
                        _original_snapshots[rel_path] = (
                            target.read_text("utf-8") if target.exists() else None
                        )
                    if target.exists():
                        target.unlink()
                except (ValueError, OSError):
                    pass

            # Save iteration state for crash recovery
            self._save_iteration_state(attempt + 1, plan_text, verify_feedback)

            # ── 4a. Real test runner ──────────────────────────────────────────
            if self.config.test_cmd or (route == Route.DIAGNOSTIC_LOOP and self.config.build_cmd):
                build_result = self._run_build_test()
                if build_result is not None:
                    ok, output, cmd = build_result
                    yield ("tool", f"run_command({cmd!r}) → {'ok' if ok else 'FAILED'}", "tool")
                    yield ("log", json.dumps({"type": "build_test", "cmd": cmd,
                                              "exit_code": 0 if ok else 1,
                                              "output": output[-3000:]}), "log")
                    if ok:
                        yield ("info", "All tests passed — task complete.", "info")
                        yield ("log", json.dumps({"type": "verify_decision",
                                                  "decision": "VERIFIED", "feedback": "",
                                                  "iteration": attempt + 1}), "log")
                        self._record_solution([p for p, _ in self.pending_file_changes],
                                              attempt + 1, task, plan_text, coder_full)
                        break
                    else:
                        verify_feedback = (
                            f"Tests failed (iteration {attempt + 1}).\n{output[-3000:]}"
                        )
                        yield ("warn", f"Tests failed — {output[-200:]}", "warn")
                        yield ("log", json.dumps({"type": "verify_decision",
                                                  "decision": "NEEDS_REVISION",
                                                  "feedback": verify_feedback,
                                                  "iteration": attempt + 1}), "log")
                        if attempt < max_iter - 1:
                            continue
                        else:
                            yield ("warn", f"Safety cap ({max_iter} iterations) reached.", "warn")
                            break

            # ── 4b. LLM Verifier (fallback when no test_cmd) ─────────────────
            yield ("header", "verify", "header")
            verify_full = yield from self._run_verifier(
                _verify_msg(task, plan_text, coder_full)
            )
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
                                      attempt + 1, task, plan_text, coder_full)
                if attempt > 0:
                    yield ("info", f"Verified after {attempt + 1} iteration(s).", "info")
                break

            # NEEDS_REVISION
            verify_feedback = verify_full.replace("NEEDS_REVISION:", "").strip()
            yield ("log", json.dumps({"type": "verify_decision", "decision": "NEEDS_REVISION",
                                      "feedback": verify_full, "iteration": attempt + 1}), "log")
            if attempt < max_iter - 1:
                yield ("warn", f"Needs revision ({attempt + 1}/{max_iter}) — {verify_feedback[:200]}", "warn")
            else:
                yield ("warn", f"Safety cap ({max_iter} iterations) reached — review manually.", "warn")

    # ── model runners ─────────────────────────────────────────────────────────

    def _auto_compress_if_needed(self, history: list[dict], model: str, num_ctx: int) -> bool:
        """Compress history in-place if it's approaching the context limit."""
        if not self.config.auto_compress:
            return False
        tok = estimate_messages(history)
        threshold = int(num_ctx * self.config.compress_threshold)
        if tok > threshold or len(history) > self.config.max_history_pairs * 2:
            return self._compressor.compress_history(history, keep_pairs=4)
        return False

    def _run_planner(
        self, msg: str, route: Route, keep_alive: str = "0",
        scope: ScopeType = "targeted", task: str = "", is_retry: bool = False,
    ) -> Generator:
        needs_thinking = _needs_thinking(task, scope, route, is_retry, 0)
        planner_think: bool | None = None if needs_thinking else False

        # Choose num_ctx dynamically
        prompt_msgs = [
            {"role": "system", "content": PLANNER_SYSTEM},
            *self._planner_history,
            {"role": "user", "content": msg},
        ]
        prompt_tok = estimate_messages(prompt_msgs)
        num_ctx = choose_num_ctx(
            model=self.config.planner_model,
            prompt_tokens=prompt_tok,
            max_new_tokens=_PLAN_PREDICT_HEAVY if needs_thinking else _PLAN_PREDICT,
            vram_ceiling=self.config.vram_ctx_ceiling,
        )
        # Cap at model's actual hardware max
        num_ctx = min(num_ctx, model_max_ctx(self.config.planner_model))

        self._auto_compress_if_needed(self._planner_history, self.config.planner_model, num_ctx)

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
                num_predict=_PLAN_PREDICT_HEAVY if needs_thinking else _PLAN_PREDICT,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                think=planner_think,
            ):
                if kind == "think":
                    thinking += token
                elif kind == "usage":
                    self._record_usage(token)
                    yield ("usage", token, "usage")
                else:
                    full += token
                    yield ("planner", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Planner interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Planner failed: {exc}", "error")

        if thinking:
            yield ("log", json.dumps({"type": "model_thinking", "role": "planner",
                                       "chars": len(thinking)}), "log")
        yield ("log", json.dumps({"type": "model_output", "role": "planner",
                                   "chars": len(full)}), "log")

        self._planner_history.append({"role": "user",      "content": msg})
        self._planner_history.append({"role": "assistant", "content": full})
        return full

    def _run_coder(self, msg: str) -> Generator:
        prompt_msgs = [
            {"role": "system", "content": CODER_SYSTEM},
            *self._coder_history,
            {"role": "user", "content": msg},
        ]
        prompt_tok = estimate_messages(prompt_msgs)
        num_ctx = choose_num_ctx(
            model=self.config.coder_model,
            prompt_tokens=prompt_tok,
            max_new_tokens=_CODE_PREDICT,
            vram_ceiling=self.config.vram_ctx_ceiling,
        )
        num_ctx = min(num_ctx, model_max_ctx(self.config.coder_model))

        self._auto_compress_if_needed(self._coder_history, self.config.coder_model, num_ctx)

        yield ("log", json.dumps({
            "type": "model_input", "role": "coder",
            "model": self.config.coder_model,
            "num_ctx": num_ctx, "prompt_tokens_est": prompt_tok,
        }), "log")

        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.coder_model,
                messages=prompt_msgs,
                temperature=self.config.coder_temp,
                num_predict=_CODE_PREDICT,
                num_ctx=num_ctx,
                think=False,
            ):
                if kind == "usage":
                    self._record_usage(token)
                    yield ("usage", token, "usage")
                elif kind != "think":
                    full += token
                    yield ("coder", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Coder interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Coder failed: {exc}", "error")

        yield ("log", json.dumps({"type": "model_output", "role": "coder",
                                   "chars": len(full)}), "log")

        self._coder_history.append({"role": "user",      "content": msg})
        self._coder_history.append({"role": "assistant", "content": full})
        return full

    def _run_verifier(self, msg: str) -> Generator:
        prompt_msgs = [
            {"role": "system", "content": PLANNER_VERIFY_SYSTEM},
            {"role": "user",   "content": msg},
        ]
        prompt_tok = estimate_messages(prompt_msgs)
        num_ctx = choose_num_ctx(
            model=self.config.planner_model,
            prompt_tokens=prompt_tok,
            max_new_tokens=_VERIFY_PREDICT,
            vram_ceiling=self.config.vram_ctx_ceiling,
        )
        num_ctx = min(num_ctx, model_max_ctx(self.config.planner_model))

        yield ("log", json.dumps({
            "type": "model_input", "role": "verifier",
            "model": self.config.planner_model, "num_ctx": num_ctx,
        }), "log")

        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.planner_model,
                messages=prompt_msgs,
                temperature=0.15,
                num_predict=_VERIFY_PREDICT,
                num_ctx=num_ctx,
                keep_alive="0",
                think=False,
            ):
                if kind == "usage":
                    self._record_usage(token)
                elif kind != "think":
                    full += token
                    yield ("verify", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Verifier interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Verifier failed: {exc}", "error")
            full = "NEEDS_REVISION: verifier error — treat as unverified"

        yield ("log", json.dumps({"type": "model_output", "role": "verifier",
                                   "chars": len(full)}), "log")
        return full

    # ── build/test runner ─────────────────────────────────────────────────────

    def _run_build_test(self) -> tuple[bool, str, str] | None:
        cmd = self.config.test_cmd or self.config.build_cmd
        if not cmd:
            return None
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=120,
                cwd=str(self._indexer.root),
            )
            output = (result.stdout or "") + (result.stderr or "")
            return (result.returncode == 0, output, cmd)
        except Exception as exc:
            return (False, str(exc), cmd)

    # ── rollback ──────────────────────────────────────────────────────────────

    def _rollback(
        self, snapshots: dict[str, str | None], project_root: Path
    ) -> None:
        """Restore files to their state before this run (called on UNSOLVABLE)."""
        for rel_path, original_content in snapshots.items():
            target = (project_root / rel_path).resolve()
            try:
                target.relative_to(project_root.resolve())
                if original_content is None:
                    if target.exists():
                        target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(original_content, encoding="utf-8")
            except (ValueError, OSError):
                pass

    # ── session persistence ────────────────────────────────────────────────────

    def _save_iteration_state(self, iteration: int, plan: str, feedback: str) -> None:
        state_dir = _project_dir()
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "iteration": iteration,
                "plan": plan[:2000],
                "last_feedback": feedback[:2000],
                "pending_files": [p for p, _ in self.pending_file_changes],
                "session_usage": self._session_usage,
            }
            (state_dir / "session.json").write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _record_solution(
        self,
        files_changed: list[str],
        iteration: int,
        task: str,
        plan: str,
        coder_output: str,
    ) -> None:
        if not files_changed:
            return
        # Semantic memory extraction via compressor
        mem_entries = self._compressor.generate_memory(task, plan, coder_output, files_changed)
        self._memory.add_from_compress(mem_entries, tier="project")
        # Also record the basic solution fact
        self._memory.add(MemoryEntry(
            category="solution",
            content=f"Solved in {iteration} iteration(s). Files: {', '.join(files_changed[:6])}",
            tags=["verified"],
        ), "project")

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
