"""Kratos agent — intelligent pipeline with intent routing, context building, and memory.

Token-stream convention (unchanged from v1):
  All generators yield ``(token: str, kind: str)`` where kind is "think" or "text".

process() yields ``(source: str, content: str, kind: str)`` where source is:
  "router"   routing decision info           (kind="info")
  "header"   section header — "planner"|"coder"|"verify"  (kind="header")
  "planner"  planner token stream            (kind="think"|"text")
  "coder"    coder token stream              (kind="think"|"text")
  "verify"   verifier token stream           (kind="think"|"text")
  "tool"     tool call display               (kind="tool")
  "direct"   direct answer without LLM      (kind="text")
  "info"     informational message           (kind="info")
  "warn"     warning                         (kind="warn")
  "error"    error                           (kind="error")
  "question" clarification request           (kind="question")
  "end"      section end — "planner"|"coder"|"verify"  (kind="end")

Pipeline for coding tasks:
  Planner → Coder → Verifier (planner in verify mode)
  If verifier says NEEDS_REVISION → re-plan → re-code → re-verify (up to max iterations)
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
from .config import KratosConfig, _project_dir, GLOBAL_DIR
from .context import ContextBuilder, ContextPackage, ProjectIndexer, ScopeType
from .memory import MemoryEntry, MemoryManager
from .router import Route, Router

# ── system prompts ────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are Kratos Planner. Analyze the task and produce a clear, practical plan.

Answer in plain text:
- What exactly needs to be done?
- Which files are relevant or need to change?
- Step-by-step: what should the coder implement?
- Risks or things to watch out for?

Rules:
- Do NOT write code or modify files yourself.
- If test files are provided, read them FIRST — they define the exact API (function signatures,
  return types, exception classes) that must be implemented. Reference them explicitly.
- If a file is provided in context, read it and reference it by name.
- If the task is a question, just answer it directly.
- Be concise. Skip anything that does not apply.
- Flag any circular import risks: if module A imports module B which imports module A, that is broken."""

PLANNER_VERIFY_SYSTEM = """\
You are Kratos Verifier. Review whether the coder's implementation correctly fulfills the task.

Check:
1. Does it implement ALL requirements from the task?
2. Is the code complete — no placeholders, no "...", no missing parts?
3. Are there obvious bugs, syntax errors, or logical mistakes?
4. Does it follow the plan?

Respond ONLY with one of these three formats:

VERIFIED
(the implementation is complete and correct — nothing missing)

NEEDS_REVISION:
<specific list of what is missing or wrong — reference exact file names and line numbers>
(changes are required but the task is still solvable)

UNSOLVABLE:
<explanation of why the task cannot be completed>
(ONLY when: requirements are contradictory, necessary external dependencies are missing,
or the codebase fundamentally cannot support what is requested)

Do not suggest enhancements beyond what is required."""

CODER_SYSTEM = """\
You are Kratos Coder. Output ONLY file changes in the exact format below. Zero prose allowed.

OUTPUT RULES:
1. Your very first characters must be "### FILE:" — no greeting, no explanation, nothing before it.
2. Write EVERY source file listed in the plan that still contains stubs or NotImplementedError.
3. Write COMPLETE file content — every function fully implemented, no "...", no pass-with-TODO.
4. Do NOT create files not in the plan (no README.md, no docs, no extra helpers).
5. Do NOT rewrite a file that is already complete unless the plan explicitly says to fix it.

CIRCULAR IMPORT PREVENTION (violation crashes the whole program):
6. Before every import ask: does that module already import from THIS module?
   - models.py → NEVER import from storage.py (storage.py already imports models)
   - cli.py → NEVER import from models.py or storage.py at module level if they import cli
   - If you need an exception class from another module, DEFINE IT locally instead of importing it.
7. ValidationError MUST be defined inside models.py — never imported from another file.
8. StorageError MUST be defined inside storage.py — never imported from another file.

API CONTRACT RULES (return types must match tests exactly):
9.  ranking()  → returns List[Tuple[Player, int]]  — NOT (rank, name, score), NOT (id, name, score)
10. stats()    → returns dict {"players": int, "events": int, "total_score": int, "leader": str}
11. from_dict() on a class → MUST be a @classmethod that creates and returns a new instance
12. validate_delta(delta) → raises ValidationError (NOT ValueError) when invalid
13. normalize_player_name(name) → strips whitespace; raises ValidationError if blank or len > 40
14. normalize_reason(reason) → returns "manual" for None or blank; raises ValidationError if len > 60
15. add_player(name) → validates, creates Player, appends to self.players, returns the Player
16. add_score(player_id, delta, reason) → validates player exists; raises ValidationError if not found

FORMAT (use exactly):
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
Changed: file1.py, file2.py — one line of what was done"""

# ── token limits ──────────────────────────────────────────────────────────────

_PLAN_NUM_PREDICT        = 2048    # concise plan
_PLAN_NUM_PREDICT_HEAVY  = 4096    # architecture / diagnostic: more room for complex reasoning
_CODE_NUM_PREDICT        = 16384   # 3 files × ~150 lines each needs headroom

# ── output parsers ───────────────────────────────────────────────────────────

_FILE_CHANGE_RE = re.compile(
    r'###\s+FILE:\s*(.+?)\s*\n```(?:\w+)?\n(.*?)```',
    re.S,
)
_FILE_DELETE_RE = re.compile(r'###\s+DELETE:\s*(.+?)\s*$', re.M)


def _parse_file_changes(text: str) -> list[tuple[str, str]]:
    """Extract (rel_path, content) pairs from ### FILE: blocks."""
    return [(m.group(1).strip(), m.group(2)) for m in _FILE_CHANGE_RE.finditer(text)]


def _parse_file_deletions(text: str) -> list[str]:
    """Extract rel_paths from ### DELETE: lines."""
    return [m.group(1).strip() for m in _FILE_DELETE_RE.finditer(text)]


# ── scope / token-budget selection ───────────────────────────────────────────

def _scope_for(route: Route, intent: Intent) -> ScopeType:
    """Scope for the PLANNER context (always broad)."""
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
        return "architecture"   # planner sees everything
    return "minimal"


def _planner_needs_thinking(scope: ScopeType, route: Route) -> bool:
    """Use CoT only when the task is genuinely complex — saves VRAM time on simple fixes."""
    return scope in ("architecture", "diagnostic") or route == Route.DIAGNOSTIC_LOOP


def _coder_scope_for(intent: Intent) -> ScopeType:
    """Scope for the CODER context — enough files to see all stubs AND test files."""
    if intent in (Intent.FOLLOWUP,):
        return "patch_context"
    return "expanded"   # max 12 files — coder needs stubs + test files for API spec


# ── message builders ──────────────────────────────────────────────────────────

def _planner_msg(task: str, ctx: ContextPackage, all_files: list | None = None) -> str:
    parts: list[str] = []
    if ctx.project_description:
        parts.append(f"Project: {ctx.project_description}")
    # Always list ALL project files so planner knows what exists
    if all_files:
        listing = "\n".join(f"  {e.rel_path}" for e in all_files[:50])
        parts.append(f"All project files:\n{listing}")
    if ctx.memory_summary:
        parts.append(ctx.memory_summary)
    if ctx.files:
        parts.append("File contents (most relevant):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(80)}")
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
        # Test files define the exact API — show them first
        test_files = [f for f in ctx.files
                      if any(x in f.rel_path for x in ("test_", "_test.", ".spec.", ".test."))]
        src_files  = [f for f in ctx.files
                      if f not in test_files and not any(x in f.rel_path for x in noise)]
        stub_files = [f for f in src_files if "NotImplementedError" in (f.content or "")]
        done_files = [f for f in src_files if f not in stub_files]

        if test_files:
            parts.append(
                "TEST FILES — these define the exact API you must implement.\n"
                "Match every import, function signature, and return type exactly:"
            )
            for f in test_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(80)}")
        if stub_files:
            names = ", ".join(f.rel_path for f in stub_files)
            parts.append(f"STUB FILES — IMPLEMENT ALL OF THESE: {names}")
            for f in stub_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(60)}")
        if done_files:
            parts.append(
                "Already-implemented files (reference only — do NOT rewrite unless the plan says to):"
            )
            for f in done_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(30)}")
    if ctx.error_lines:
        parts.append("Errors:\n" + "\n".join(ctx.error_lines[:10]))
    parts.append(f"Task: {task}")
    return "\n\n".join(p for p in parts if p)


def _planner_retry_msg(task: str, prev_plan: str, verify_feedback: str,
                        ctx: "ContextPackage | None" = None) -> str:
    parts = [f"Task: {task}"]
    if ctx and ctx.files:
        parts.append("Current file state (updated since last iteration):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(40)}")
    parts.append(f"Previous plan:\n{prev_plan[:800]}")
    parts.append(
        f"Test / verifier feedback — what still needs to be fixed:\n{verify_feedback[:3000]}\n\n"
        "Produce a precise plan for what the coder must still implement to make all tests pass. "
        "List each file and each function that still needs to be written. "
        "If there is a circular import error, name the exact import line to remove."
    )
    return "\n\n".join(parts)


def _coder_retry_msg(task: str, ctx: "ContextPackage", plan: str, verify_feedback: str) -> str:
    parts: list[str] = []
    if ctx.files:
        noise = ("README", "ARCH", "REQUIRE", "docs/", "CHANGELOG", "LICENSE")
        test_files = [f for f in ctx.files
                      if any(x in f.rel_path for x in ("test_", "_test.", ".spec.", ".test."))]
        src_files  = [f for f in ctx.files
                      if f not in test_files and not any(x in f.rel_path for x in noise)]
        if test_files:
            parts.append(
                "TEST FILES — exact API you must satisfy (check imports and return types):"
            )
            for f in test_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(60)}")
        parts.append("Current source file state:")
        for f in src_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(80)}")
    parts.append(f"Revised plan:\n{plan}")
    parts.append(
        f"Test failures / required fixes:\n{verify_feedback[:3000]}\n\n"
        "Fix ALL failing tests. Do NOT skip any file — every NotImplementedError must be replaced."
    )
    parts.append(f"Task: {task}")
    return "\n\n".join(p for p in parts if p)


def _verify_msg(task: str, plan: str, coder_output: str) -> str:
    return (
        f"Original task: {task}\n\n"
        f"Plan that was given to the coder:\n{plan[:800]}\n\n"
        f"Coder's output:\n{coder_output[:2500]}"
    )


# ── clarification message ─────────────────────────────────────────────────────

def _clarification_msg(analysis, intent: Intent) -> str:
    kw_str = ", ".join(analysis.keywords[:6]) if analysis.keywords else "none detected"
    return (
        f"Your request is unclear. Detected keywords: [{kw_str}]. "
        f"Could you specify: what should be changed, which file(s), and what the expected result is?"
    )


# ── direct answer (no LLM) ────────────────────────────────────────────────────

def _direct_file_search(
    indexer: ProjectIndexer, keywords: list[str], file_paths: list[str]
) -> str:
    results: list[str] = []
    index = indexer.index
    if not index:
        return "No project files indexed."

    # Exact path matches first
    for fp in file_paths:
        fp_l = fp.lower().replace("\\", "/")
        for e in index:
            if fp_l in e.rel_path.lower():
                results.append(f"  {e.rel_path}  [pri={e.priority}]")

    # Keyword matches in filename
    for kw in keywords[:5]:
        for e in index:
            line = f"  {e.rel_path}  [pri={e.priority}]"
            if kw.lower() in e.rel_path.lower() and line not in results:
                results.append(line)

    if not results:
        # Fall back to listing top files
        top = [f"  {e.rel_path}" for e in index[:20]]
        return "No direct matches. Top project files:\n" + "\n".join(top)

    return "Matching files:\n" + "\n".join(results[:20])


def _direct_code_search(
    indexer: ProjectIndexer, keywords: list[str]
) -> str:
    lines: list[str] = []
    for kw in keywords[:3]:
        hits = indexer.search_content(kw, max_results=8)
        if hits:
            lines.append(f"\nResults for '{kw}':")
            for entry, lineno, text in hits:
                lines.append(f"  {entry.rel_path}:{lineno}  {text}")
    return "\n".join(lines) if lines else "No matches found in project files."


# ── history compression ───────────────────────────────────────────────────────

def _compress_history(history: list[dict], max_pairs: int = 5) -> None:
    """Compress history: summarize dropped pairs into a context note instead of silently removing them.

    When the window overflows, the oldest N pairs are replaced by a compact
    "[Compressed context: ...]" prefix on the next remaining message so the
    model still knows what was tried and what errors occurred.
    """
    if len(history) <= max_pairs * 2:
        return

    surplus = len(history) - max_pairs * 2
    # Surplus must be even (pairs); round up to keep pairs intact
    if surplus % 2:
        surplus += 1
    surplus = min(surplus, len(history) - 2)  # always keep at least 1 pair

    # Extract useful facts from the pairs about to be dropped
    facts: list[str] = []
    for i in range(0, surplus, 2):
        if i + 1 >= len(history):
            break
        user_content = history[i].get("content", "")
        asst_content = history[i + 1].get("content", "")
        # Files that were written in this pair
        files_written = re.findall(r'###\s+FILE:\s*(\S+)', asst_content)
        if files_written:
            facts.append("wrote " + ", ".join(files_written[:3]))
        # Errors / failures from the user turn (test feedback)
        err_lines = [
            ln.strip()
            for ln in user_content.splitlines()
            if any(x in ln.lower() for x in ("error", "fail", "importerror", "except", "circular"))
        ]
        if err_lines:
            facts.append("issue: " + err_lines[0][:120])

    # Drop the oldest pairs
    del history[:surplus]

    # Prepend compact summary to the first remaining user message
    if facts and history:
        summary = "[Prior context compressed: " + " | ".join(facts[:6]) + "]\n\n"
        history[0] = {**history[0], "content": summary + history[0]["content"]}


# ── main agent ────────────────────────────────────────────────────────────────

class KratosAgent:
    def __init__(self, config: KratosConfig, bridge: OllamaBridge) -> None:
        self.config = config
        self.bridge = bridge

        # Legacy per-model histories (kept for backward compat with plan/code/run)
        self._planner_history: list[dict] = []
        self._coder_history: list[dict] = []

        # Pipeline components
        self._analyzer = InputAnalyzer()
        self._classifier = IntentClassifier()
        self._router = Router()

        # Project root = CWD at invocation time (wherever `kratos` was called from)
        project_root = Path.cwd()
        self._indexer = ProjectIndexer(project_root)
        self._ctx_builder = ContextBuilder(self._indexer)

        self._memory = MemoryManager(_project_dir(), GLOBAL_DIR)

        # File operations extracted from last coder run — applied by caller
        self.pending_file_changes: list[tuple[str, str]] = []   # (rel_path, content)
        self.pending_file_deletions: list[str] = []              # rel_paths to delete

    # ── intelligent pipeline ─────────────────────────────────────────────────

    def process(self, task: str) -> Generator[tuple[str, str, str], None, None]:
        """Full intelligent pipeline. Yields (source, content, kind) events.

        Coding pipeline:
          Planner → Coder → Verifier → if NEEDS_REVISION → re-plan → re-code → re-verify
          (up to config.build_test_retries iterations)
        """
        self.pending_file_changes.clear()
        self.pending_file_deletions.clear()

        analysis = self._analyzer.analyze(task)
        intent   = self._classifier.classify(analysis)
        route    = self._router.route(intent)

        yield ("router", f"intent={intent.value}  route={route.value}", "info")

        # ── memory retrieval ──────────────────────────────────────────────────
        mem_entries    = self._memory.get_relevant(analysis.keywords,
                             categories=["solution", "file", "decision", "convention"])
        memory_summary = self._memory.format_for_prompt(mem_entries)

        # ── context build ─────────────────────────────────────────────────────
        scope     = _scope_for(route, intent)
        n_indexed = len(self._indexer.index)
        yield ("tool", f"index_project({self._indexer.root.name!r}) → {n_indexed} files", "tool")

        ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary=memory_summary, scope=scope,
            token_budget=self.config.planner_num_ctx,
        )
        # Smaller context for coder: only the files it needs to change
        coder_scope = _coder_scope_for(intent)
        coder_ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary="", scope=coder_scope,
            token_budget=self.config.coder_num_ctx,
        )
        for f in ctx.files:
            sz = f"{f.size / 1024:.1f} KB" if f.size > 0 else "n/a"
            yield ("tool", f"read_file({f.rel_path!r}) → {sz}", "tool")

        # ── log: full index + full context package ────────────────────────────
        yield ("log", json.dumps({
            "type": "index_project",
            "project": self._indexer.root.name,
            "file_count": n_indexed,
            "files": [{"rel_path": e.rel_path, "size": e.size, "priority": e.priority}
                      for e in self._indexer.index],
        }), "log")
        yield ("log", json.dumps({
            "type": "context_package",
            "intent": intent.value, "route": route.value, "scope": scope,
            "memory_summary": memory_summary,
            "files_loaded": [{"rel_path": f.rel_path, "size": f.size,
                               "excerpt_lines": len((f.content or "").splitlines())}
                             for f in ctx.files],
            "full_context_prompt": ctx.to_prompt(),
        }), "log")

        # ── terminal routes (no coding loop) ──────────────────────────────────
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
                route,
                keep_alive="5m",   # keep loaded for likely follow-up
                scope=scope,
            )
            yield ("end", "planner", "end")
            self._memory.extract_and_store(plan_full, intent.value, analysis.file_paths)
            return

        # ── plan → code → verify loop ─────────────────────────────────────────
        # Loop until VERIFIED, UNSOLVABLE, or hard safety cap
        max_iter        = self.config.max_verify_iterations   # default 10
        plan_text       = ""
        verify_feedback = ""
        needs_plan      = route in (Route.PLANNER_THEN_CODER, Route.DIAGNOSTIC_LOOP)
        # Accumulated file changes across all iterations (rel_path → content)
        _accumulated_changes: dict[str, str] = {}
        _accumulated_deletions: set[str] = set()

        for attempt in range(max_iter):
            is_retry = attempt > 0
            iter_label = f"iteration {attempt + 1}/{max_iter}"

            if is_retry:
                yield ("info", f"Revising — {iter_label}", "info")
                # Rebuild context so planner/coder sees files written in previous iterations
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

            # ── 1. Planner ────────────────────────────────────────────────────
            if needs_plan or is_retry:
                yield ("header", "planner", "header")
                if is_retry:
                    planner_input = _planner_retry_msg(task, plan_text, verify_feedback, ctx=ctx)
                else:
                    planner_input = _planner_msg(task, ctx, all_files=self._indexer.index)

                # keep_alive="0" → planner unloads immediately so coder gets full VRAM
                plan_full = yield from self._run_planner(
                    planner_input, route, keep_alive="0", scope=scope
                )
                yield ("end", "planner", "end")
                plan_text = plan_full

            # ── 2. Coder — uses smaller context, think=False for direct output ──
            yield ("header", "coder", "header")
            if is_retry:
                coder_input = _coder_retry_msg(task, coder_ctx, plan_text, verify_feedback)
            else:
                coder_input = _coder_msg(task, coder_ctx, plan=plan_text)

            coder_full = yield from self._run_coder(coder_input)
            yield ("end", "coder", "end")

            # Accumulate ops: merge this iteration's output with previous iterations.
            # Later iterations override earlier ones for the same file path.
            new_changes   = _parse_file_changes(coder_full)
            new_deletions = _parse_file_deletions(coder_full)
            for rel_path, content in new_changes:
                _accumulated_changes[rel_path] = content
            _accumulated_deletions.update(new_deletions)
            self.pending_file_changes   = list(_accumulated_changes.items())
            self.pending_file_deletions = list(_accumulated_deletions)
            self._memory.extract_and_store(
                coder_full, intent.value, [c[0] for c in self.pending_file_changes]
            )

            # Auto-save iteration state to .kratos/ for crash recovery / resumability
            self._save_iteration_state(attempt + 1, plan_text, verify_feedback)

            # Apply accumulated changes to disk so the verifier / next planner
            # can see what was already written.
            project_root = self._indexer.root
            for rel_path, content in self.pending_file_changes:
                target = (project_root / rel_path).resolve()
                try:
                    target.relative_to(project_root.resolve())
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                except (ValueError, OSError):
                    pass
            for rel_path in self.pending_file_deletions:
                target = (project_root / rel_path).resolve()
                try:
                    target.relative_to(project_root.resolve())
                    if target.exists():
                        target.unlink()
                except (ValueError, OSError):
                    pass

            # ── 3a. Real test runner (preferred when test_cmd is configured) ──
            if self.config.test_cmd:
                build_result = self._run_build_test()
                if build_result is not None:
                    ok, output, cmd = build_result
                    yield ("tool", f"run_command({cmd!r}) → exit={'0' if ok else '1'}", "tool")
                    if ok:
                        yield ("info", "All tests passed — task complete.", "info")
                        yield ("log", json.dumps({"type": "verify_decision",
                                                  "decision": "VERIFIED",
                                                  "feedback": "",
                                                  "iteration": attempt + 1}), "log")
                        self._record_solution(
                            [p for p, _ in self.pending_file_changes], attempt + 1
                        )
                        break
                    else:
                        verify_feedback = (
                            f"Tests failed (iteration {attempt + 1}).\n"
                            f"Fix ALL remaining failures before this is done.\n\n"
                            f"{output[-3000:]}"
                        )
                        yield ("warn", f"Tests failed — {output[-200:]}", "warn")
                        yield ("log", json.dumps({"type": "verify_decision",
                                                  "decision": "NEEDS_REVISION",
                                                  "feedback": verify_feedback,
                                                  "iteration": attempt + 1}), "log")
                        self._record_failure_pattern(verify_feedback, attempt + 1)
                        if attempt < max_iter - 1:
                            continue  # skip LLM verifier, use test output directly
                        else:
                            yield ("warn", f"Safety cap reached ({max_iter} iterations). Final state may need review.", "warn")
                            break

            # Optional build/test for DIAGNOSTIC_LOOP (no test_cmd path)
            if route == Route.DIAGNOSTIC_LOOP and not self.config.test_cmd:
                build_result = self._run_build_test()
                if build_result is not None:
                    ok, output, cmd = build_result
                    yield ("tool", f"run_command({cmd!r}) → exit={'0' if ok else '1'}", "tool")
                    if not ok:
                        yield ("warn", f"Build/test failed:\n{output[-400:]}", "warn")
                        verify_feedback = f"Build/test failed:\n{output[-800:]}"
                        continue  # skip verifier, go straight to next iteration

            # ── 3b. LLM Verifier (fallback when no test_cmd) ─────────────────
            yield ("header", "verify", "header")
            verify_full = yield from self._run_verifier(
                _verify_msg(task, plan_text, coder_full)
            )
            yield ("end", "verify", "end")

            vf_upper = verify_full.upper()

            if "UNSOLVABLE" in vf_upper:
                reason = verify_full.replace("UNSOLVABLE:", "").replace("UNSOLVABLE", "").strip()
                yield ("warn", f"Verifier: task cannot be solved — {reason[:300]}", "warn")
                yield ("log", json.dumps({"type": "verify_decision",
                                          "decision": "UNSOLVABLE",
                                          "feedback": verify_full,
                                          "iteration": attempt + 1}), "log")
                break

            if "VERIFIED" in vf_upper and "NEEDS_REVISION" not in vf_upper:
                yield ("log", json.dumps({"type": "verify_decision",
                                          "decision": "VERIFIED",
                                          "feedback": "",
                                          "iteration": attempt + 1}), "log")
                self._record_solution(
                    [p for p, _ in self.pending_file_changes], attempt + 1
                )
                if attempt > 0:
                    yield ("info", f"Verified after {attempt + 1} iteration(s).", "info")
                break

            # NEEDS_REVISION
            verify_feedback = verify_full.replace("NEEDS_REVISION:", "").strip()
            yield ("log", json.dumps({"type": "verify_decision",
                                      "decision": "NEEDS_REVISION",
                                      "feedback": verify_full,
                                      "iteration": attempt + 1}), "log")
            if attempt < max_iter - 1:
                yield ("warn", f"Needs revision ({attempt + 1}/{max_iter}) — {verify_feedback[:200]}", "warn")
            else:
                yield ("warn", f"Safety cap reached ({max_iter} iterations). Final state may need review.", "warn")

    # ── model runners (internal) ──────────────────────────────────────────────

    def _run_planner(self, msg: str, route: Route,
                     keep_alive: str = "0",
                     scope: ScopeType = "targeted") -> Generator:
        """Stream planner, yield events + log events, return full text.

        keep_alive="0" (default) unloads the model immediately so the coder
        gets full VRAM on the next call. Pass keep_alive="5m" for
        planner-only routes where a follow-up is likely.
        """
        # Thinking is scope-driven: complex scopes (architecture/diagnostic) get full CoT;
        # targeted scope uses think=False so the model outputs the plan immediately.
        needs_thinking = _planner_needs_thinking(scope, route)
        planner_think: bool | None = None if needs_thinking else False

        yield ("log", json.dumps({
            "type": "model_input",
            "role": "planner",
            "model": self.config.planner_model,
            "system_prompt": PLANNER_SYSTEM,
            "message": msg,
        }), "log")

        thinking = ""
        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.planner_model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM},
                    *self._planner_history,
                    {"role": "user", "content": msg},
                ],
                temperature=self.config.planner_temp,
                num_predict=_PLAN_NUM_PREDICT_HEAVY if needs_thinking else _PLAN_NUM_PREDICT,
                num_ctx=self.config.planner_num_ctx,
                keep_alive=keep_alive,
                think=planner_think,
            ):
                if kind == "think":
                    thinking += token
                else:
                    full += token
                yield ("planner", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Planner interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Planner failed: {exc}", "error")

        yield ("log", json.dumps({"type": "model_thinking", "role": "planner",
                                   "text": thinking}), "log")
        yield ("log", json.dumps({"type": "model_output", "role": "planner",
                                   "model": self.config.planner_model,
                                   "text": full}), "log")

        self._planner_history.append({"role": "user", "content": msg})
        self._planner_history.append({"role": "assistant", "content": full})
        _compress_history(self._planner_history, max_pairs=4)
        return full

    def _run_coder(self, msg: str) -> Generator:
        """Stream coder, yield events + log events, return full text.

        Coder uses think=False: its job is to implement the plan, not re-reason about it.
        This prevents 4B models from looping in CoT when they should be writing code.
        """
        forced_msg = (
            "CRITICAL REQUIREMENT: Your response MUST begin with '### FILE:' on the very first line.\n"
            "Do NOT write README.md, documentation, or any explanation — only Python source files.\n"
            "Do NOT add imports that create circular dependencies (see system prompt rules 6-8).\n\n"
            + msg
        )

        yield ("log", json.dumps({
            "type": "model_input",
            "role": "coder",
            "model": self.config.coder_model,
            "system_prompt": CODER_SYSTEM,
            "message": forced_msg,
        }), "log")

        thinking = ""
        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.coder_model,
                messages=[
                    {"role": "system", "content": CODER_SYSTEM},
                    *self._coder_history,
                    {"role": "user", "content": forced_msg},
                ],
                temperature=self.config.coder_temp,
                num_predict=_CODE_NUM_PREDICT,
                num_ctx=self.config.coder_num_ctx,
                think=False,
            ):
                if kind == "think":
                    thinking += token
                else:
                    full += token
                yield ("coder", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Coder interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Coder failed: {exc}", "error")

        yield ("log", json.dumps({"type": "model_thinking", "role": "coder",
                                   "text": thinking}), "log")
        yield ("log", json.dumps({"type": "model_output", "role": "coder",
                                   "model": self.config.coder_model,
                                   "text": full}), "log")

        self._coder_history.append({"role": "user", "content": msg})
        self._coder_history.append({"role": "assistant", "content": full})
        _compress_history(self._coder_history, max_pairs=4)
        return full

    def _run_verifier(self, msg: str) -> Generator:
        """Stream verifier (planner in verify mode), yield events + log events."""
        yield ("log", json.dumps({
            "type": "model_input",
            "role": "verifier",
            "model": self.config.planner_model,
            "system_prompt": PLANNER_VERIFY_SYSTEM,
            "message": msg,
        }), "log")

        thinking = ""
        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.planner_model,
                messages=[
                    {"role": "system", "content": PLANNER_VERIFY_SYSTEM},
                    {"role": "user", "content": msg},
                ],
                temperature=0.2,
                num_predict=512,
                num_ctx=self.config.planner_num_ctx,
                keep_alive="0",
                think=False,
            ):
                if kind == "think":
                    thinking += token
                else:
                    full += token
                yield ("verify", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Verifier interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Verifier failed: {exc}", "error")
            full = "NEEDS_REVISION: verifier model error — treat as unverified"

        yield ("log", json.dumps({"type": "model_thinking", "role": "verifier",
                                   "text": thinking}), "log")
        yield ("log", json.dumps({"type": "model_output", "role": "verifier",
                                   "model": self.config.planner_model,
                                   "text": full}), "log")
        return full

    # ── build/test runner ─────────────────────────────────────────────────────

    def _run_build_test(self) -> tuple[bool, str, str] | None:
        """Returns (ok, output, cmd) or None if no command configured."""
        cmd = self.config.test_cmd or self.config.build_cmd
        if not cmd:
            return None
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                cwd=str(self._indexer.root),
            )
            output = (result.stdout or "") + (result.stderr or "")
            return (result.returncode == 0, output, cmd)
        except Exception as exc:
            return (False, str(exc), cmd)

    # ── session persistence (.kratos/) ───────────────────────────────────────

    def _save_iteration_state(self, iteration: int, plan: str, feedback: str) -> None:
        """Persist iteration state to .kratos/session.json for crash recovery."""
        state_dir = _project_dir()
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "iteration": iteration,
                "plan": plan[:2000],
                "last_feedback": feedback[:2000],
                "pending_files": [p for p, _ in self.pending_file_changes],
            }
            (state_dir / "session.json").write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _record_solution(self, files_changed: list[str], iteration: int) -> None:
        """Save successful solution pattern to project memory."""
        if not files_changed:
            return
        entry = MemoryEntry(
            category="solution",
            content=f"Solved in {iteration} iteration(s). Files: {', '.join(files_changed[:6])}",
            tags=["verified"],
        )
        self._memory.add(entry, "project")

    def _record_failure_pattern(self, feedback: str, iteration: int) -> None:
        """Save recurring error patterns to project memory so future planners avoid them."""
        patterns = [
            ("circular import", "circular_import"),
            ("importerror", "import_error"),
            ("notimplementederror", "stub_not_implemented"),
            ("attributeerror", "attribute_error"),
            ("from_dict", "from_dict_api"),
        ]
        fb_lower = feedback.lower()
        for keyword, tag in patterns:
            if keyword in fb_lower:
                first_relevant = next(
                    (ln.strip() for ln in feedback.splitlines()
                     if keyword in ln.lower() and len(ln.strip()) > 10),
                    keyword,
                )
                entry = MemoryEntry(
                    category="error_cause",
                    content=f"Iteration {iteration} failed — {first_relevant[:200]}",
                    tags=[tag],
                )
                self._memory.add(entry, "project")
                break   # one pattern per failure is enough

    # ── legacy API (v1 compatibility) ─────────────────────────────────────────

    def plan(self, task: str) -> Generator[tuple[str, str], None, None]:
        self._planner_history.append({"role": "user", "content": task})
        full_text = ""
        for token, kind in self.bridge.chat(
            model=self.config.planner_model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                *self._planner_history,
            ],
            temperature=self.config.planner_temp,
            num_predict=_PLAN_NUM_PREDICT,
            num_ctx=self.config.planner_num_ctx,
        ):
            if kind == "text":
                full_text += token
            yield (token, kind)
        self._planner_history.append({"role": "assistant", "content": full_text})

    def code(self, task: str, plan: str | None = None) -> Generator[tuple[str, str], None, None]:
        content = f"Task:\n{task}\n\nExecution plan:\n{plan}\n\nImplement this now." if plan else task
        self._coder_history.append({"role": "user", "content": content})
        full_text = ""
        for token, kind in self.bridge.chat(
            model=self.config.coder_model,
            messages=[
                {"role": "system", "content": CODER_SYSTEM},
                *self._coder_history,
            ],
            temperature=self.config.coder_temp,
            num_predict=_CODE_NUM_PREDICT,
            num_ctx=self.config.coder_num_ctx,
        ):
            if kind == "text":
                full_text += token
            yield (token, kind)
        self._coder_history.append({"role": "assistant", "content": full_text})

    def run(self, task: str) -> Generator[tuple[str, str, str], None, None]:
        plan_text = ""
        for token, kind in self.plan(task):
            if kind == "text":
                plan_text += token
            yield ("planner", token, kind)
        for token, kind in self.code(task, plan=plan_text):
            yield ("coder", token, kind)

    def clear_history(self) -> None:
        self._planner_history.clear()
        self._coder_history.clear()
        self._memory.clear_task()
        self._memory.clear_session()

    # ── project indexer access ────────────────────────────────────────────────

    def rebuild_index(self) -> int:
        self._indexer.invalidate()
        return len(self._indexer.build_index())

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def indexer(self) -> ProjectIndexer:
        return self._indexer
