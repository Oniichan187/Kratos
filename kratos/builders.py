"""Message builders and scope selectors for the Kratos agent pipeline."""

from __future__ import annotations

import json

from .classifier import Intent
from .context import ContextPackage, ProjectIndexer, ScopeType
from .knowledge import RetrievedChunk  # for type hints / rendering
from .prompts import load_prompts
from .router import Route
from .verification import ProvenWork


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


def _needs_thinking(
    task: str, scope: ScopeType, route: Route, is_retry: bool, n_files: int
) -> bool:
    """CoT is disabled — the 8b model takes 20+ minutes with think=True on a 6 GB laptop
    and causes Ollama server timeouts.  Context-rich prompts + PROVEN_WORK feedback give
    the planner all the signal it needs without chain-of-thought."""
    return False


def _coder_context_block(ctx: ContextPackage, pm, step_mode: bool = False) -> str:
    """Build the file-context section used in coder prompts.

    Prefers rich retrieved chunks from the vector knowledge base (the "continuous gets").
    Falls back to the old whole-file excerpts only when no chunks are present.
    """
    # NEW best-possible path: use the dynamically retrieved chunks
    if getattr(ctx, "retrieved_chunks", None):
        parts: list[str] = []
        parts.append(pm.get_snippet("file_contents_header") or "Relevant code (dynamically retrieved for this step):")
        for ch in ctx.retrieved_chunks[:8]:
            if isinstance(ch, RetrievedChunk):
                parts.append(ch.to_prompt_block(700))
            else:
                parts.append(str(ch)[:500])
        return "\n\n".join(parts)

    if not ctx.files:
        return ""
    noise = ("README", "ARCH", "REQUIRE", "docs/", "CHANGELOG", "LICENSE")
    test_files = [f for f in ctx.files
                  if any(x in f.rel_path for x in ("test_", "_test.", ".spec.", ".test.",
                                                     "Tests.", "Tests/", "tests/"))]
    src_files  = [f for f in ctx.files
                  if f not in test_files and not any(x in f.rel_path for x in noise)]

    def _is_stub(f) -> bool:
        c = f.content or ""
        if "NotImplementedError" in c:
            return True
        if step_mode and ("// TODO" in c or "/* TODO" in c):
            return True
        return False

    stub_files = [f for f in src_files if _is_stub(f)]
    done_files = [f for f in src_files if f not in stub_files]

    parts: list[str] = []
    if test_files:
        parts.append(pm.get_snippet("test_files_header") or
                     "TEST FILES — these define the exact API. Match every signature exactly:")
        for f in test_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(7000)}")
    if stub_files:
        names = ", ".join(f.rel_path for f in stub_files)
        parts.append((pm.get_snippet("stub_files_header") or "STUB FILES — IMPLEMENT ALL: ") + names)
        for f in stub_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(5000)}")
    if done_files:
        parts.append(pm.get_snippet("done_files_header") or
                     "Already-implemented (reference only — do NOT rewrite unless plan says to):")
        for f in done_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(3000)}")
    return "\n\n".join(p for p in parts if p)


def _planner_msg(task: str, ctx: ContextPackage, all_files: list | None = None, verify_hint: str = "") -> str:
    pm = load_prompts()
    parts: list[str] = []
    if ctx.project_description:
        parts.append(ctx.project_description)
    if all_files and not ctx.project_description:
        listing = "\n".join(f"  {e.rel_path}" for e in all_files[:100])
        parts.append((pm.get_snippet("all_project_files_header") or "All project files:\n") + listing)
    if ctx.memory_summary:
        parts.append(ctx.memory_summary)

    # NEW: prefer rich retrieved chunks from the vector knowledge base (continuous "gets")
    if getattr(ctx, "retrieved_chunks", None):
        parts.append("## Relevant Code (dynamically retrieved via project vector knowledge base)")
        for ch in ctx.retrieved_chunks[:10]:
            if isinstance(ch, RetrievedChunk):
                parts.append(ch.to_prompt_block(800))
            else:
                parts.append(str(ch)[:600])
    elif ctx.files:
        parts.append(pm.get_snippet("file_contents_header") or "File contents (most relevant):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1500)}")

    if ctx.error_lines:
        parts.append((pm.get_snippet("errors_header") or "Errors/logs:\n") + "\n".join(ctx.error_lines[:15]))
    if verify_hint:
        parts.append(verify_hint)
    parts.append(f"{pm.get_snippet('task_label') or 'Task: '}{task}")
    return "\n\n".join(p for p in parts if p)


def _coder_msg(task: str, ctx: ContextPackage, plan: str) -> str:
    pm = load_prompts()
    parts: list[str] = []
    if plan:
        parts.append(f"{pm.get_snippet('plan_label') or 'Plan:\\n'}{plan}")
    ctx_block = _coder_context_block(ctx, pm, step_mode=True)
    if ctx_block:
        parts.append(ctx_block)
    if ctx.error_lines:
        parts.append((pm.get_snippet("errors_short_header") or "Errors:\n") + "\n".join(ctx.error_lines[:10]))
    parts.append(f"{pm.get_snippet('task_label') or 'Task: '}{task}")
    return "\n\n".join(p for p in parts if p)


def _planner_retry_msg(
    task: str, prev_plan: str, verify_feedback: str, ctx: ContextPackage | None = None
) -> str:
    pm = load_prompts()
    parts = [f"{pm.get_snippet('task_label') or 'Task: '}{task}"]
    if ctx and ctx.files:
        parts.append(pm.get_snippet("current_file_state_header") or "Current file state (updated since last iteration):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(800)}")
    parts.append(f"{pm.get_snippet('previous_plan_label') or 'Previous plan:\\n'}{prev_plan[:600]}")
    fb_intro = pm.get_snippet("verify_feedback_intro") or "Verifier / test feedback — what still needs to be fixed:\n"
    fb_action = pm.get_snippet("verify_feedback_action") or (
        "Produce a precise plan for what the coder must implement to fix all issues. "
        "List each file and function. For circular imports name the exact import line to remove."
    )
    parts.append(f"{fb_intro}{verify_feedback[:2000]}\n\n{fb_action}")
    return "\n\n".join(parts)


def _coder_retry_msg(
    task: str, ctx: ContextPackage, plan: str, verify_feedback: str
) -> str:
    pm = load_prompts()
    parts: list[str] = []
    if ctx.files:
        noise = ("README", "ARCH", "REQUIRE", "docs/", "CHANGELOG", "LICENSE")
        test_files = [f for f in ctx.files
                      if any(x in f.rel_path for x in (
                          "test_", "_test.", ".spec.", ".test.",
                          "Tests.", "Tests/", "tests/",
                      ))]
        src_files  = [f for f in ctx.files
                      if f not in test_files and not any(x in f.rel_path for x in noise)]
        if test_files:
            parts.append(pm.get_snippet("test_files_header_short") or "TEST FILES — exact API you must satisfy:")
            for f in test_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(3000)}")
        parts.append("Current source files:")
        for f in src_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(5000)}")
    parts.append(f"{pm.get_snippet('revised_plan_label') or 'Revised plan:\\n'}{plan}")
    rf_intro = pm.get_snippet("required_fixes_intro") or "Required fixes:\n"
    rf_action = pm.get_snippet("required_fixes_action") or "Fix ALL issues. Every NotImplementedError must be replaced."
    parts.append(f"{rf_intro}{verify_feedback[:2000]}\n\n{rf_action}")
    if any(f.rel_path.endswith(".cs") for f in ctx.files):
        parts.append(
            "C# compiler repair notes:\n"
            "- Use explicit generic enum parsing, e.g. Enum.TryParse<TaskStatus>(value, ignoreCase: true, out var status).\n"
            "- Enum.Parse<TEnum>(...) returns a value; it is not a TryParse method and has no out parameter.\n"
            "- Hyphenated CLI/file values such as in-progress usually need explicit mapping to enum names like InProgress.\n"
            "- Do not use StringSplitOptions.RemoveEmptyEntries when empty columns are invalid; split first, then validate count and trim.\n"
            "- Convert invalid user/file values into the exception type required by tests; do not leak ArgumentException from Enum.Parse.\n"
            "- Filter blank/comment input lines in repository/file loading before calling a parser that expects a task record.\n"
            "- Compare enum values directly for business logic; do not compare TaskStatus.ToString() to lowercase CLI strings.\n"
            "- Priority order High, Medium, Low means lower rank index sorts first.\n"
            "- After any compiler or test failure, rewrite the complete affected file with the exact broken lines fixed."
        )
    parts.append(f"{pm.get_snippet('task_label') or 'Task: '}{task}")
    return "\n\n".join(p for p in parts if p)


def _verify_msg(task: str, plan: str, coder_output: str, proof: ProvenWork | None = None) -> str:
    pm = load_prompts()
    proof_text = json.dumps(proof.to_dict(), ensure_ascii=False, indent=2) if proof else "{}"
    tl = pm.get_snippet("task_label") or "Task: "
    pl = pm.get_snippet("plan_label") or "Plan given to coder:\n"
    cl = pm.get_snippet("proven_work_label") or "PROVEN_WORK evidence from Kratos runtime:\n"
    return (
        f"{tl}{task}\n\n"
        f"{pl}{plan[:600]}\n\n"
        f"Coder output:\n{coder_output[:2000]}\n\n"
        f"{cl}{proof_text[:4000]}"
    )


def _clarification_msg(analysis, intent: Intent) -> str:
    pm = load_prompts()
    kw_str = ", ".join(analysis.keywords[:6]) if analysis.keywords else "none"
    tmpl = pm.get_snippet("clarification_template") or (
        "Your request is unclear. Detected keywords: [{kw}]. "
        "Please specify: what should change, which file(s), and what the expected result is."
    )
    return tmpl.format(kw=kw_str)


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
