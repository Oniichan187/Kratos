"""Scope selection, coder context-block rendering, and search/clarification helpers.

Shared building blocks used across the planner/coder/verifier message builders
and the terminal "direct answer" routes.
"""

from __future__ import annotations

from ..classifier import Intent
from ..context import ContextPackage, ProjectIndexer, ScopeType
from ..knowledge import RetrievedChunk  # for type hints / rendering
from ..router import Route


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


def _clarification_msg(analysis, intent: Intent) -> str:
    from ..prompts import load_prompts
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
