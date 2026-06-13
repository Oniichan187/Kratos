"""Planner-role message builders.

The planner turns a task + context package into a step-by-step plan that
guides the coder. These builders shape the user-turn text sent to it.
"""

from __future__ import annotations

from ..context import ContextPackage
from ..knowledge import RetrievedChunk


def _planner_msg(task: str, ctx: ContextPackage, all_files: list | None = None, verify_hint: str = "") -> str:
    from ..prompts import load_prompts
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


def _planner_retry_msg(
    task: str, prev_plan: str, verify_feedback: str, ctx: ContextPackage | None = None
) -> str:
    from ..prompts import load_prompts
    pm = load_prompts()
    parts = [f"{pm.get_snippet('task_label') or 'Task: '}{task}"]
    if ctx and ctx.files:
        parts.append(pm.get_snippet("current_file_state_header") or "Current file state (updated since last iteration):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(800)}")
    prev_plan_label = pm.get_snippet('previous_plan_label') or 'Previous plan:\n'
    parts.append(f"{prev_plan_label}{prev_plan[:600]}")
    fb_intro = pm.get_snippet("verify_feedback_intro") or "Verifier / test feedback — what still needs to be fixed:\n"
    fb_action = pm.get_snippet("verify_feedback_action") or (
        "Produce a precise plan for what the coder must implement to fix all issues. "
        "List each file and function. For circular imports name the exact import line to remove."
    )
    parts.append(f"{fb_intro}{verify_feedback[:2000]}\n\n{fb_action}")
    return "\n\n".join(parts)
