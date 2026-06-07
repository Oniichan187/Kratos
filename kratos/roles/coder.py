"""Coder-role message builders.

Phase 1: keeps the existing one-shot/stepwise message shaping (relocated
verbatim from builders.py). Phase 2 adds the adaptive ReAct action-loop here.
"""

from __future__ import annotations

from ..context import ContextPackage
from .prompts import _coder_context_block


def _coder_msg(task: str, ctx: ContextPackage, plan: str) -> str:
    from ..prompts import load_prompts
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


def _coder_retry_msg(
    task: str, ctx: ContextPackage, plan: str, verify_feedback: str
) -> str:
    from ..prompts import load_prompts
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
