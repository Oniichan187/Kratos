"""Coder-role message builders.

Phase 1: keeps the existing one-shot/stepwise message shaping (relocated
verbatim from builders.py). Phase 2 adds the adaptive ReAct action-loop here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from ..context import ContextPackage
from ..execution.tools import (
    do_command,
    do_delete,
    do_inspect,
    do_read,
    do_write,
    format_observation,
    parse_actions,
)
from ..planning import (
    ExecutionPlan,
    parse_execution_plan,
    plan_all_done,
    refresh_plan_status,
    render_checklist,
    render_plan_status,
)
from ..verification import CommandRegistry, ProvenWork
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


def _fresh_knowledge_block(agent, query: str, iteration: int) -> Generator[tuple, None, str]:
    if getattr(agent, "_knowledge", None) is None or not query.strip():
        return ""
    try:
        yield ("info", f"retrieving (coder loop {iteration}) from project vector KB...", "info")
        chunks = agent._knowledge.retrieve(
            query[:2500],
            top_k=getattr(agent.config, "retrieval_top_k", 12),
        )
        yield ("tool", f"knowledge_get(coder loop {iteration}) -> {len(chunks)} chunks", "tool")
    except Exception as exc:
        yield ("warn", f"coder-loop retrieval failed: {exc}", "warn")
        return ""
    if not chunks:
        return ""
    blocks = ["FRESH RELEVANT CODE (retrieved after the latest observation):"]
    for ch in chunks[:8]:
        try:
            blocks.append(ch.to_prompt_block(900))
        except Exception:
            blocks.append(f"### {getattr(ch, 'rel_path', '?')}\n{getattr(ch, 'text', '')[:800]}")
    return "\n\n".join(blocks)


def _observation_query(task: str, plan_text: str, observations: list[dict]) -> str:
    failed = [
        obs for obs in observations
        if obs.get("kind") == "command" and not obs.get("skipped") and not obs.get("ok")
    ]
    if failed:
        obs = failed[-1]
        return "\n".join([
            task,
            f"Failed command: {obs.get('cmd')}",
            str(obs.get("output", "")),
        ])

    failed_inspect = [
        obs for obs in observations
        if obs.get("kind") == "inspect" and not obs.get("skipped") and not obs.get("ok")
    ]
    if failed_inspect:
        obs = failed_inspect[-1]
        return "\n".join([
            task,
            f"Failed inspect: {obs.get('cmd')}",
            str(obs.get("output", "")),
        ])

    skipped = [obs for obs in observations if obs.get("kind") == "command" and obs.get("skipped")]
    if skipped:
        obs = skipped[-1]
        return "\n".join([task, f"Skipped command: {obs.get('cmd')}", str(obs.get("detail", ""))])

    skipped_inspect = [obs for obs in observations if obs.get("kind") == "inspect" and obs.get("skipped")]
    if skipped_inspect:
        obs = skipped_inspect[-1]
        return "\n".join([task, f"Skipped inspect: {obs.get('cmd')}", str(obs.get("detail", ""))])

    reads = [obs for obs in observations if obs.get("kind") == "read" and obs.get("ok")]
    if reads:
        obs = reads[-1]
        return "\n".join([task, f"Read file: {obs.get('path')}", str(obs.get("content", ""))[:1200]])

    changed = [
        str(obs.get("path")) for obs in observations
        if obs.get("kind") in {"file", "delete"} and obs.get("ok")
    ]
    if changed:
        return "\n".join([task, "Changed files: " + ", ".join(changed), plan_text[:1200]])

    return "\n".join([task, plan_text[:1200]])


def _last_test_passed(proof: ProvenWork) -> bool:
    test_cmds = [c for c in proof.commands if c.get("is_test")]
    return bool(test_cmds and test_cmds[-1].get("exit_code") == 0)


def _sync_pending(agent, changes: dict[str, str], deletions: set[str]) -> None:
    agent.pending_file_changes = list(changes.items())
    agent.pending_file_deletions = list(deletions)


def run_coder_loop(
    agent,
    task: str,
    plan,
    analysis,
    intent,
    route,
    ctx: ContextPackage,
    cmd_registry: CommandRegistry,
    proof: ProvenWork,
    attempt: int,
    verify_feedback: str,
    project_root: Path,
    original_snapshots: dict | None = None,
) -> Generator[tuple, None, tuple[str, dict[str, str], set[str]]]:
    """Adaptive OBSERVE -> ACT coder loop.

    The loop writes/reads/deletes/runs only through the existing guarded runtime
    helpers and returns accumulated file state for the unchanged outer
    PROVEN_WORK/verifier/retry block.
    """
    from ..prompts import load_prompts

    del analysis, intent, route  # kept in the public call shape for future prompt refinements

    pm = load_prompts()
    original_snapshots = original_snapshots if original_snapshots is not None else {}
    accumulated_changes: dict[str, str] = dict(agent.pending_file_changes)
    accumulated_deletions: set[str] = set(agent.pending_file_deletions)
    coder_transcript: list[str] = []

    plan_state = plan if isinstance(plan, ExecutionPlan) else parse_execution_plan(str(plan or ""))
    plan_text = plan_state.markdown

    ctx_block = _coder_context_block(ctx, pm, step_mode=False)
    cmd_block = cmd_registry.format_for_prompt()
    forced = pm.get_snippet("coder_loop_forced_prefix") or ""
    protocol = pm.get_snippet("coder_action_protocol") or ""

    parts = [
        forced + protocol,
        f"{pm.get_snippet('task_label') or 'Task: '}{task}",
    ]
    if plan_text:
        parts.append(
            "PLAN GUIDANCE (use as context, not as a rigid per-step driver):\n"
            f"{plan_text}"
        )
    if plan_state.items:
        parts.append(
            "PLAN CHECKLIST (the loop must keep going until all items are done):\n"
            f"{render_checklist(plan_state.items, compact=False)}"
        )
    if verify_feedback:
        parts.append(
            f"{pm.get_snippet('verify_feedback_intro') or 'Verifier / test feedback:'}"
            f"{verify_feedback[:2000]}"
        )
    if cmd_block:
        parts.append(cmd_block)
    inspect_hint = pm.get_snippet("inspect_hint")
    if inspect_hint:
        parts.append(inspect_hint)
    if ctx_block:
        parts.append("PROJECT FILES / CURRENT CONTEXT:\n" + ctx_block)
    if getattr(ctx, "error_lines", None):
        parts.append((pm.get_snippet("errors_short_header") or "Errors:\n") + "\n".join(ctx.error_lines[:10]))

    turn = "\n\n".join(p for p in parts if p)
    max_iterations = max(1, int(getattr(agent.config, "max_coder_iterations", 6) or 6))

    for loop_idx in range(max_iterations):
        loop_num = loop_idx + 1
        if agent._is_cancelled():
            yield ("warn", "Run cancelled before coder loop iteration.", "warn")
            break

        yield ("header", "coder", "header")
        coder_out = yield from agent._run_coder(turn)
        yield ("end", "coder", "end")
        coder_transcript.append(f"CODER LOOP ITERATION {loop_num}\n{coder_out}")

        if agent._is_cancelled():
            yield ("warn", "Run cancelled before applying coder output.", "warn")
            break

        actions = parse_actions(coder_out, pm)
        observations: list[dict] = []
        touched_files: list[str] = []

        for rel_path in actions["reads"]:
            observations.append((yield from do_read(project_root, rel_path)))

        for cmd in actions["inspects"]:
            observations.append((yield from do_inspect(agent, project_root, cmd)))

        for rel_path, content in actions["files"]:
            if not agent.config.can_write():
                yield ("warn", f"write_file({rel_path!r}) -> skipped (write permission disabled)", "warn")
                observations.append({
                    "kind": "file", "path": rel_path, "ok": False,
                    "detail": "write permission disabled",
                })
                continue
            obs = yield from do_write(project_root, rel_path, content, proof, attempt, original_snapshots)
            observations.append(obs)
            if obs.get("ok"):
                accumulated_changes[rel_path] = content
                accumulated_deletions.discard(rel_path)
                touched_files.append(rel_path)

        for rel_path in actions["deletes"]:
            if not agent.config.can_delete():
                yield ("warn", f"delete_file({rel_path!r}) -> skipped (delete permission disabled)", "warn")
                observations.append({
                    "kind": "delete", "path": rel_path, "ok": False,
                    "detail": "delete permission disabled",
                })
                continue
            obs = yield from do_delete(project_root, rel_path, proof, attempt, original_snapshots)
            observations.append(obs)
            if obs.get("ok"):
                accumulated_changes.pop(rel_path, None)
                accumulated_deletions.add(rel_path)
                touched_files.append(rel_path)

        _sync_pending(agent, accumulated_changes, accumulated_deletions)
        if touched_files:
            agent._memory.track_files(touched_files)

        for cmd in actions["commands"]:
            observations.append((yield from do_command(agent, project_root, cmd_registry, cmd, proof)))

        refresh_plan_status(plan_state, proof, touched_files)
        plan_block = render_plan_status(plan_state.items)

        if actions["done"]:
            if _last_test_passed(proof) and plan_all_done(plan_state):
                yield ("info", f"Coder loop converged after {loop_num} iteration(s).", "info")
                break
            observations.append({
                "kind": "done",
                "ok": False,
                "detail": "plan items remain unfinished" if not plan_all_done(plan_state) else "no passing test command yet",
            })

        if not any((actions["reads"], actions["files"], actions["deletes"], actions["commands"], actions["done"])):
                observations.append({
                    "kind": "note",
                    "detail": (
                        "No valid action markers were found. Use ### READ, ### INSPECT, ### FILE, "
                        "### DELETE, ### VERIFY/### RUN, or ### DONE."
                    ),
                })

        observation_text = format_observation(observations, pm)
        observation_text += "\n\n" + plan_block
        knowledge_query = _observation_query(task, plan_text, observations)
        knowledge_block = yield from _fresh_knowledge_block(agent, knowledge_query, loop_num)
        if knowledge_block:
            observation_text += "\n\n" + knowledge_block
        if not (_last_test_passed(proof) and plan_all_done(plan_state)):
            observation_text += "\n\n" + (
                pm.get_snippet("observation_continue") or
                "Continue the loop: react to the observation above, fix failures, then re-verify."
            )

        coder_transcript.append(observation_text)
        turn = observation_text
    else:
        yield ("warn", f"Coder loop reached max_coder_iterations={max_iterations}; continuing to final verifier.", "warn")

    _sync_pending(agent, accumulated_changes, accumulated_deletions)
    return ("\n\n".join(coder_transcript), accumulated_changes, accumulated_deletions)
