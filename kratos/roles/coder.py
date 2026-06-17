"""Coder-role message builders.

Phase 1: keeps the existing one-shot/stepwise message shaping (relocated
verbatim from builders.py). Phase 2 adds the adaptive ReAct action-loop here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from ..context import ContextPackage
from ..execution.tools import (
    do_command,
    do_delete,
    do_edit,
    do_glob,
    do_grep,
    do_inspect,
    do_read,
    do_read_range,
    do_search,
    do_web_fetch,
    do_web_search,
    do_write,
    format_observation,
    has_any_action,
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
from ..verification import (
    CommandRegistry,
    ProvenWork,
    VerificationCommand,
    _clean_command_line,
    _is_safe_verification_command,
    _is_test_verification_command,
    _missing_command_paths,
)
from .prompts import _coder_context_block


# Additional lookup/research markers available to the coder loop (runtime-side
# docs so even a user-customized prompts JSON keeps these tools discoverable).
_EXTRA_MARKERS_DOC = (
    "ADDITIONAL LOOKUP ACTIONS (read-only, instant):\n"
    "  ### SEARCH: <pattern> [:: <glob>]   - smart search: literal text, a|b alternation,\n"
    "                                        regex, all case-insensitive. Returns file:line:col hits.\n"
    "                                        Example: ### SEARCH: Feldkirch|weather-card :: **/*.html\n"
    "  ### GREP: <regex> [:: <glob>]       - regex-first search across project files\n"
    "  ### GLOB: <pattern>                 - find files by name, e.g. ### GLOB: **/*.test.ts\n"
    "  ### READ_RANGE: <path>:<start>-<end> - read exact line range, e.g. ### READ_RANGE: src/app.py:40-90\n"
    "  ### WEB_SEARCH: <query>             - web search (docs/API/error research); cite sources\n"
    "  ### WEB_FETCH: <url>                - fetch one documentation page as text\n"
    "SEARCH wants a concrete pattern, NOT a sentence describing what you want to find.\n"
    "Use SEARCH/GREP before editing unfamiliar files. Use WEB_* only for documentation or error research."
)

# When the task explicitly asks for web research/scraping, weak/abliterated
# models often skip it and invent sources. This forces a REAL tool call.
_WEB_TASK_RE = re.compile(
    r"\b(web[- ]?such|websuche|web[- ]?scrap|webscrap|internet[- ]?recherche|recherche|"
    r"web\s*search|web\s*scrap|research|quellen|sources|https?://)\b", re.I,
)


def _web_research_hint(task: str) -> str:
    """Strong, explicit web instruction injected only when the task needs it."""
    if not _WEB_TASK_RE.search(task or ""):
        return ""
    return (
        "\nWEB RESEARCH REQUIRED: this task involves the web. You MUST actually call the "
        "tools — emit `### WEB_SEARCH: <query>` and/or `### WEB_FETCH: <url>` and use the "
        "returned text. Do NOT invent or guess URLs/sources: only real fetched pages count, "
        "and the final report lists exactly the sources the runtime fetched. If a fetch fails "
        "or no provider is configured, say so honestly instead of fabricating links.\n"
    )


def _coder_msg(task: str, ctx: ContextPackage, plan: str) -> str:
    from ..prompts import load_prompts
    pm = load_prompts()
    parts: list[str] = []
    if plan:
        plan_label = pm.get_snippet('plan_label') or 'Plan:\n'
        parts.append(f"{plan_label}{plan}")
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
    revised_plan_label = pm.get_snippet('revised_plan_label') or 'Revised plan:\n'
    parts.append(f"{revised_plan_label}{plan}")
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


def _pick_auto_verify_cmd(item, cmd_registry: CommandRegistry, project_root: Path) -> VerificationCommand | None:
    """Choose the verification command the RUNTIME runs itself when the model
    edited files but failed to emit ### VERIFY (the most common weak-model
    failure: edits land on disk, but nothing ever proves them)."""
    cand = _clean_command_line(getattr(item, "verify_cmd", "") or "") if item is not None else ""
    if (cand and _is_safe_verification_command(cand, project_root)
            and not cmd_registry.is_toolchain_mismatch(cand)
            and not _missing_command_paths(cand, project_root)):
        # Nested layouts: planner items say `VERIFY: python -m pytest` without
        # knowing the project lives in a subdir — inherit the discovered cwd,
        # otherwise the command runs in the wrong directory and ALWAYS fails
        # with import errors the model cannot fix (real-run root cause).
        return VerificationCommand(cand, f"auto-verify checklist item {getattr(item, 'index', '?')}",
                                   "auto-verify", _is_test_verification_command(cand),
                                   cwd=cmd_registry.cwd_for(cand))
    tests = cmd_registry.test_commands()
    if tests:
        t = tests[0]
        return VerificationCommand(t.cmd, "auto-verify (project test command)", "auto-verify",
                                   t.is_test, cwd=getattr(t, "cwd", None))
    return None


def _run_auto_verify(
    agent, item, cmd_registry: CommandRegistry, proof: ProvenWork,
    project_root: Path, observations: list[dict],
) -> Generator[tuple, None, None]:
    """Run the auto-chosen verify command, record evidence, append observation."""
    import json as _json
    vcmd = _pick_auto_verify_cmd(item, cmd_registry, project_root)
    if vcmd is None:
        return
    yield ("info", f"auto-verify: model emitted no VERIFY — running `{vcmd.cmd}`", "info")
    yield ("command", vcmd.cmd, "command_echo")
    res = agent._run_verification_command(vcmd)
    proof.commands.append(res)
    proof.commands_planned.append({"cmd": vcmd.cmd, "purpose": vcmd.purpose,
                                   "source": vcmd.source, "is_test": vcmd.is_test})
    status = "ok" if res["exit_code"] == 0 else "FAILED"
    yield ("tool", f"verify_command({vcmd.cmd!r}) -> {status} exit={res['exit_code']} [auto]", "tool")
    yield ("log", _json.dumps({
        "type": "build_test", "cmd": vcmd.cmd, "purpose": vcmd.purpose,
        "source": vcmd.source, "is_test": vcmd.is_test,
        "exit_code": res["exit_code"],
        "duration_seconds": res.get("duration_seconds"),
        "output": res.get("output", ""),
        "stdout": res.get("stdout", ""),
        "stderr": res.get("stderr", ""),
        "cwd": res.get("cwd"),
        "shell": res.get("shell"),
        "timeout_seconds": res.get("timeout_seconds"),
        "timed_out": res.get("timed_out"),
        "blocked": res.get("blocked"),
        "block_reason": res.get("block_reason"),
        "result": res,
    }, ensure_ascii=False), "log")
    obs = {
        "kind": "command", "cmd": vcmd.cmd, "ok": res["exit_code"] == 0,
        "skipped": False, "exit_code": res["exit_code"], "is_test": vcmd.is_test,
        "output": res.get("output", "")[-1500:],
    }
    # Failed auto-verify: turn the raw output into a concrete, actionable
    # diagnosis and register it for stall detection. Without this the weak-model
    # structured loop only ever saw raw tracebacks per item (session 13-12-22:
    # 15 failing runs, the diagnosis fired only once at the outer gate).
    if res["exit_code"] != 0:
        from ..execution.diagnostics import diagnose_command
        diag = diagnose_command(res)
        if diag is not None:
            tracker = getattr(agent, "_repair_tracker", None)
            seen = tracker.register(diag.signature) if tracker is not None else 1
            obs["detail"] = diag.as_feedback()
            if tracker is not None and tracker.is_stalled(diag.signature):
                obs["detail"] += "\n" + tracker.escalation_note(diag.signature)
                yield ("warn",
                       f"Repair stall detected ({diag.category} ×{seen}) — escalating diagnosis.",
                       "warn")
            yield ("log", _json.dumps({
                "type": "failure_diagnosis", "seen": seen,
                "purpose": vcmd.purpose, **diag.to_dict()}, ensure_ascii=False), "log")
    observations.append(obs)


_FAIL_READBACK_MAX_FILES = 2
_FAIL_READBACK_MAX_CHARS = 4000


def _read_disk_excerpt(project_root: Path, rel: str, max_chars: int = _FAIL_READBACK_MAX_CHARS) -> str | None:
    target = (project_root / rel)
    try:
        target.resolve().relative_to(project_root.resolve())
        if not target.is_file():
            return None
        raw = target.read_text("utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    return raw[:max_chars] + ("\n... [truncated]" if len(raw) > max_chars else "")


def _inject_failure_readback(
    project_root: Path, touched: list[str], observations: list[dict],
) -> None:
    """After a FAILED verification, automatically show the model the CURRENT
    on-disk content of the files it just wrote. In real runs the model never
    re-read its own broken file and could not converge — this closes that
    loop deterministically."""
    failed_cmds = [
        o for o in observations
        if o.get("kind") == "command" and not o.get("skipped") and not o.get("ok")
    ]
    if not failed_cmds:
        return
    # Files the failure DIAGNOSIS blames (e.g. the module named in the traceback)
    # are often the ones that actually need the fix — not just what was touched.
    diag_files: list[str] = []
    try:
        from ..execution.diagnostics import diagnose_command
        diag = diagnose_command(failed_cmds[-1])
        if diag is not None:
            diag_files = [f for f in (diag.files or []) if f]
    except Exception:
        diag_files = []
    seen: set[str] = set()
    paths: list[str] = []
    for rel in [*(touched or []), *diag_files]:
        rel = (rel or "").replace("\\", "/")
        if rel and rel not in seen:
            seen.add(rel)
            paths.append(rel)
    if not paths:
        return
    for rel in paths[:_FAIL_READBACK_MAX_FILES]:
        excerpt = _read_disk_excerpt(project_root, rel)
        if excerpt is None:
            continue
        observations.append({
            "kind": "note",
            "detail": (
                f"VERIFICATION FAILED. CURRENT on-disk content of `{rel}` — fix the error "
                f"reported above with a TARGETED ### EDIT (do not rewrite the whole file):\n"
                f"```\n{excerpt}\n```"
            ),
        })


def _lookup_signature(actions: dict) -> str:
    """Stable signature of a turn's read-only lookups (loop detection)."""
    import json as _json
    return _json.dumps(
        {k: actions.get(k, []) for k in ("reads", "ranges", "searches", "greps", "globs", "inspects")},
        sort_keys=True,
    )


def _run_lookup_actions(
    project_root: Path, actions: dict, observations: list[dict],
) -> Generator[tuple, None, None]:
    """Execute all read-only lookup actions (search/grep/glob/read/range/web)
    in a fixed safe order, appending their observations."""
    for arg in actions.get("searches", []):
        observations.append((yield from do_search(project_root, arg)))
    for arg in actions.get("greps", []):
        observations.append((yield from do_grep(project_root, arg)))
    for arg in actions.get("globs", []):
        observations.append((yield from do_glob(project_root, arg)))
    for rel_path in actions.get("reads", []):
        observations.append((yield from do_read(project_root, rel_path)))
    for arg in actions.get("ranges", []):
        observations.append((yield from do_read_range(project_root, arg)))
    for url in actions.get("web_fetches", []):
        observations.append((yield from do_web_fetch(project_root, url)))
    for query in actions.get("web_searches", []):
        observations.append((yield from do_web_search(project_root, query)))


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

    last_plan_compact: str = ""  # for change-aware de-spam of UI events

    ctx_block = _coder_context_block(ctx, pm, step_mode=False)
    cmd_block = cmd_registry.format_for_prompt()
    forced = pm.get_snippet("coder_loop_forced_prefix") or ""
    protocol = pm.get_snippet("coder_action_protocol") or ""

    parts = [
        forced + protocol + "\n" + _EXTRA_MARKERS_DOC + _web_research_hint(task),
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
    raw_iterations = getattr(agent.config, "max_coder_iterations", 6)
    configured_iterations = 6 if raw_iterations is None else int(raw_iterations)
    unbounded = configured_iterations <= 0
    max_iterations = None if unbounded else max(1, configured_iterations)
    loop_idx = 0

    while max_iterations is None or loop_idx < max_iterations:
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

        yield from _run_lookup_actions(project_root, actions, observations)

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

        for rel_path, search, replace in actions.get("edits", []):
            if not agent.config.can_write():
                yield ("warn", f"edit_file({rel_path!r}) -> skipped (write permission disabled)", "warn")
                observations.append({"kind": "edit", "path": rel_path, "ok": False,
                                     "detail": "write permission disabled"})
                continue
            obs = yield from do_edit(project_root, rel_path, search, replace, proof, attempt, original_snapshots)
            observations.append(obs)
            if obs.get("ok") and obs.get("status") != "noop":
                accumulated_changes[rel_path] = obs.get("content", accumulated_changes.get(rel_path, ""))
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

        # After file writes, give the model an explicit nudge (in the *next* obs) to
        # actually issue the Befehl (### VERIFY / RUN) listed in the checklist for those files.
        # This addresses "coder does edits but 'befehl geben nicht viel'" and repetitive
        # re-edits of the same file (e.g. models.py) without verification.
        if touched_files:
            verify_hints = []
            for item in (plan_state.items or []):
                if item.verify_cmd and any(ref in touched_files for ref in (item.file_refs or [])):
                    verify_hints.append(f"  {item.title[:60]} -> use: {item.verify_cmd}")
            if verify_hints:
                observations.append({
                    "kind": "note",
                    "detail": "Checklist items touched by your writes. Emit the matching VERIFY/RUN now (do not re-edit the same file next turn unless the verify shows a *new* failure):\n" + "\n".join(verify_hints),
                })

        for cmd in actions["commands"]:
            observations.append((yield from do_command(agent, project_root, cmd_registry, cmd, proof)))

        _inject_failure_readback(project_root, touched_files, observations)
        refresh_plan_status(plan_state, proof, touched_files)
        plan_block = render_plan_status(plan_state.items)
        plan_compact = render_checklist(plan_state.items, compact=True)

        # Surface live plan/todo status to the outer UI (CLI rich stream + TUI PlanBox)
        # **change-aware** to prevent spam: only yield when the compact view actually changed.
        # We send the compact form for the bottom live stats bar (the "über dem userinput" area).
        # The full plan_block is still appended to the LLM observation_text below (for the model).
        if plan_compact != last_plan_compact:
            last_plan_compact = plan_compact
            # Include a tiny header so handlers know it's the live compact version.
            yield ("plan_status", f"PLAN {sum(1 for i in plan_state.items if i.status=='done')}/{len(plan_state.items)} | {plan_compact}", "plan")

        if actions["done"]:
            if _last_test_passed(proof) and plan_all_done(plan_state):
                # Final (only if it would be new)
                refresh_plan_status(plan_state, proof, touched_files)
                final_compact = render_checklist(plan_state.items, compact=True)
                if final_compact != last_plan_compact:
                    yield ("plan_status", f"PLAN all done | {final_compact}", "plan")
                yield ("info", f"Coder loop converged after {loop_num} iteration(s).", "info")
                break
            observations.append({
                "kind": "done",
                "ok": False,
                "detail": "plan items remain unfinished" if not plan_all_done(plan_state) else "no passing test command yet",
            })

        if not has_any_action(actions):
                observations.append({
                    "kind": "note",
                    "detail": (
                        "No valid action markers were found. Use ### SEARCH, ### GREP, ### GLOB, "
                        "### READ, ### READ_RANGE, ### INSPECT, ### FILE, ### DELETE, "
                        "### VERIFY/### RUN, ### WEB_SEARCH, ### WEB_FETCH, or ### DONE. "
                        "Plain prose is NOT an action — nothing was changed on disk."
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
        loop_idx += 1

    if max_iterations is not None and loop_idx >= max_iterations:
        yield ("warn", f"Coder loop reached max_coder_iterations={max_iterations}; continuing to final verifier.", "warn")

    _sync_pending(agent, accumulated_changes, accumulated_deletions)
    return ("\n\n".join(coder_transcript), accumulated_changes, accumulated_deletions)


def execute_structured_work_steps_for_plan(
    agent,
    task: str,
    plan,
    ctx: ContextPackage,
    cmd_registry: CommandRegistry,
    proof: ProvenWork,
    attempt: int,
    verify_feedback: str,
    project_root: Path,
    original_snapshots: dict | None = None,
) -> Generator[tuple, None, tuple[str, dict[str, str], set[str]]]:
    """
    New structured driver (per the approved plan): autonomously processes the planner
    checklist one item (or one focused work step toward an item) at a time.

    Enforced high-level sequence per work step (model is told to follow it):
      1. Search first (### INSPECT with visible regex/pattern) → runtime reports match count.
      2. Targeted read of the section (runtime reports exact line range in TUI + transcript).
      3. (Fallback to vector DB "get" if search misses.)
      4. Edit (### FILE) → runtime immediately reports file + exact line deltas (+added / -removed).
      5. Verify (### VERIFY / RUN for the item) → tests close the work step and advance the todo.

    This replaces the free multi-iteration ReAct loop for normal planner-driven work.
    The model no longer does arbitrary long back-and-forth inside one item; the driver
    advances the todo list with clear, auditable steps.

    The low-level action execution reuses the existing parse + do_* machinery.
    """
    from ..prompts import load_prompts

    pm = load_prompts()
    original_snapshots = original_snapshots if original_snapshots is not None else {}
    accumulated_changes: dict[str, str] = dict(getattr(agent, "pending_file_changes", []))
    accumulated_deletions: set[str] = set(getattr(agent, "pending_file_deletions", []))
    transcript: list[str] = []

    plan_state = plan if isinstance(plan, ExecutionPlan) else parse_execution_plan(str(plan or ""))
    plan_text = plan_state.markdown

    if not plan_state.items:
        # No checklist — fall back to a single narrow turn (old behavior for simple cases).
        forced = pm.get_snippet("work_step_forced_prefix") or pm.get_snippet("coder_loop_forced_prefix") or ""
        turn = forced + f"Task: {task}\n\nPlan:\n{plan_text}"
        yield ("header", "coder", "header")
        out = yield from agent._run_coder(turn)
        yield ("end", "coder", "end")
        transcript.append(out)
        return ("\n\n".join(transcript), accumulated_changes, accumulated_deletions)

    # Process items one by one (autonomous per-todo). Each item gets MULTIPLE
    # micro-turns: the previous "one model turn per item" implementation was
    # the root cause of empty files_changed runs — a chatty model produced
    # prose without markers, the driver silently moved on, and the run ended
    # with zero edits. Now the driver loops per item, feeds observations back,
    # re-prompts hard when a turn contains no actions, and only abandons an
    # item after max_turns or repeated no-progress turns.
    max_turns_per_item = max(1, int(getattr(agent.config, "max_work_step_turns", 4)))

    # Failure context carried across items: when item N leaves the project
    # broken, item N+1 must see WHY instead of starting blind.
    last_failure: str = (verify_feedback or "").strip()
    # Live plan rendering for the UI (change-aware, like run_coder_loop) —
    # without this the TUI plan box stays frozen at "PLAN 0/N" forever.
    last_plan_compact: str = ""

    for item in plan_state.items:
        if item.status == "done":
            continue

        item_desc = f"Item {item.index}: {item.title}"
        if item.file_refs:
            item_desc += f" (files: {', '.join(item.file_refs)})"
        if item.verify_cmd:
            item_desc += f"  VERIFY: {item.verify_cmd}"

        # Show the model the CURRENT on-disk state of the item's files up
        # front — real runs proved it otherwise rewrites files blind.
        file_context = ""
        for ref in (item.file_refs or [])[:_FAIL_READBACK_MAX_FILES]:
            excerpt = _read_disk_excerpt(project_root, ref)
            if excerpt:
                file_context += f"\nCURRENT CONTENT of {ref} (on disk right now):\n```\n{excerpt}\n```\n"
        carry = (
            f"\nLAST VERIFICATION FAILURE (fix this first if related):\n{last_failure[:1500]}\n"
            if last_failure else ""
        )

        forced = pm.get_snippet("work_step_forced_prefix") or ""
        turn_prompt = (
            forced + _EXTRA_MARKERS_DOC + _web_research_hint(task) + "\n\n" +
            f"Current checklist item to advance with focused work steps:\n{item_desc}\n\n"
            f"Overall task: {task}\n"
            + carry + file_context +
            "\nFollow the work step sequence strictly: search (### SEARCH/### GREP/### GLOB, pattern visible + report hits) "
            "→ read (### READ / ### READ_RANGE path:start-end — you will receive the exact line range) "
            "→ edit (### FILE — will be reported with file + +/- lines) "
            "→ verify (### VERIFY with the item's test command). "
            "Plain prose without markers does NOTHING. Emit at least one marker per turn."
        )

        no_action_turns = 0
        item_done = False
        recent_lookup_sigs: list[str] = []   # window of recent lookup-only turns
        repeated_lookups = 0
        consecutive_lookup_turns = 0  # flail guard: any lookup-only turns, varied or not

        for turn_idx in range(max_turns_per_item):
            if agent._is_cancelled():
                break

            yield ("header", "coder", "header")
            coder_out = yield from agent._run_coder(turn_prompt)
            yield ("end", "coder", "end")
            transcript.append(f"WORK STEP turn {turn_idx + 1} for {item_desc}\n{coder_out}")

            if agent._is_cancelled():
                break

            actions = parse_actions(coder_out, pm)
            observations: list[dict] = []
            touched: list[str] = []

            yield from _run_lookup_actions(project_root, actions, observations)

            for cmd in actions["inspects"]:
                observations.append((yield from do_inspect(agent, project_root, cmd)))

            for rel, content in actions["files"]:
                if not agent.config.can_write():
                    yield ("warn", f"write_file({rel!r}) -> skipped (permission)", "warn")
                    continue
                obs = yield from do_write(project_root, rel, content, proof, attempt, original_snapshots)
                observations.append(obs)
                if obs.get("ok"):
                    accumulated_changes[rel] = content
                    accumulated_deletions.discard(rel)
                    touched.append(rel)

            for rel, search, replace in actions.get("edits", []):
                if not agent.config.can_write():
                    yield ("warn", f"edit_file({rel!r}) -> skipped (permission)", "warn")
                    continue
                obs = yield from do_edit(project_root, rel, search, replace, proof, attempt, original_snapshots)
                observations.append(obs)
                if obs.get("ok") and obs.get("status") != "noop":
                    accumulated_changes[rel] = obs.get("content", accumulated_changes.get(rel, ""))
                    accumulated_deletions.discard(rel)
                    touched.append(rel)
            for rel in actions["deletes"]:
                if not agent.config.can_delete():
                    continue
                obs = yield from do_delete(project_root, rel, proof, attempt, original_snapshots)
                observations.append(obs)
                if obs.get("ok"):
                    accumulated_changes.pop(rel, None)
                    accumulated_deletions.add(rel)
                    touched.append(rel)

            _sync_pending(agent, accumulated_changes, accumulated_deletions)
            if touched:
                agent._memory.track_files(touched)

            for cmd in actions["commands"]:
                observations.append((yield from do_command(agent, project_root, cmd_registry, cmd, proof)))

            # AUTO-VERIFY: files were edited but the model ran no command —
            # the runtime proves the edit itself instead of waiting for a
            # ### VERIFY that weak models often never emit.
            ran_real_command = any(
                o.get("kind") == "command" and not o.get("skipped") for o in observations
            )
            if touched and not ran_real_command:
                yield from _run_auto_verify(agent, item, cmd_registry, proof, project_root, observations)

            # After a FAILED verification: show the model its own broken file
            # and remember the failure for the next item's prompt.
            _inject_failure_readback(project_root, touched, observations)
            for o in observations:
                if o.get("kind") == "command" and not o.get("skipped"):
                    if o.get("ok"):
                        last_failure = ""
                    else:
                        last_failure = f"`{o.get('cmd')}` -> exit={o.get('exit_code')}\n{o.get('output', '')}"

            # LOOP DETECTION: lookup-only turns without any write/delete/command.
            # Two layers of detection:
            #   1. Identical-signature guard — the model ran the exact same lookups
            #      again without editing (A,B,A,B pattern).
            #   2. Consecutive-lookup guard — even when the model varies its searches
            #      (e.g. mathx.add() → mathx.py → **/*.py → web → is_prime), any
            #      2+ consecutive lookup-only turns without a write means it is
            #      "flailing" and needs a hard directive to write the code now.
            # Note: edits (### EDIT) count as writes — include them in the "not
            # lookup-only" check so an EDIT turn correctly resets the counters.
            is_lookup_only = not (
                actions["files"] or actions.get("edits", []) or actions["deletes"]
                or actions["commands"] or actions["done"]
            )
            sig = _lookup_signature(actions) if is_lookup_only else ""
            if is_lookup_only and sig and sig in recent_lookup_sigs:
                repeated_lookups += 1
                observations.append({
                    "kind": "note",
                    "detail": (
                        "LOOP DETECTED: you already ran this exact lookup in a previous turn "
                        "without editing afterwards. You have all the information above. "
                        "Emit ### FILE: <path> with the COMPLETE fixed file content NOW. "
                        "Do not search or read again."
                    ),
                })
            if is_lookup_only:
                consecutive_lookup_turns += 1
                if consecutive_lookup_turns >= 2:
                    observations.append({
                        "kind": "note",
                        "detail": (
                            f"LOOKUP FLAIL: {consecutive_lookup_turns} consecutive turns of "
                            "searching/reading without writing. The file content is already "
                            "shown above. You MUST now emit ### FILE: <path> with the COMPLETE "
                            "corrected content. Do NOT search, read, or use ### WEB_SEARCH "
                            "again — write the implementation now."
                        ),
                    })
            else:
                repeated_lookups = 0
                consecutive_lookup_turns = 0
                recent_lookup_sigs.clear()
            if is_lookup_only and sig:
                recent_lookup_sigs.append(sig)
                del recent_lookup_sigs[:-4]   # keep the last 4 lookup turns

            # Hard re-prompt when the model produced prose instead of actions.
            if not has_any_action(actions):
                no_action_turns += 1
                observations.append({
                    "kind": "note",
                    "detail": (
                        "NOTHING HAPPENED: your last answer contained no action marker, so no "
                        "file was changed and no command ran. You MUST emit markers "
                        "(### SEARCH / ### READ / ### READ_RANGE / ### FILE / ### VERIFY ...). "
                        "Do not describe the change — make it."
                    ),
                })
                if no_action_turns >= 2:
                    yield ("warn",
                           f"Item {item.index}: {no_action_turns} turns without any action — "
                           "moving to next item (will be retried by the outer loop).", "warn")
                    transcript.append(f"WORK STEP ABORTED (no actions) for {item_desc}")
                    break

            # Update plan progress for this item + push the LIVE status to the
            # UI (CLI bottom bar / TUI plan box). Change-aware to avoid spam.
            refresh_plan_status(plan_state, proof, touched)
            plan_block = render_plan_status(plan_state.items)
            plan_compact = render_checklist(plan_state.items, compact=True)
            if plan_compact != last_plan_compact:
                last_plan_compact = plan_compact
                done_n = sum(1 for i in plan_state.items if i.status == "done")
                yield ("plan_status", f"PLAN {done_n}/{len(plan_state.items)} | {plan_compact}", "plan")

            obs_text = format_observation(observations, pm)
            obs_text += "\n\n" + plan_block
            transcript.append("WORK STEP OBSERVATION:\n" + obs_text)

            if any(i.status == "done" for i in plan_state.items if i.index == item.index):
                yield ("info", f"Checklist item {item.index} completed after {turn_idx + 1} work-step turn(s).", "info")
                item_done = True
                break

            # Feed the observation back for the next micro-turn on this item.
            turn_prompt = (
                obs_text
                + "\n\nContinue working on the SAME checklist item:\n" + item_desc
                + "\nReact to the observation above. If the edit is done, run the verify command. "
                  "Emit only action markers."
            )

        if agent._is_cancelled():
            break
        if not item_done and item.status != "done":
            yield ("warn", f"Item {item.index} not verifiably finished after {max_turns_per_item} turn(s).", "warn")

    _sync_pending(agent, accumulated_changes, accumulated_deletions)
    return ("\n\n".join(transcript), accumulated_changes, accumulated_deletions)
