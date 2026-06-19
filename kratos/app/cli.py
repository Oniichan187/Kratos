"""Classic Kratos REPL — the framed-input, streamed-console interaction loop.

This is the primary entry point launched by the root ``kratos.py`` (and via
``kratos.bat`` / ``python kratos.py``). It owns the agent-streaming → console
event mapping (``_stream_agent``), startup checks (``_ensure_ready``), and the
main read-eval-print loop (``main``). The framed-input widget and stream
filters live in ``app/prompt_frame``; the slash-command tree lives in
``app/slash``.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

try:
    from rich.markup import escape
    from rich.live import Live
except ImportError:
    sys.exit("Install dependencies first:  pip install -r requirements.txt")

from kratos.config import KratosConfig, GLOBAL_DIR
from kratos.llm.bridge import OllamaBridge
from kratos.core.agent import KratosAgent
from kratos.commands import handle
from kratos.logger import SessionLogger
from kratos.prompts import load_prompts
from kratos.llm.tokens import role_context_windows
from kratos.ui import (
    console,
    print_banner, print_error, print_info, print_warn, print_success,
    print_usage,
    user_message_panel, task_summary_panel,
    section_banner, elapsed_str,
    status_bar,
)

from .prompt_frame import _BottomPanel, _CoderFilter, _LineFilter, _LiveBar, _PlannerFilter
from .slash import make_completer

_HISTORY_FILE = GLOBAL_DIR / "history.txt"
_COMPLETER = make_completer()


def _ctx_display(config: KratosConfig) -> dict[str, int]:
    d = role_context_windows(config)
    if getattr(config, "always_max_ctx", True):
        d["max_policy"] = 1
    return d


# ── file operation display (agent already wrote files mid-loop) ───────────────

def _show_file_ops(agent: KratosAgent, logger: "SessionLogger | None" = None) -> None:
    """Log the file operations that the agent already applied to disk.

    The agent writes files during the verify loop so verifier and test_cmd see
    the updated state. The live stream already displayed apply/verify events;
    this function only records full file contents for the session log.
    """
    changes   = agent.pending_file_changes
    deletions = agent.pending_file_deletions
    if not changes and not deletions:
        return
    perm = agent.config.permission

    for rel_path, content in changes:
        if not agent.config.can_write():
            print_warn(f"Write blocked (permission={perm}). Use /permission mid or high.")
            break
        if logger:
            target = (agent.indexer.root / rel_path).resolve()
            try:
                previous = target.read_text("utf-8") if target.exists() else None
            except OSError:
                previous = None
            logger.log_file_write(
                rel_path,
                content,
                previous_content=previous,
                sha256=hashlib.sha256(content.encode("utf-8", "replace")).hexdigest(),
                source="pending_file_changes_summary",
            )

    for rel_path in deletions:
        if not agent.config.can_delete():
            print_warn(f"Delete blocked (permission={perm}). Use /permission high.")
            break
        if logger:
            target = (agent.indexer.root / rel_path).resolve()
            try:
                previous = target.read_text("utf-8") if target.exists() else None
                existed = target.exists()
            except OSError:
                previous = None
                existed = None
            logger.log_file_delete(
                rel_path,
                previous_content=previous,
                existed=existed,
                source="pending_file_deletions_summary",
            )


def _stream_agent(
    agent: KratosAgent, task: str, logger: "SessionLogger",
    _ctx_state: "dict | None" = None,
    _active_role: "dict | None" = None,
    _plan_state: "dict | None" = None,  # in/out: caller can pass a dict to receive final compact plan
) -> float:
    """Run the agent pipeline; stream output inline via console.print,
    with a `rich.Live` status bar pinned to the bottom for the duration.

    Single render system (rich) — `Live(console=console)` installs a render
    hook so every `console.print(...)` call automatically prints *above* the
    live area and the bar redraws below; no second UI layer fighting for the
    terminal (that fight — a background prompt_toolkit app + patch_stdout —
    is what previously corrupted the stream and froze the stats; see the
    removed `_LiveStatus`). Returns elapsed seconds.
    """
    windows = role_context_windows(agent.config)
    ctx_live: dict[str, tuple[int, int]] = {
        "planner": (0, windows["planner"]),
        "coder":   (0, windows["coder"]),
        "verifier": (0, windows["verifier"]),
    }
    current_section = ""
    last_banner_section = ""
    section_start = time.monotonic()
    task_start = time.monotonic()
    planner_buf = ""
    coder_buf = ""
    coder_filter = _CoderFilter()
    planner_filter = _PlannerFilter()
    verify_filter = _LineFilter()
    coder_think_filter = _LineFilter(style="dim italic")

    # Live plan/todo state (compact) — updated from planner flush + change-aware coder events.
    # Mirrors the ctx_live pattern so the bottom status bar can show it next to the live P/C/V stats.
    plan_live: dict = {"done": 0, "total": 0, "compact": "", "label": "", "active": ""}
    # Tracks the last checklist we reprinted to the transcript so we refresh the
    # visible todo list (□ -> ☑) when progress advances, without spamming on every
    # micro-event. The one-time "PLAN CHECKLIST" box never updates by itself.
    plan_render_cache: dict = {"done": 0, "text": ""}

    # Running completion-token estimate — Ollama only reports real counts at
    # the *end* of each model call (the "usage"/"ctx_info" events), so without
    # this the bar's ∑ and "%→compose" numbers would sit frozen mid-stream.
    # ~4 chars/token is the standard rough estimate; good enough for a live
    # indicator that the real end-of-call numbers immediately correct.
    running_completion = 0

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    _ROLE_BY_SOURCE = {"planner": "planner", "coder": "coder", "verify": "verifier"}

    def _build_bar():
        role = _ROLE_BY_SOURCE.get(current_section, "")
        state = dict(ctx_live)
        if role and running_completion:
            used, total = state.get(role, (0, 0))
            state[role] = (used + running_completion, total)
        return status_bar(
            scope=agent.config.scope or "project",
            permission=agent.config.permission,
            ctx_state=state,
            elapsed_s=time.monotonic() - task_start,
            project_name=agent._indexer.root.name,
            current_section=role,
            goal=agent.config.goal,
            hint="Ctrl+C to stop",
            plan_state=dict(plan_live),  # compact live todo for the bottom bar (next to ctx stats)
        )

    def _smodel(role: str) -> str:
        if role == "planner": return agent.config.planner_model
        if role == "coder":   return agent.config.coder_model
        if role == "verify":  return agent.config.verifier_model
        return ""

    def _log_tool(c: str) -> None:
        if "(" in c:
            name = c[:c.index("(")]
            args = c[c.index("(") + 1:c.rindex(")")]
            res = c.split("→")[-1].strip() if "→" in c else (c.split("->")[-1].strip() if "->" in c else "")
            logger.log_tool(name=name, args={"raw": args}, result=res)
        else:
            logger.log_tool(name=c, args={})

    live = Live(
        _LiveBar(_build_bar), console=console, refresh_per_second=8,
        transient=True, vertical_overflow="visible",
    )
    live.start()
    try:
        for source, content, kind in agent.process(task):
            logger.log_agent_event(source, content, kind)

            if source == "log":
                if logger.enabled:
                    try:
                        import json as _j
                        d = _j.loads(content); et = d.pop("type", "unknown")
                        logger.log_event(et, **d)
                    except Exception:
                        pass

            elif source == "router":
                console.print(f"  [dim cyan]⟶[/dim cyan]  [dim]{content}[/dim]")
                kv = dict(p.split("=", 1) for p in content.split("  ") if "=" in p)
                logger.log_route(intent=kv.get("intent", ""), route=kv.get("route", ""))

            elif source == "tool":
                # Make the critical parts of a work step (search result, read range, write with deltas) very visible to the human.
                # The model is supposed to do Search (with match count) → Read (lines X-Y) → Write (file + +/- lines) → Verify.
                if content.startswith("write_file("):
                    # Write command feedback must clearly show file + line deltas (user request).
                    console.print(f"  [bold green]↳ WRITE[/bold green]  {content}")
                elif content.startswith("read_file("):
                    console.print(f"  [cyan]↳ READ[/cyan]  {content}")
                elif content.startswith("inspect_result("):
                    console.print(f"  [yellow]↳ INSPECT RESULT[/yellow]  {content}")
                else:
                    console.print(f"  [dim blue]↳[/dim blue]  [dim]{content}[/dim]")
                _log_tool(content)

            elif source == "header":
                running_completion = 0
                current_section = content
                section_start = time.monotonic()
                if _active_role is not None:
                    _active_role["role"] = content
                # Dedup across stepwise re-entries too: print the role banner only when
                # the role changes (e.g. coder -> verify -> coder), not on every step's
                # repeated "coder" header — the "Step N/M" info line is the per-step marker.
                if content != last_banner_section:
                    last_banner_section = content
                    console.print()
                    console.print(section_banner(content, _smodel(content)))

            elif source == "planner":
                running_completion += _est_tokens(content)
                if kind != "think":
                    planner_filter.feed(content)
                    planner_buf += content

            elif source == "verify":
                running_completion += _est_tokens(content)
                style = "dim italic" if kind == "think" else ""
                verify_filter.feed(content, style=style)

            elif source == "coder":
                running_completion += _est_tokens(content)
                if kind == "think":
                    coder_think_filter.feed(content, style="dim italic")
                else:
                    coder_buf += content
                    coder_filter.feed(content)

            elif source == "direct":
                console.print()
                console.print(section_banner("direct", ""))
                console.print(content, highlight=False)

            elif source == "usage":
                try:
                    import json as _j
                    d = _j.loads(content); logger.log_event("token_usage", **d)
                except Exception:
                    pass

            elif source == "end":
                sec_elapsed = time.monotonic() - section_start
                if current_section == "planner":
                    planner_filter.flush()
                    # Feed the live plan state for the bottom bar right after the (one-time) checklist print.
                    try:
                        from kratos.planning import parse_execution_plan, render_checklist, active_checklist_line
                        p = parse_execution_plan(planner_buf or "")
                        if p.items:
                            done = sum(1 for it in p.items if it.status == "done")
                            plan_live["done"] = done
                            plan_live["total"] = len(p.items)
                            plan_live["compact"] = render_checklist(p.items, compact=True)
                            plan_live["label"] = f"PLAN {done}/{len(p.items)}"
                            plan_live["active"] = active_checklist_line(plan_live["compact"])
                    except Exception:
                        pass
                elif current_section == "verify":
                    verify_filter.flush()
                elif current_section == "coder":
                    coder_filter.flush()
                    coder_think_filter.flush()
                console.print()
                console.print(f"  [dim]⏱ {elapsed_str(sec_elapsed)}[/dim]")
                console.print()
                if current_section == "planner" and planner_buf:
                    logger.log_model_output("planner", agent.config.planner_model, planner_buf)
                    planner_buf = ""
                elif current_section == "coder":
                    if coder_buf:
                        logger.log_model_output("coder", agent.config.coder_model, coder_buf)
                    coder_buf = ""
                running_completion = 0
                current_section = ""
                if _active_role is not None:
                    _active_role["role"] = ""
                if _ctx_state is not None:
                    _ctx_state.update(ctx_live)

            elif source == "info":
                console.print(f"  [blue]ℹ[/blue]  [dim]{content}[/dim]")
                logger.log_info(content)

            elif source == "command":
                # Prominent echo of the actual command the coder decided + runtime is running.
                # Complements the model ### VERIFY lines (now printed by _CoderFilter) and the dim tool notes.
                console.print(f"  [bold cyan]$[/bold cyan] {content}")
                logger.log_info(f"command: {content}")

            elif source == "plan_status":
                # Live plan update (change-aware from coder loop or planner). The
                # bottom status bar shows the compact count; in addition we reprint
                # the full checklist to the transcript whenever progress advances,
                # so the visible todo list actually ticks □ -> ☑ (the one-time
                # "PLAN CHECKLIST" box printed by the planner never updates itself).
                if content:
                    # Parse the compact form we now receive: "PLAN d/t | <checklist>"
                    try:
                        label, rest = content.split(" | ", 1)
                        plan_live["label"] = label.strip()
                        plan_live["compact"] = rest.strip()
                        from kratos.planning import active_checklist_line
                        plan_live["active"] = active_checklist_line(rest.strip())
                        # crude done/total from the label if present
                        if "/" in label:
                            nums = label.split()[-1]
                            if "/" in nums:
                                d, t = nums.split("/")
                                plan_live["done"] = int(d)
                                plan_live["total"] = int(t)
                        # Reprint the checklist only when the done-count changed, so
                        # the user watches it progress without per-micro-turn spam.
                        if (plan_live["done"] != plan_render_cache["done"]
                                and rest.strip() and rest.strip() != plan_render_cache["text"]):
                            plan_render_cache["done"] = plan_live["done"]
                            plan_render_cache["text"] = rest.strip()
                            console.print(f"  [bold magenta]{plan_live['label']}[/bold magenta]")
                            for _ln in rest.strip().splitlines():
                                console.print(f"    {_ln}")
                    except Exception:
                        plan_live["compact"] = str(content)[:200]
                # very quiet in transcript (no more full 20-item spam)
                # console.print(f"  [dim]PLAN {plan_live.get('label','')}[/dim]")  # optional one-liner


            elif source == "warn":
                console.print(f"  [yellow]⚠[/yellow]  {content}")

            elif source == "error":
                console.print(f"  [bold red]✗[/bold red]  {content}")
                logger.log_error(content)

            elif source == "ctx_info":
                pc = content.split("|")
                if len(pc) == 3:
                    try:
                        rn, us, ts = pc[0], int(pc[1]), int(pc[2])
                        ctx_live[rn] = (us, ts)
                        if _ctx_state is not None:
                            _ctx_state[rn] = (us, ts)
                    except ValueError:
                        pass

            elif source == "compress":
                console.print(f"  [magenta]⇒ compress[/magenta]  [dim]{content}[/dim]")
                logger.log_info(f"[compress] {content}")

            elif source == "question":
                console.print()
                console.print(f"  [cyan]Kratos:[/cyan] {content}")

            elif source == "report":
                # Evidence-based final report from the Reporter (never free-form
                # model text). Printed verbatim so the user sees real status,
                # real file changes, real command results.
                console.print()
                try:
                    from rich.markdown import Markdown
                    from rich.panel import Panel
                    console.print(Panel(Markdown(content), title="Abschlussbericht (evidenzbasiert)",
                                        border_style="cyan"))
                except Exception:
                    console.print(content)
                logger.log_info(f"final_report:\n{content}")

    except KeyboardInterrupt:
        # Stop the Live render FIRST so console.print() below is not re-entering
        # the rich Live context, which causes a secondary crash on Ctrl+C.
        try:
            live.stop()
        except Exception:
            pass
        console.print()
        console.print("[yellow]⚠[/yellow]  Interrupted.")
    finally:
        planner_filter.flush()
        verify_filter.flush()
        coder_think_filter.flush()
        # Sync final compact plan back to caller so the next idle _BottomPanel toolbar
        # (the info line with the live stats, above the input) can show it.
        if _plan_state is not None:
            _plan_state.update(plan_live)
        # Idempotent — already stopped in the KeyboardInterrupt handler above;
        # a second stop() is harmless but kept here for the non-interrupt path.
        try:
            live.stop()
        except Exception:
            pass

    _show_file_ops(agent, logger)
    console.print()

    total_elapsed = time.monotonic() - task_start
    files = [p for p, _ in agent.pending_file_changes] + agent.pending_file_deletions
    usage = agent.session_usage
    tok = (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    console.print(task_summary_panel(total_elapsed, files, tok))
    console.print()

    return total_elapsed


# ── startup checks ────────────────────────────────────────────────────────────

def _ensure_ready(bridge: OllamaBridge, config: KratosConfig) -> bool:
    print_info("Connecting to Ollama…")
    if bridge.is_running():
        print_success("Ollama ready.")
    else:
        print_info("Starting Ollama…")
        if bridge.start():
            print_success("Ollama started.")
        else:
            print_warn(
                "Cannot reach Ollama. Start it manually:\n"
                "  ollama serve\n"
                "Then re-run Kratos."
            )
            return False

    missing = []
    for model, role in [
        (config.planner_model,    "planner"),
        (config.coder_model,      "coder"),
        (config.verifier_model,   "verifier"),
        (config.compressor_model, "compressor"),
    ]:
        if not bridge.model_exists(model):
            missing.append(f"{model} ({role})")
    if missing:
        print_warn("Missing models: " + ", ".join(missing))
        print_info("Run [cyan]python setup_models.py[/cyan] to install them.")
        return False
    return True


# ── main REPL ─────────────────────────────────────────────────────────────────

def main() -> None:
    config = KratosConfig.load()
    scope  = config.scope or "project"

    bridge = OllamaBridge(config.ollama_host)
    prompts = load_prompts()
    agent  = KratosAgent(config, bridge, prompts=prompts)

    from kratos.config import _project_dir
    project_root = Path.cwd()
    logger = SessionLogger(_project_dir())

    print_banner(
        config.planner_model,
        config.coder_model,
        scope,
        config.permission,
        verifier=config.verifier_model,
        compressor=config.compressor_model,
        ctx=_ctx_display(config),
        show_status_panel=True,
    )
    print_info(f"Project root: [bold]{project_root}[/bold]")
    _ensure_ready(bridge, config)
    console.print()
    print_info("Enter your task, or [cyan]/help[/cyan] for commands.  [dim]/exit[/dim] to quit.")

    _session_start = time.time()

    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

    # Live ctx tracking: updated by _stream_agent as model calls report token counts.
    _windows = role_context_windows(config)
    _ctx_live: dict[str, tuple[int, int]] = {
        "planner": (0, _windows["planner"]),
        "coder":   (0, _windows["coder"]),
        "verifier": (0, _windows["verifier"]),
    }
    _active_role: dict[str, str] = {"role": ""}
    _last_task_s: float | None = None
    _last_plan: dict = {"done": 0, "total": 0, "label": "", "compact": ""}  # compact live todo shown in the idle bottom info line (with the live stats)

    # ── live info content for the panel frame ─────────────────────────────────
    def _make_toolbar() -> list[tuple[str, str]]:
        """Builds prompt_toolkit bottom-toolbar: ⏱ (grows left on m/h) + ctx + project + perm.
        The live context bars are the active signal; no session-total or compose meter.
        """
        SEP = "   │   "
        out: list[tuple[str, str]] = [("", "  ")]

        # Time (Uhr) early in the line: its string growth (s→m→h) makes the clock "move further left"
        # and pushes the following stats right in a controlled way inside the frame.
        _time_val = _last_task_s if _last_task_s is not None else (time.time() - _session_start)
        _t_str = elapsed_str(_time_val) if _time_val and _time_val > 0 else "0s"
        out += [("", "⏱ "), ("ansiyellow", _t_str), ("", SEP)]

        _windows_now = role_context_windows(config)
        _role = _active_role.get("role", "")
        if _role in ("planner", "coder", "verifier"):
            _abbr = {"planner": "P", "coder": "C", "verifier": "V"}[_role]
            _used, _total = _ctx_live.get(_role, (0, _windows_now[_role]))
            out += [("", f"{_abbr} {_used // 1024}k/{_total // 1024}k"), ("", SEP)]
        else:
            for _role, _abbr in (("planner", "P"), ("coder", "C"), ("verifier", "V")):
                _used = _ctx_live.get(_role, (0, _windows_now[_role]))[0]
                out += [("", f"{_abbr} {_used // 1024}k/{_windows_now[_role] // 1024}k"), ("", SEP)]

        # Project name + permission
        _perm = config.permission
        _pc = {"low": "ansiyellow", "mid": "ansigreen", "high": "ansired"}.get(_perm, "ansigreen")
        out += [("", f"{project_root.name}  "), (f"{_pc} bold", _perm)]

        # Compact live plan/todo in the same info line (the "über dem userinput mit den live stats").
        # Populated from the last task's plan_live (updated during the rich bar + planner flush).
        _plabel = (_last_plan or {}).get("label") or ""
        _pcomp = (_last_plan or {}).get("compact") or ""
        if _plabel or _pcomp:
            out += [("", "  │  "), ("magenta", _plabel)]
            if _pcomp:
                short = _pcomp[:45] + ("…" if len(_pcomp) > 45 else "")
                out += [("", " "), ("dim", short)]

        out.append(("", "  "))
        return out

    panel = _BottomPanel(
        get_info=_make_toolbar,
        history_path=_HISTORY_FILE,
        completer=_COMPLETER,
    )

    while True:
        try:
            line = panel.prompt()
        except KeyboardInterrupt:
            print_info("Use [cyan]/exit[/cyan] to quit.")
            continue
        except EOFError:
            console.print()
            print_info("Goodbye.")
            break

        if not line:
            continue

        if line.startswith("/"):
            config, scope, signal = handle(line, config, scope, agent=agent, logger=logger)
            agent.config = config
            if signal == "exit":
                logger.disable()
                print_info("Goodbye.")
                break
            elif signal == "clear_history":
                agent.clear_history()
                print_success("Conversation history cleared.")
            elif signal == "clear_screen":
                console.clear()
            elif signal == "show_tokens":
                usage = agent.session_usage
                print_usage(usage["prompt_tokens"], usage["completion_tokens"])
            continue

        # immediate feedback on main screen
        console.print(user_message_panel(line))
        logger.log_input(line)

        try:
            _plan_holder = dict(_last_plan)  # mutable for the task
            _last_task_s = _stream_agent(agent, line, logger, _ctx_state=_ctx_live, _active_role=_active_role, _plan_state=_plan_holder)
            # Persist for the next idle prompt's toolbar (the line above the input with the live stats)
            if _plan_holder.get("label") or _plan_holder.get("compact"):
                _last_plan.update(_plan_holder)
        except Exception as exc:
            print_error(f"Agent error: {escape(str(exc))}")
            logger.log_error(str(exc))
