#!/usr/bin/env python3
"""Kratos — abliterated 4-role CLI AI agent (max context, stepwise, lossless memory).

ALL models are abliterated (no safety filters).

Roles (each uses the best abliterated model for a laptop + its job):
  Planner    : huihui_ai/qwen3-abliterated:8b   — plans + CoT, FULL 40K ctx every call
  Coder      : huihui_ai/qwen3.5-abliterated:4b — implements, FULL 262K ctx (huge repos)
  Verifier   : huihui_ai/qwen3-abliterated:8b   — strict PROVEN_WORK + step checks, FULL ctx
  Auto-Composer (Compressor): kratos-planner (Phi-4-mini-abliterated) — .kratos memory + history
                              NEVER destroys information, always max ctx.

Pipeline guarantees:
- Coder walks the plan ONE STEP AT A TIME: think how, show verify cmd, code it, runtime runs test,
  only then next step.
- Verifier only says VERIFIED after real tests executed per step + final sweep.
- Compressor (auto-composer) + Memory in .kratos are lossless (exhaustive facts, verbatim quotes).
- Every model call uses the MAXIMUM context window the model supports (within vram cap).
- Works for tiny tasks and for repos that massively exceed any single ctx (relay + memory + compress).

Usage:
    python kratos.py            # interactive REPL
    python kratos.py --setup    # model setup wizard
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from rich.console import Console
    from rich.markup import escape
except ImportError:
    sys.exit("Install dependencies first:  pip install -r requirements.txt")

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.styles import Style as PTStyle
    _HAS_PT = True
except ImportError:
    _HAS_PT = False

try:
    from rich.live import Live
    from rich.layout import Layout
except ImportError:
    pass

from kratos.config import KratosConfig, GLOBAL_DIR
from kratos.bridge import OllamaBridge
from kratos.agent import KratosAgent
from kratos.commands import handle
from kratos.logger import SessionLogger
from kratos.prompts import load_prompts, reload_prompts
from kratos.ui import (
    console,
    print_banner, print_error, print_info, print_warn, print_success,
    print_user_msg, print_section_time,
    print_usage, print_ctx_bar, print_compress_event,
    route_info, tool_call,
    LiveBuffer, status_bar, user_message_panel, task_summary_panel,
    section_banner, elapsed_str,
    input_capsule,
)

_HISTORY_FILE = GLOBAL_DIR / "history.txt"
_PT_STYLE = PTStyle.from_dict({
    "prompt":                               "ansicyan bold",
    "completion-menu.completion":           "bg:#1e2a35 #aaaaaa",
    "completion-menu.completion.current":   "bg:#0078d4 #ffffff bold",
    "completion-menu.meta.completion":      "bg:#1e2a35 #666666",
    "completion-menu.meta.completion.current": "bg:#005fa3 #cccccc",
}) if _HAS_PT else None

# ── slash-command autocomplete tree ──────────────────────────────────────────
_SLASH_TREE: dict[str, tuple[dict[str, str] | None, str]] = {
    "exit":       (None,                                               "Quit Kratos"),
    "quit":       (None,                                               "Quit Kratos"),
    "q":          (None,                                               "Quit Kratos"),
    "help":       (None,                                               "Show all commands"),
    "clear":      (None,                                               "Clear screen"),
    "status":     (None,                                               "Show status bar"),
    "setup":      (None,                                               "Model setup info"),
    "tokens":     (None,                                               "Show session token usage"),
    "goal":       ({"clear": "Clear goal"},                            "Set or show goal"),
    "scope":      ({"global": "Machine-wide config",
                    "project": "Per-project config",
                    "info":    "Show paths"},                          "Config scope"),
    "permission": ({"low":  "Read only",
                    "mid":  "Read + write",
                    "high": "Read + write + delete"},                  "Coder permissions"),
    "models":     ({"planner":    "Change planner model",
                    "coder":      "Change coder model",
                    "verifier":   "Change verifier model",
                    "compressor": "Change compressor model"},          "Model config"),
    "index":      ({"rebuild": "Rescan project files"},                "Project file index"),
    "memory":     ({"list":    "Show all entries",
                    "clear":   "Clear session/project/all"},           "Persistent memory"),
    "prompts":    ({"list": "Show roles/snippets", "reload": "Reload from json", "dump": "Write defaults to file"}, "Edit system prompts (JSON)"),
    "history":    ({"clear": "Reset conversation"},                    "Conversation history"),
    "build":      ({"clear": "Remove build command"},                  "Build command"),
    "test":       ({"clear": "Remove test command"},                   "Test command"),
    "logging":    ({"on": "Start logging", "off": "Stop logging"},     "Session logging"),
}

if _HAS_PT:
    class _SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            after_slash = text[1:]
            space_idx   = after_slash.find(" ")
            if space_idx == -1:
                partial_cmd = after_slash.lower()
                for name, (_, desc) in sorted(_SLASH_TREE.items()):
                    if name.startswith(partial_cmd):
                        yield Completion(
                            "/" + name, start_position=-len(text),
                            display=f"/{name}", display_meta=desc,
                        )
            else:
                cmd         = after_slash[:space_idx].lower()
                partial_sub = after_slash[space_idx + 1:].lower()
                if cmd in _SLASH_TREE:
                    subcmds, _ = _SLASH_TREE[cmd]
                    if subcmds:
                        for sub, sub_desc in sorted(subcmds.items()):
                            if sub.startswith(partial_sub):
                                yield Completion(
                                    sub, start_position=-len(partial_sub),
                                    display=sub, display_meta=sub_desc,
                                )
    _COMPLETER: "Completer | None" = _SlashCompleter()
else:
    _COMPLETER = None


# ── input helper ──────────────────────────────────────────────────────────────

def _input(session: "PromptSession | None") -> str:
    """Read one line using prompt_toolkit (with history/completion) or plain input."""
    if session and _HAS_PT:
        return session.prompt(
            [("class:prompt", "  kratos ❯ ")],
            style=_PT_STYLE,
        )
    return input("kratos ❯ ")


def _ctx_display(config: KratosConfig) -> dict[str, int]:
    d = {
        "planner": config.planner_num_ctx,
        "coder": config.coder_num_ctx,
        "verifier": config.verifier_num_ctx,
        "compressor": config.compressor_num_ctx,
        "relay": config.relay_num_ctx,
        "vram_cap": config.vram_ctx_ceiling,
    }
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
            logger.log_file_write(rel_path, content)

    for rel_path in deletions:
        if not agent.config.can_delete():
            print_warn(f"Delete blocked (permission={perm}). Use /permission high.")
            break
        if logger:
            logger.log_file_delete(rel_path)


# ── coder output filter ───────────────────────────────────────────────────────

class _CoderFilter:
    """Line-buffer that shows only file-operation markers and the SUMMARY section.

    The raw code bodies are suppressed — the agent wrote them to disk.
    Output goes to a LiveBuffer instead of directly to console.
    """

    def __init__(self, buffer: "LiveBuffer") -> None:
        self._buf: str = ""
        self._in_summary = False
        self._in_code = False
        self._seen: set[str] = set()
        self._buffer = buffer

    def feed(self, token: str) -> None:
        self._buf += token
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._process(line)

    def flush(self) -> None:
        if self._buf.strip():
            self._process(self._buf)
        self._buf = ""

    def _process(self, line: str) -> None:
        s = line.strip()
        try:
            from kratos.prompts import get_marker
            fm = get_marker("file") or "### FILE:"
            dm = get_marker("delete") or "### DELETE:"
            sm = get_marker("summary") or "### SUMMARY"
        except Exception:
            fm, dm, sm = "### FILE:", "### DELETE:", "### SUMMARY"

        if s.startswith(fm):
            path = s[len(fm):].strip()
            if path and path not in self._seen:
                self._seen.add(path)
                self._buffer.add(f"  ↳ write_file({path!r})", "dim blue")
            self._in_summary = False
            self._in_code = False
        elif s.startswith(dm):
            path = s[len(dm):].strip()
            if path:
                self._buffer.add(f"  ↳ delete_file({path!r})", "dim blue")
            self._in_summary = False
            self._in_code = False
        elif s.startswith(sm):
            self._in_summary = True
            self._in_code = False
            self._buffer.add("  ── Summary ──", "dim")
        elif s.startswith("```"):
            self._in_code = not self._in_code
        elif self._in_summary and not self._in_code and s:
            self._buffer.add(f"  {s}", "dim")


# ── agent streaming ───────────────────────────────────────────────────────────

def _stream_agent(
    agent: KratosAgent, task: str, logger: "SessionLogger",
    _ctx_state: "dict | None" = None,
) -> float:
    """Run the pipeline with rich.live.Live for a persistent bottom status bar.

    Streaming output goes to an alternate screen via ``Live(screen=True)``.
    After the task completes, the full output + summary are printed to the
    main screen so prompt_toolkit can continue below them.

    Returns the total elapsed seconds.
    """
    # Shared state
    ctx_live: dict[str, tuple[int, int]] = {
        "planner": (0, agent.config.planner_num_ctx),
        "coder":   (0, agent.config.coder_num_ctx),
        "verifier": (0, agent.config.verifier_num_ctx),
    }
    current_section = ""
    section_start = time.monotonic()
    task_start = time.monotonic()
    planner_buf = ""
    coder_buf = ""

    # Render output through LiveBuffer → Live display
    body = LiveBuffer(max_lines=400)
    coder_filter = _CoderFilter(body)

    # ── build user label for footer (merged with status bar) ──────────────
    from rich.markup import escape as _escape
    _user_label = _escape(task)
    if len(_user_label) > 60:
        _user_label = _user_label[:57] + "…"

    # ── build 2-row layout: body | footer ─────────────────────────────────
    layout = Layout()
    layout.split_column(
        Layout(name="body", ratio=1, minimum_size=1),
        Layout(name="footer", size=4),
    )

    def _make_footer(elapsed: float, section: str) -> "Panel":
        return status_bar(
            agent.config.scope or "project",
            agent.config.permission,
            ctx_live,
            elapsed,
            Path.cwd().name,
            current_section=section,
            session_tokens=(
                agent.session_usage.get("prompt_tokens", 0),
                agent.session_usage.get("completion_tokens", 0),
            ),
            goal=agent.config.goal,
            user_label=_user_label,
            hint="^C stop",
        )

    def _section_model(role: str) -> str:
        if role == "planner":
            return agent.config.planner_model
        if role == "coder":
            return agent.config.coder_model
        if role == "verify":
            return agent.config.verifier_model
        return ""

    def _log_tool(content: str) -> None:
        if "(" in content:
            name = content[:content.index("(")]
            args_str = content[content.index("(") + 1:content.rindex(")")]
            result = ""
            if "→" in content:
                result = content.split("→")[-1].strip()
            elif "->" in content:
                result = content.split("->")[-1].strip()
            logger.log_tool(name=name, args={"raw": args_str}, result=result)
        else:
            logger.log_tool(name=content, args={})

    layout["body"].update(body.render())
    layout["footer"].update(_make_footer(0, ""))

    try:
        with Live(layout, screen=True, refresh_per_second=10,
                  transient=True) as live:

            for source, content, kind in agent.process(task):
                elapsed = time.monotonic() - task_start

                # ── log events (never shown in UI) ─────────────────────────
                if source == "log":
                    if logger.enabled:
                        try:
                            import json as _json
                            data = _json.loads(content)
                            event_type = data.pop("type", "unknown")
                            logger._write(event_type, **data)
                        except Exception:
                            pass

                elif source == "router":
                    body.add(f"  ⟶  {content}", "dim cyan")
                    parts = dict(p.split("=", 1) for p in content.split("  ") if "=" in p)
                    logger.log_route(intent=parts.get("intent", ""), route=parts.get("route", ""))

                elif source == "tool":
                    body.add(f"  ↳ {content}", "dim blue")
                    _log_tool(content)

                elif source == "header":
                    current_section = content
                    section_start = time.monotonic()
                    body.add("", "")
                    body.add(section_banner(content, _section_model(content)).plain, _role_style_for(content))

                elif source == "planner":
                    if kind == "think":
                        body.add(content, "dim italic")
                    else:
                        planner_buf += content
                        body.add(content, "")

                elif source == "verify":
                    if kind == "think":
                        body.add(content, "dim italic")
                    else:
                        body.add(content, "")

                elif source == "coder":
                    if kind == "think":
                        body.add(content, "dim italic")
                    else:
                        coder_buf += content
                        coder_filter.feed(content)

                elif source == "relay":
                    pass

                elif source == "direct":
                    body.add("", "")
                    body.add(section_banner("direct", "").plain, _role_style_for("direct"))
                    body.add(content, "")

                elif source == "usage":
                    try:
                        import json as _json
                        d = _json.loads(content)
                        logger._write("token_usage", **d)
                    except Exception:
                        pass

                elif source == "end":
                    sec_elapsed = time.monotonic() - section_start
                    body.add(f"  ⏱ {elapsed_str(sec_elapsed)}", "dim")
                    body.add("", "")
                    if current_section == "planner" and planner_buf:
                        logger.log_model_output("planner", agent.config.planner_model, planner_buf)
                        planner_buf = ""
                    elif current_section == "coder":
                        coder_filter.flush()
                        if coder_buf:
                            logger.log_model_output("coder", agent.config.coder_model, coder_buf)
                        coder_buf = ""
                    current_section = ""

                elif source == "info":
                    body.add(f"  ℹ  {content}", "blue")
                    logger.log_info(content)

                elif source == "warn":
                    body.add(f"  ⚠  {content}", "yellow")

                elif source == "error":
                    body.add(f"  ✗  {content}", "bold red")
                    logger.log_error(content)

                elif source == "ctx_info":
                    parts = content.split("|")
                    if len(parts) == 3:
                        try:
                            role_name, used_s, total_s = parts[0], int(parts[1]), int(parts[2])
                            ctx_live[role_name] = (used_s, total_s)
                            if _ctx_state is not None:
                                _ctx_state[role_name] = (used_s, total_s)
                        except ValueError:
                            pass

                elif source == "compress":
                    body.add(f"  ⇒ compress  {content}", "magenta")
                    logger.log_info(f"[compress] {content}")

                elif source == "question":
                    body.add("", "")
                    body.add(f"  Kratos: {content}", "cyan")

                # Update the Live display
                layout["body"].update(body.render())
                layout["footer"].update(_make_footer(elapsed, current_section))

    except KeyboardInterrupt:
        # Live context already exited; fall through to summary
        pass

    # ── post-stream: print output + summary to main screen ────────────────
    console.print(body.render())

    _show_file_ops(agent, logger)
    console.print()

    total_elapsed = time.monotonic() - task_start
    files = [p for p, _ in agent.pending_file_changes] + agent.pending_file_deletions
    usage = agent.session_usage
    tok = (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    console.print(task_summary_panel(total_elapsed, files, tok))

    return total_elapsed


def _role_style_for(role: str) -> str:
    return {
        "planner": "bold cyan", "coder": "bold green", "verify": "bold yellow",
        "relay": "bold magenta", "direct": "bold blue",
    }.get(role, "bold")


# (The previous custom prompt_toolkit full-screen "chat" execution view with its
# own bottom input window has been removed. See comment in the main REPL loop
# for why we now use the stable rich Live path for everything after you submit
# a real prompt.)


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

    from pathlib import Path as _Path
    from kratos.config import _project_dir
    project_root = _Path.cwd()
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

    session = None
    if _HAS_PT:
        GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
        session = PromptSession(
            history=FileHistory(str(_HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=_COMPLETER,
            complete_while_typing=True,
        )

    # Live ctx tracking: pre-seeded so the capsule shows real numbers from the start.
    _ctx_live: dict[str, tuple[int, int]] = {
        "planner": (0, config.planner_num_ctx),
        "coder":   (0, config.coder_num_ctx),
        "verifier": (0, config.verifier_num_ctx),
    }
    _last_task_s: float | None = None

    while True:
        try:
            _tok = agent.session_usage
            input_capsule(
                _ctx_live, config, _last_task_s,
                session_start=_session_start,
                total_tokens=_tok.get("prompt_tokens", 0) + _tok.get("completion_tokens", 0),
                project_name=project_root.name,
            )
            line = _input(session).strip()
        except KeyboardInterrupt:
            console.print()
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
            _last_task_s = _stream_agent(agent, line, logger, _ctx_state=_ctx_live)
        except Exception as exc:
            print_error(f"Agent error: {escape(str(exc))}")
            logger.log_error(str(exc))


if __name__ == "__main__":
    if "--setup" in sys.argv:
        import setup_models as sm
        sm.setup()
    else:
        main()