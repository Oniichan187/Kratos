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
    from prompt_toolkit.completion import Completer, Completion
    _HAS_PT = True
except ImportError:
    _HAS_PT = False


from kratos.config import KratosConfig, GLOBAL_DIR
from kratos.bridge import OllamaBridge
from kratos.agent import KratosAgent
from kratos.commands import handle
from kratos.logger import SessionLogger
from kratos.prompts import load_prompts, reload_prompts
from kratos.ui import (
    console,
    print_banner, print_error, print_info, print_warn, print_success,
    print_usage,
    user_message_panel, task_summary_panel,
    section_banner, elapsed_str,
)

_HISTORY_FILE = GLOBAL_DIR / "history.txt"

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


# ── framed bottom prompt panel ────────────────────────────────────────────────

if _HAS_PT:
    class _BottomPanel:
        """Non-fullscreen prompt_toolkit Application — framed input + live info bar.

        Terminal layout while waiting for input:
          ╭────────────────────────────────────────────────────────╮
          │  ⏱ 3m  ∑ 8k  ░░░░░░░░ 0%→compose  project  mid       │
          ├────────────────────────────────────────────────────────┤
          │  kratos ❯ _                                            │
          ╰────────────────────────────────────────────────────────╯

        erase_when_done=True erases the frame on Enter so the chat log above
        grows cleanly and the frame always re-renders at the bottom.
        """

        def __init__(
            self,
            get_info: "Callable[[], list[tuple[str, str]]]",
            history_path: "Path",
            completer: "Completer | None" = None,
        ) -> None:
            from prompt_toolkit.buffer import Buffer as _Buf
            from prompt_toolkit.history import FileHistory as _FH
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory as _ASFH
            self._get_info = get_info
            self._buf = _Buf(
                history=_FH(str(history_path)),
                completer=completer,
                complete_while_typing=True,
                auto_suggest=_ASFH(),
            )

        def prompt(self) -> str:
            """Block until Enter; return stripped text. Raises KeyboardInterrupt / EOFError."""
            import shutil
            from prompt_toolkit.application import Application as _App
            from prompt_toolkit.layout import Layout as _Lay
            from prompt_toolkit.layout.containers import HSplit as _HS, Window as _W
            from prompt_toolkit.layout.controls import (
                BufferControl as _BC,
                FormattedTextControl as _FTC,
            )
            from prompt_toolkit.layout.processors import BeforeInput as _BI
            from prompt_toolkit.key_binding import KeyBindings as _KB
            from prompt_toolkit.key_binding.defaults import load_key_bindings as _lkb
            from prompt_toolkit.key_binding import merge_key_bindings as _mkb
            from prompt_toolkit.styles import Style as _Sty

            buf = self._buf

            def _cols() -> int:
                return shutil.get_terminal_size((80, 24)).columns

            def _top() -> list[tuple[str, str]]:
                return [("class:bdr", "╭" + "─" * (_cols() - 2) + "╮")]

            def _mid() -> list[tuple[str, str]]:
                return [("class:bdr", "├" + "─" * (_cols() - 2) + "┤")]

            def _bot() -> list[tuple[str, str]]:
                return [("class:bdr", "╰" + "─" * (_cols() - 2) + "╯")]

            def _info() -> list[tuple[str, str]]:
                row: list[tuple[str, str]] = [("class:bdr", "│  ")]
                row.extend(self._get_info())
                return row

            kb = _KB()

            @kb.add("enter")
            def _enter(event) -> None:
                text = buf.text
                buf.reset(append_to_history=True)
                event.app.exit(result=text)

            @kb.add("c-c")
            def _cc(event) -> None:
                buf.reset()
                event.app.exit(exception=KeyboardInterrupt())

            @kb.add("c-d")
            def _cd(event) -> None:
                if not buf.text:
                    event.app.exit(exception=EOFError())
                else:
                    buf.reset()

            layout = _Lay(
                _HS([
                    _W(_FTC(_top), height=1, dont_extend_height=True),
                    _W(_FTC(_info), height=1, dont_extend_height=True),
                    _W(_FTC(_mid), height=1, dont_extend_height=True),
                    _W(
                        _BC(
                            buffer=buf,
                            input_processors=[_BI([("class:prompt", "│  kratos ❯ ")])],
                        ),
                        height=1,
                        dont_extend_height=True,
                    ),
                    _W(_FTC(_bot), height=1, dont_extend_height=True),
                ]),
                focused_element=buf,
            )

            style = _Sty.from_dict({
                "bdr":                                      "#2d6070",
                "prompt":                                   "ansicyan bold",
                "auto-suggestion":                          "#445566",
                "completion-menu.completion":               "bg:#1e2a35 #aaaaaa",
                "completion-menu.completion.current":       "bg:#0078d4 #ffffff bold",
                "completion-menu.meta.completion":          "bg:#1e2a35 #666666",
                "completion-menu.meta.completion.current":  "bg:#005fa3 #cccccc",
            })

            app = _App(
                layout=layout,
                key_bindings=_mkb([_lkb(), kb]),
                style=style,
                erase_when_done=True,
                full_screen=False,
                mouse_support=False,
            )
            result = app.run()
            return (result or "").strip()

else:
    class _BottomPanel:  # type: ignore[no-redef]
        def __init__(self, get_info, history_path=None, completer=None):
            self._get_info = get_info

        def prompt(self) -> str:
            return input("kratos ❯ ").strip()


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
    """Suppress code bodies; print only file-op markers and the SUMMARY section."""

    def __init__(self) -> None:
        self._buf: str = ""
        self._in_summary = False
        self._in_code = False
        self._seen: set[str] = set()

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
                console.print(f"  [dim blue]↳[/dim blue] write_file([dim]{path!r}[/dim])")
            self._in_summary = False
            self._in_code = False
        elif s.startswith(dm):
            path = s[len(dm):].strip()
            if path:
                console.print(f"  [dim blue]↳[/dim blue] delete_file([dim]{path!r}[/dim])")
            self._in_summary = False
            self._in_code = False
        elif s.startswith(sm):
            self._in_summary = True
            self._in_code = False
            console.print()
            console.print("  [dim]── Summary ──[/dim]")
        elif s.startswith("```"):
            self._in_code = not self._in_code
        elif self._in_summary and not self._in_code and s:
            console.print(f"  [dim]{s}[/dim]")


# ── agent streaming ───────────────────────────────────────────────────────────

def _stream_agent(
    agent: KratosAgent, task: str, logger: "SessionLogger",
    _ctx_state: "dict | None" = None,
) -> float:
    """Run the agent pipeline; stream output inline via console.print.

    No alternate screen / Live — output prints directly so the chat
    history accumulates above the prompt frame.  Returns elapsed seconds.
    """
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
    coder_filter = _CoderFilter()

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

    try:
        for source, content, kind in agent.process(task):

            if source == "log":
                if logger.enabled:
                    try:
                        import json as _j
                        d = _j.loads(content); et = d.pop("type", "unknown")
                        logger._write(et, **d)
                    except Exception:
                        pass

            elif source == "router":
                console.print(f"  [dim cyan]⟶[/dim cyan]  [dim]{content}[/dim]")
                kv = dict(p.split("=", 1) for p in content.split("  ") if "=" in p)
                logger.log_route(intent=kv.get("intent", ""), route=kv.get("route", ""))

            elif source == "tool":
                console.print(f"  [dim blue]↳[/dim blue]  [dim]{content}[/dim]")
                _log_tool(content)

            elif source == "header":
                current_section = content
                section_start = time.monotonic()
                console.print()
                console.print(section_banner(content, _smodel(content)))

            elif source == "planner":
                if kind == "think":
                    console.print(content, end="", style="dim italic", highlight=False)
                else:
                    console.print(content, end="", highlight=False)
                    planner_buf += content

            elif source == "verify":
                if kind == "think":
                    console.print(content, end="", style="dim italic", highlight=False)
                else:
                    console.print(content, end="", highlight=False)

            elif source == "coder":
                if kind == "think":
                    console.print(content, end="", style="dim italic", highlight=False)
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
                    d = _j.loads(content); logger._write("token_usage", **d)
                except Exception:
                    pass

            elif source == "end":
                sec_elapsed = time.monotonic() - section_start
                console.print()
                console.print(f"  [dim]⏱ {elapsed_str(sec_elapsed)}[/dim]")
                console.print()
                if current_section == "planner" and planner_buf:
                    logger.log_model_output("planner", agent.config.planner_model, planner_buf)
                    planner_buf = ""
                elif current_section == "coder":
                    coder_filter.flush()
                    if coder_buf:
                        logger.log_model_output("coder", agent.config.coder_model, coder_buf)
                    coder_buf = ""
                current_section = ""
                if _ctx_state is not None:
                    _ctx_state.update(ctx_live)

            elif source == "info":
                console.print(f"  [blue]ℹ[/blue]  [dim]{content}[/dim]")
                logger.log_info(content)

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

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]⚠[/yellow]  Interrupted.")

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

    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

    # Push cursor to bottom of terminal so the frame appears anchored there
    # on startup (subsequent prompts are already at the bottom after output).
    import shutil as _sh
    _rows = _sh.get_terminal_size((80, 24)).lines
    sys.stdout.write("\n" * max(0, _rows - 8))
    sys.stdout.flush()

    # Live ctx tracking: updated by _stream_agent as model calls report token counts.
    _ctx_live: dict[str, tuple[int, int]] = {
        "planner": (0, config.planner_num_ctx),
        "coder":   (0, config.coder_num_ctx),
        "verifier": (0, config.verifier_num_ctx),
    }
    _last_task_s: float | None = None

    # ── live info content for the panel frame ─────────────────────────────────
    def _make_toolbar() -> list[tuple[str, str]]:
        """Builds prompt_toolkit bottom-toolbar: ⏱ lifetime · ∑ tokens · %→compose · project · perm."""
        import time as _t
        SEP = "   │   "
        out: list[tuple[str, str]] = [("", "  ")]

        # Session lifetime
        age_s = _t.time() - _session_start
        if age_s < 60:
            life = "<1m"
        elif age_s < 3600:
            life = f"{int(age_s // 60)}m"
        else:
            _h, _m = int(age_s // 3600), int((age_s % 3600) // 60)
            life = f"{_h}h {_m}m"
        out += [("", "⏱ "), ("bold", life), ("", SEP)]

        # Cumulative token usage
        _tok = agent.session_usage
        _total = _tok.get("prompt_tokens", 0) + _tok.get("completion_tokens", 0)
        if _total >= 1_000_000:
            tok_str = f"{_total / 1_000_000:.1f}M"
        elif _total >= 1000:
            tok_str = f"{_total // 1000}k"
        else:
            tok_str = str(_total)
        out += [("", "∑ "), ("ansicyan", tok_str), ("", SEP)]

        # % until auto-compose
        _coder_total = config.coder_num_ctx
        _threshold = float(getattr(config, "compress_threshold", 0.75))
        _coder_used = _ctx_live.get("coder", (0, _coder_total))[0]
        _compose_at = max(1, _threshold * _coder_total)
        _pct = min(100, int(_coder_used * 100 / _compose_at))
        _bw = 8
        _filled = min(_bw, int(_bw * _pct / 100))
        _bar = "▓" * _filled + "░" * (_bw - _filled)
        _bc = "ansired" if _pct > 80 else "ansiyellow" if _pct > 50 else "ansibrightblack"
        out += [(_bc, _bar), ("", f" {_pct}%→compose"), ("", SEP)]

        # Project name + permission
        _perm = config.permission
        _pc = {"low": "ansiyellow", "mid": "ansigreen", "high": "ansired"}.get(_perm, "ansigreen")
        out += [("", f"{project_root.name}  "), (f"{_pc} bold", _perm)]

        # Last task duration
        if _last_task_s is not None:
            _t_str = f"{_last_task_s:.0f}s" if _last_task_s < 60 else f"{_last_task_s / 60:.1f}m"
            out += [("", SEP), ("", f"last {_t_str}")]

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