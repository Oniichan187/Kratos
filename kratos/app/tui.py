"""Textual full-screen TUI for Kratos.

Launch via:  python kratos.py --tui

Layout:
  KratosApp
  └─ Screen
     ├─ VerticalScroll(id="log")        # overall scrollable conversation area
     │    ├─ Static(...)                # banner, startup messages, slash output
     │    ├─ UserMessage(...)           # each user turn — max 7 rows, internal scroll
     │    └─ AssistantTurn(...)         # each agent run — full height, streamed into
     └─ Vertical(id="inputbar")         # docked bottom
          ├─ StatusFooter               # ⏱ tokens compose% project perm
          └─ Horizontal(id="input_row")
               ├─ Static("│  kratos ❯ ")
               └─ PromptInput(TextArea) # 1→5 lines, then internal scroll
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path

from rich.markup import escape as _rich_escape
from rich.text import Text as RichText
from rich.console import Console as RichConsole

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.message import Message
from textual.widgets import Static, TextArea
from textual import events

from ..llm.tokens import role_context_windows
from kratos.ui import elapsed_str, _tok_short
from .slash import _SLASH_TREE, slash_completions


# ── Ported CoderFilter ────────────────────────────────────────────────────────

class _TuiCoderFilter:
    """Port of kratos.py _CoderFilter; routes output via a callback instead of console.print."""

    def __init__(self, emit: "callable[[str, str], None]") -> None:
        self._emit = emit          # emit(text: str, style: str)
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
                self._emit(f"  ↳ write_file({path!r})", "dim blue")
            self._in_summary = False
            self._in_code = False
        elif s.startswith(dm):
            path = s[len(dm):].strip()
            if path:
                self._emit(f"  ↳ delete_file({path!r})", "dim blue")
            self._in_summary = False
            self._in_code = False
        elif s.startswith(sm):
            self._in_summary = True
            self._in_code = False
            self._emit("  ── Summary ──", "dim")
        elif s.startswith("```"):
            self._in_code = not self._in_code
        elif self._in_summary and not self._in_code and s:
            self._emit(f"  {s}", "dim")


# ── Custom Messages ───────────────────────────────────────────────────────────

class AgentEvent(Message):
    """One (source, content, kind) event from agent.process()."""

    def __init__(self, source: str, content: str, kind: str) -> None:
        self.source = source
        self.content = content
        self.kind = kind
        super().__init__()


class AgentDone(Message):
    """Agent pipeline finished (or was cancelled)."""

    def __init__(
        self,
        elapsed: float,
        files_changed: list[str],
        token_usage: tuple[int, int],
        *,
        interrupted: bool = False,
    ) -> None:
        self.elapsed = elapsed
        self.files_changed = files_changed
        self.token_usage = token_usage
        self.interrupted = interrupted
        super().__init__()


# ── Widgets ───────────────────────────────────────────────────────────────────

class UserMessage(VerticalScroll):
    """Compact display of a past user message.

    Grows from 1 row to MAX_ROWS (7 = 5 content + 2 Panel border) then
    scrolls internally — the outer ConversationLog does NOT grow for this box.
    """

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        max-height: 7;
        overflow-y: auto;
        margin-bottom: 1;
        background: transparent;
    }
    """

    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text

    def compose(self) -> ComposeResult:
        from kratos.ui import user_message_panel
        yield Static(user_message_panel(self._text))


class AssistantTurn(Vertical):
    """Container for one complete agent response.

    No height cap — participates in the overall ConversationLog scroll.
    Widgets are mounted dynamically as events arrive.
    """

    DEFAULT_CSS = """
    AssistantTurn {
        height: auto;
        margin-bottom: 1;
    }
    """


class PromptInput(TextArea):
    """Multi-line input field.

    Grows from 1 to 5 rows, then scrolls internally within the box.
    Enter = submit.  Shift+Enter / Ctrl+Enter = insert newline.
    Ctrl+C = cancel running task / hint.  Ctrl+D on empty = quit app.
    """

    DEFAULT_CSS = """
    PromptInput {
        height: auto;
        max-height: 5;
        background: transparent;
        border: none;
        padding: 0;
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "user_cancel", "Cancel", show=False, priority=True),
        Binding("ctrl+d", "user_quit", "Quit", show=False, priority=True),
    ]

    def on_mount(self) -> None:
        # Visible slider once content exceeds the 5-row max-height (matches the
        # legacy CLI's prompt_toolkit ScrollbarMargin on the same box).
        self.show_vertical_scrollbar = True

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text
            if text.strip():
                self.load_text("")
                self.post_message(self.Submitted(text))
        elif event.key in ("shift+enter", "ctrl+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")

    def action_user_cancel(self) -> None:
        self.post_message(self.Submitted("\x03"))

    def action_user_quit(self) -> None:
        if not self.text.strip():
            self.app.exit()


class StatusFooter(Static):
    """One-line info bar: ⏱ lifetime · ∑ tokens · compose% · project · perm · last task."""

    DEFAULT_CSS = """
    StatusFooter {
        height: 1;
        width: 100%;
        color: $text-muted;
        padding: 0 1;
        background: transparent;
    }
    """


# ── Main App ──────────────────────────────────────────────────────────────────

class KratosApp(App):
    """Persistent full-screen Textual TUI for Kratos."""

    CSS = """
    Screen {
        background: $surface;
    }

    #log {
        height: 1fr;
        width: 100%;
        overflow-y: auto;
        padding: 0 1;
    }

    #inputbar {
        height: auto;
        max-height: 9;
        width: 100%;
        dock: bottom;
        background: $panel;
        border-top: tall $primary-darken-2;
    }

    #status_footer_row {
        height: 1;
        width: 100%;
        border-bottom: tall $primary-darken-3;
    }

    #input_row {
        height: auto;
        max-height: 7;
        width: 100%;
        layout: horizontal;
    }

    #prompt_label {
        width: 14;
        height: auto;
        content-align: left middle;
        color: $accent;
        text-style: bold;
        padding: 0 1;
    }

    PromptInput {
        width: 1fr;
        height: auto;
        max-height: 5;
        background: transparent;
        border: none;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_task", "Cancel", show=False, priority=True),
    ]

    def __init__(
        self,
        config,
        bridge,
        agent,
        logger,
        project_root: Path,
        session_start: float,
        scope: str = "project",
    ) -> None:
        super().__init__()
        self.kratos_config = config
        self.kratos_bridge = bridge
        self.kratos_agent = agent
        self.kratos_logger = logger
        self.project_root = project_root
        self.session_start = session_start
        self.scope = scope

        # runtime state
        windows = role_context_windows(config)
        self._ctx_live: dict[str, tuple[int, int]] = {
            "planner": (0, windows["planner"]),
            "coder":   (0, windows["coder"]),
            "verifier": (0, windows["verifier"]),
        }
        self._cancel_event = threading.Event()
        self._task_start: float = 0.0
        self._last_task_s: float | None = None
        self._busy: bool = False

        # streaming state (reset per agent run)
        self._current_turn: AssistantTurn | None = None
        self._stream_text = RichText()
        self._stream_static: Static | None = None
        self._current_section: str = ""
        self._last_banner_section: str = ""
        self._section_start: float = 0.0
        self._coder_filter: _TuiCoderFilter | None = None
        self._planner_buf: str = ""
        self._coder_buf: str = ""

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        with Vertical(id="inputbar"):
            with Horizontal(id="status_footer_row"):
                yield StatusFooter("", id="status_footer")
            with Horizontal(id="input_row"):
                yield Static("│  kratos ❯ ", id="prompt_label")
                yield PromptInput(id="prompt_input")

    def on_mount(self) -> None:
        self._show_startup()
        self.set_interval(1.0, self._refresh_footer)
        self.query_one("#prompt_input", PromptInput).focus()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _show_startup(self) -> None:
        output = self._capture(self._do_startup_prints)
        if output.strip():
            self._log_mount(Static(RichText.from_ansi(output)))

    def _do_startup_prints(self) -> None:
        from kratos.ui import print_banner, print_info
        c = self.kratos_config
        print_banner(
            c.planner_model, c.coder_model, self.scope, c.permission,
            verifier=c.verifier_model, compressor=c.compressor_model,
            ctx=self._ctx_display(), show_status_panel=True,
        )
        print_info(f"Project root: {self.project_root}")
        print_info("Type your task or /help for commands.  Ctrl+D to quit.")

    def _ctx_display(self) -> dict[str, int]:
        c = self.kratos_config
        d = role_context_windows(c)
        if getattr(c, "always_max_ctx", True):
            d["max_policy"] = 1
        return d

    # ── Input handling ────────────────────────────────────────────────────────

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.text.strip()
        if not text:
            return

        if text == "\x03":           # Ctrl+C from PromptInput action
            self.action_cancel_task()
            return

        if text.startswith("/"):
            self._handle_slash(text)
            return

        self._mount_user_message(text)
        self.kratos_logger.log_input(text)
        self._start_agent(text)

    def action_cancel_task(self) -> None:
        if self._busy:
            self._cancel_event.set()
            try:
                self.kratos_bridge.cancel_active()
            except Exception:
                pass
            self._log_markup("  [yellow]⚠[/yellow]  Cancelling current run…")
            try:
                inp = self.query_one("#prompt_input", PromptInput)
                inp.disabled = False
                inp.focus()
            except Exception:
                pass
        else:
            self._log_markup("  [dim]Use /exit or Ctrl+D to quit.[/dim]")

    # ── Slash commands ────────────────────────────────────────────────────────

    def _handle_slash(self, line: str) -> None:
        from kratos.commands import handle as _handle
        import kratos.ui as _ui
        import kratos.commands as _cmd

        buf = io.StringIO()
        w = self._term_width()
        cap = RichConsole(file=buf, force_terminal=True, width=w,
                          legacy_windows=False, highlight=False)
        old_ui = _ui.console
        old_cmd = getattr(_cmd, "console", None)
        _ui.console = cap
        if old_cmd is not None:
            _cmd.console = cap
        signal = None
        try:
            new_cfg, new_scope, signal = _handle(
                line, self.kratos_config, self.scope,
                agent=self.kratos_agent, logger=self.kratos_logger,
            )
            self.kratos_config = new_cfg
            self.kratos_agent.config = new_cfg
            self.scope = new_scope
        except Exception as exc:
            cap.print(f"[red]Error:[/red] {_rich_escape(str(exc))}")
        finally:
            _ui.console = old_ui
            if old_cmd is not None:
                _cmd.console = old_cmd

        out = buf.getvalue()
        if out.strip():
            self._log_mount(Static(RichText.from_ansi(out)))

        if signal == "exit":
            self.exit()
        elif signal == "clear_history":
            self.kratos_agent.clear_history()
            self._log_markup("  [green]✓[/green]  Conversation history cleared.")
        elif signal == "clear_screen":
            self.query_one("#log", VerticalScroll).remove_children()
        elif signal == "show_tokens":
            usage = self.kratos_agent.session_usage
            out2 = self._capture(lambda: __import__("kratos.ui", fromlist=["print_usage"])
                                  .print_usage(usage["prompt_tokens"], usage["completion_tokens"]))
            if out2.strip():
                self._log_mount(Static(RichText.from_ansi(out2)))

    # ── Agent worker ─────────────────────────────────────────────────────────

    def _start_agent(self, task: str) -> None:
        if self._busy:
            self._log_markup("  [yellow]⚠[/yellow]  Already running a task.")
            return
        self._busy = True
        self._cancel_event.clear()
        self.kratos_agent.set_cancel_event(self._cancel_event)
        windows = role_context_windows(self.kratos_config)
        for role in ("planner", "coder", "verifier"):
            self._ctx_live[role] = (0, windows[role])
        self._task_start = time.monotonic()
        self._planner_buf = ""
        self._coder_buf = ""
        self._last_banner_section = ""

        turn = AssistantTurn()
        self._current_turn = turn
        self._log_mount(turn)

        self.query_one("#prompt_input", PromptInput).disabled = True
        self.run_worker(self._agent_worker, task, thread=True, group="agent")

    def _agent_worker(self, task: str) -> None:
        """Background thread: iterate agent.process() and post AgentEvent messages."""
        logger = self.kratos_logger
        try:
            for source, content, kind in self.kratos_agent.process(task):
                if self._cancel_event.is_set():
                    break

                # log-only events — handle here without posting to UI
                if source == "log" and logger.enabled:
                    try:
                        import json as _j
                        d = _j.loads(content)
                        et = d.pop("type", "unknown")
                        logger._write(et, **d)
                    except Exception:
                        pass
                    continue
                if source == "usage":
                    try:
                        import json as _j
                        d = _j.loads(content)
                        logger._write("token_usage", **d)
                    except Exception:
                        pass
                    continue

                self.post_message(AgentEvent(source, content, kind))

        except Exception as exc:
            self.post_message(AgentEvent("error", str(exc), "error"))

        elapsed = time.monotonic() - self._task_start
        interrupted = self._cancel_event.is_set()
        if interrupted:
            self.kratos_agent.pending_file_changes.clear()
            self.kratos_agent.pending_file_deletions.clear()

        changes   = self.kratos_agent.pending_file_changes
        deletions = self.kratos_agent.pending_file_deletions
        files     = [p for p, _ in changes] + deletions
        perm      = self.kratos_agent.config.permission

        for rel_path, content in changes:
            if not self.kratos_agent.config.can_write():
                self.post_message(AgentEvent(
                    "warn", f"Write blocked (permission={perm}). Use /permission mid or high.", "warn"))
                break
            if logger:
                logger.log_file_write(rel_path, content)

        for rel_path in deletions:
            if not self.kratos_agent.config.can_delete():
                self.post_message(AgentEvent(
                    "warn", f"Delete blocked (permission={perm}). Use /permission high.", "warn"))
                break
            if logger:
                logger.log_file_delete(rel_path)

        usage = self.kratos_agent.session_usage
        tok   = (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        self.post_message(AgentDone(
            elapsed=elapsed, files_changed=files, token_usage=tok,
            interrupted=interrupted,
        ))

    # ── Agent event handler (the mapping ladder) ──────────────────────────────

    def on_agent_event(self, event: AgentEvent) -> None:  # noqa: C901
        src     = event.source
        content = event.content
        kind    = event.kind

        if src == "router":
            self._turn_markup(
                f"  [dim cyan]⟶[/dim cyan]  [dim]{_rich_escape(content)}[/dim]")
            try:
                kv = dict(p.split("=", 1) for p in content.split("  ") if "=" in p)
                self.kratos_logger.log_route(
                    intent=kv.get("intent", ""), route=kv.get("route", ""))
            except Exception:
                pass

        elif src == "tool":
            self._turn_markup(
                f"  [dim blue]↳[/dim blue]  [dim]{_rich_escape(content)}[/dim]")
            self._log_tool(content)

        elif src == "header":
            self._current_section = content
            self._section_start = time.monotonic()

            # Section banner — printed once per role (not on every stepwise re-entry);
            # the "Step N/M" info line is the per-step marker instead.
            if content != self._last_banner_section:
                self._last_banner_section = content
                from kratos.ui import section_banner
                model_name = {
                    "planner": self.kratos_config.planner_model,
                    "coder":   self.kratos_config.coder_model,
                    "verify":  self.kratos_config.verifier_model,
                }.get(content, "")
                self._turn_static(section_banner(content, model_name))

            # Fresh streaming target
            self._stream_text   = RichText()
            self._stream_static = Static(RichText())
            if self._current_turn is not None:
                self._current_turn.mount(self._stream_static)

            if content == "coder":
                self._coder_filter = _TuiCoderFilter(self._coder_emit)

        elif src in ("planner", "verify"):
            if self._stream_static is None:
                return
            if kind == "think":
                self._stream_text.append(content, style="dim italic")
            else:
                self._stream_text.append(content)
                if src == "planner":
                    self._planner_buf += content
            self._stream_static.update(self._stream_text)
            self._scroll_log()

        elif src == "coder":
            if kind == "think":
                if self._stream_static:
                    self._stream_text.append(content, style="dim italic")
                    self._stream_static.update(self._stream_text)
            else:
                self._coder_buf += content
                if self._coder_filter:
                    self._coder_filter.feed(content)
            self._scroll_log()

        elif src == "direct":
            self._turn_markup(
                f"  [bold blue]◆ DIRECT[/bold blue]\n{_rich_escape(content)}")

        elif src == "end":
            from kratos.ui import elapsed_str
            sec_elapsed = time.monotonic() - self._section_start
            self._turn_markup(f"  [dim]⏱ {elapsed_str(sec_elapsed)}[/dim]")

            if self._current_section == "planner" and self._planner_buf:
                self.kratos_logger.log_model_output(
                    "planner", self.kratos_config.planner_model, self._planner_buf)
                self._planner_buf = ""
            elif self._current_section == "coder":
                if self._coder_filter:
                    self._coder_filter.flush()
                if self._coder_buf:
                    self.kratos_logger.log_model_output(
                        "coder", self.kratos_config.coder_model, self._coder_buf)
                self._coder_buf = ""

            self._stream_static   = None
            self._stream_text     = RichText()
            self._current_section = ""

        elif src == "info":
            self._turn_markup(
                f"  [blue]ℹ[/blue]  [dim]{_rich_escape(content)}[/dim]")
            self.kratos_logger.log_info(content)

        elif src == "warn":
            self._turn_markup(f"  [yellow]⚠[/yellow]  {_rich_escape(content)}")

        elif src == "error":
            self._turn_markup(
                f"  [bold red]✗[/bold red]  {_rich_escape(content)}")
            self.kratos_logger.log_error(content)

        elif src == "compress":
            self._turn_markup(
                f"  [magenta]⇒ compress[/magenta]  [dim]{_rich_escape(content)}[/dim]")
            self.kratos_logger.log_info(f"[compress] {content}")

        elif src == "question":
            self._turn_markup(f"  [cyan]Kratos:[/cyan] {_rich_escape(content)}")

        elif src == "ctx_info":
            parts = content.split("|")
            if len(parts) == 3:
                try:
                    rn, us, ts = parts[0], int(parts[1]), int(parts[2])
                    self._ctx_live[rn] = (us, ts)
                except ValueError:
                    pass

    def on_agent_done(self, event: AgentDone) -> None:
        self._last_task_s = event.elapsed

        if event.interrupted:
            self._turn_markup("[yellow]⚠[/yellow]  Interrupted.")

        from kratos.ui import task_summary_panel
        summary = task_summary_panel(
            event.elapsed, event.files_changed, event.token_usage,
            status="interrupted" if event.interrupted else "completed",
        )
        if self._current_turn is not None:
            self._current_turn.mount(Static(summary))
        else:
            self._log_mount(Static(summary))

        self._current_turn    = None
        self._stream_static   = None
        self._current_section = ""
        self._coder_filter    = None
        self._busy            = False

        inp = self.query_one("#prompt_input", PromptInput)
        inp.disabled = False
        inp.focus()
        self._scroll_log()
        self._refresh_footer()

    # ── CoderFilter emit callback ─────────────────────────────────────────────

    def _coder_emit(self, text: str, style: str = "") -> None:
        rt = RichText(text, style=style)
        if self._current_turn is not None:
            self._current_turn.mount(Static(rt))
        self._scroll_log()

    # ── Status footer ─────────────────────────────────────────────────────────

    def _refresh_footer(self) -> None:
        try:
            self.query_one("#status_footer", StatusFooter).update(self._footer_text())
        except Exception:
            pass

    def _footer_text(self) -> RichText:
        """Builds the bottom status line for TUI. Always includes both ⏱ (Uhr) and %→compose
        together with the other live stats. Uhr grows and shifts left as it goes s→m→h.
        """
        SEP = "  [dim]│[/dim]  "
        parts: list[str] = []

        # Time early so longer values (1h 05m etc) expand left and push rest right gracefully.
        display_t = self._last_task_s
        if display_t is None:
            display_t = time.monotonic() - self.session_start
        t_str = elapsed_str(display_t) if display_t and display_t > 0 else "0s"
        parts.append(f"[dim]⏱[/dim] [cyan]{t_str}[/cyan]")

        tok   = self.kratos_agent.session_usage
        total = tok.get("prompt_tokens", 0) + tok.get("completion_tokens", 0)
        parts.append(f"∑ [cyan]{total:,}[/cyan]")

        windows = role_context_windows(self.kratos_config)
        for role, abbr in (("planner", "P"), ("coder", "C"), ("verifier", "V")):
            used = self._ctx_live.get(role, (0, windows[role]))[0]
            window = windows[role]
            color = "red" if window and used / window > 0.8 else "yellow" if window and used / window > 0.6 else "green"
            parts.append(f"[{color}]{abbr} {_tok_short(used)}/{_tok_short(window)}[/]")

        coder_total  = windows["coder"]
        threshold    = float(getattr(self.kratos_config, "compress_threshold", 0.75))
        coder_used   = self._ctx_live.get("coder", (0, coder_total))[0]
        compose_at   = max(1, threshold * coder_total)
        pct          = min(100, int(coder_used * 100 / compose_at))
        bw           = 8
        filled       = min(bw, int(bw * pct / 100))
        bar          = "▓" * filled + "░" * (bw - filled)
        bc           = "red" if pct > 80 else "yellow" if pct > 50 else "bright_black"
        parts.append(f"[{bc}]{bar} {pct}%[/] [dim]→compose[/dim]")

        perm   = self.kratos_config.permission
        pc     = {"low": "yellow", "mid": "green", "high": "red"}.get(perm, "green")
        parts.append(f"[dim]{self.project_root.name}[/dim]  [{pc}]{perm}[/]")

        if self._busy:
            parts.append("[cyan blink]running…[/cyan blink]")

        return RichText.from_markup(SEP.join(parts))

    # ── Mounting helpers ──────────────────────────────────────────────────────

    def _log_mount(self, widget) -> None:
        """Mount a widget into the conversation log and scroll to bottom."""
        self.query_one("#log", VerticalScroll).mount(widget)
        self._scroll_log()

    def _log_markup(self, markup: str) -> None:
        """Append a single marked-up line to the log."""
        self._log_mount(Static(RichText.from_markup(markup)))

    def _mount_user_message(self, text: str) -> None:
        self._log_mount(UserMessage(text))

    def _turn_markup(self, markup: str) -> None:
        """Append a marked-up line to the current AssistantTurn (or log if none)."""
        widget = Static(RichText.from_markup(markup))
        self._turn_static(widget)

    def _turn_static(self, renderable) -> None:
        target = self._current_turn or self.query_one("#log", VerticalScroll)
        if isinstance(renderable, Static):
            target.mount(renderable)
        else:
            target.mount(Static(renderable))
        self._scroll_log()

    def _scroll_log(self) -> None:
        try:
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _capture(self, fn: "callable[[], None]") -> str:
        """Run fn() redirecting all kratos.ui console output; return ANSI string."""
        import kratos.ui as _ui
        import kratos.commands as _cmd
        buf = io.StringIO()
        cap = RichConsole(file=buf, force_terminal=True, width=self._term_width(),
                          legacy_windows=False, highlight=False)
        old_ui  = _ui.console
        old_cmd = getattr(_cmd, "console", None)
        _ui.console = cap
        if old_cmd is not None:
            _cmd.console = cap
        try:
            fn()
        finally:
            _ui.console = old_ui
            if old_cmd is not None:
                _cmd.console = old_cmd
        return buf.getvalue()

    def _term_width(self) -> int:
        try:
            return self.size.width or 120
        except Exception:
            return 120

    def _log_tool(self, content: str) -> None:
        logger = self.kratos_logger
        try:
            if "(" in content:
                name = content[:content.index("(")]
                args = content[content.index("(") + 1:content.rindex(")")]
                res  = (content.split("→")[-1].strip() if "→" in content else
                        content.split("->")[-1].strip() if "->" in content else "")
                logger.log_tool(name=name, args={"raw": args}, result=res)
            else:
                logger.log_tool(name=content, args={})
        except Exception:
            pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _tok_short(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_tui(
    config,
    bridge,
    agent,
    logger,
    project_root: Path,
    session_start: float,
    scope: str = "project",
) -> None:
    """Launch the Textual TUI. Blocks until the user exits."""
    app = KratosApp(
        config=config,
        bridge=bridge,
        agent=agent,
        logger=logger,
        project_root=project_root,
        session_start=session_start,
        scope=scope,
    )
    app.run()
