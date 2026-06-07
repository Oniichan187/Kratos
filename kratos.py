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
    from rich.live import Live
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
from kratos.tokens import role_context_windows
from kratos.ui import (
    console,
    print_banner, print_error, print_info, print_warn, print_success,
    print_usage,
    user_message_panel, task_summary_panel,
    section_banner, elapsed_str,
    status_bar,
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


# ── prompt input + slash-command completion ──────────────────────────────────

if _HAS_PT:
    class _BottomPanel:
        """Non-fullscreen prompt_toolkit Application — framed input + live info bar.

        Terminal layout while waiting for input (input expands up to 5 lines):
          ╭────────────────────────────────────────────────────────╮
          │  ⏱ 3m  ∑ 8k  ░░░░░░░░ 0%→compose  project  mid       │
          ├────────────────────────────────────────────────────────┤
          │  kratos ❯ line 1                                       │
          │           line 2  (Alt+Enter to insert newline)        │
          ╰────────────────────────────────────────────────────────╯
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
                multiline=True,
            )

        def prompt(self) -> str:
            """Block until Enter; return stripped text. Raises KeyboardInterrupt / EOFError."""
            import shutil
            from prompt_toolkit.application import Application as _App
            from prompt_toolkit.layout import Layout as _Lay
            from prompt_toolkit.layout.containers import (
                HSplit as _HS, VSplit as _VS, Window as _W,
                FloatContainer as _FC, Float as _Fl,
            )
            from prompt_toolkit.layout.controls import (
                BufferControl as _BC,
                FormattedTextControl as _FTC,
            )

            from prompt_toolkit.layout.menus import CompletionsMenu as _CM
            from prompt_toolkit.key_binding import KeyBindings as _KB
            from prompt_toolkit.key_binding.defaults import load_key_bindings as _lkb
            from prompt_toolkit.key_binding import merge_key_bindings as _mkb
            from prompt_toolkit.styles import Style as _Sty

            buf = self._buf

            _BORDER = "│  "
            _LABEL  = "kratos ❯ "
            _PREFIX_LEN = len(_BORDER) + len(_LABEL)

            def _cols() -> int:
                return shutil.get_terminal_size((80, 24)).columns

            def _line_prefix(line_number: int, wrap_count: int) -> list[tuple[str, str]]:
                """Left border on every visual row — keeps the frame intact when wrapping."""
                if line_number == 0 and wrap_count == 0:
                    return [("class:bdr", _BORDER), ("class:prompt", _LABEL)]
                return [("class:bdr", _BORDER), ("", " " * len(_LABEL))]

            def _content_width() -> int:
                return max(1, _cols() - _PREFIX_LEN - 1)  # -1 reserves the scrollbar column

            def _wrapped_rows(line: str, width: int) -> int:
                return max(1, -(-len(line) // width))

            def _total_rows() -> int:
                w = _content_width()
                return sum(_wrapped_rows(ln, w) for ln in buf.document.lines)

            def _input_height() -> int:
                """Visual row count after line-wrapping, capped at 5 (then scrolls)."""
                return max(1, min(_total_rows(), 5))

            def _cursor_visual_row() -> int:
                """Visual row of the cursor — used to position the scrollbar thumb."""
                w = _content_width()
                doc = buf.document
                row = 0
                for i, ln in enumerate(doc.lines):
                    if i == doc.cursor_position_row:
                        return row + doc.cursor_position_col // w
                    row += _wrapped_rows(ln, w)
                return row

            def _visual_rows_map() -> list[tuple[int, int]]:
                """(start, end) buffer-offset of each visually-wrapped row, across all lines."""
                w = _content_width()
                rows: list[tuple[int, int]] = []
                offset = 0
                for ln in buf.document.lines:
                    n = len(ln)
                    if n == 0:
                        rows.append((offset, offset))
                    else:
                        i = 0
                        while i < n:
                            seg_len = min(w, n - i)
                            rows.append((offset + i, offset + i + seg_len))
                            i += seg_len
                    offset += n + 1  # +1 for the '\n' separator
                return rows

            def _move_visual_row(delta: int) -> bool:
                """Move the cursor up/down one *wrapped* row, preserving its column.

                Returns False at the first/last visual row so the caller can fall
                back to history navigation (matches Buffer.auto_up/auto_down, which
                only do this correctly for logical — i.e. unwrapped — lines).
                """
                rows = _visual_rows_map()
                pos = buf.cursor_position
                cur = next(i for i, (s, e) in enumerate(rows) if pos <= e)
                target = cur + delta
                if target < 0 or target >= len(rows):
                    return False
                col = pos - rows[cur][0]
                t_start, t_end = rows[target]
                buf.cursor_position = min(t_start + col, t_end)
                return True

            def _scrollbar() -> list[tuple[str, str]]:
                """Right-edge slider — '│' tinted where the thumb covers the visible window."""
                displayed = _input_height()
                total = max(displayed, _total_rows())
                if total <= displayed:
                    return [("class:bdr", "│\n" * (displayed - 1) + "│")]
                thumb = max(1, round(displayed * displayed / total))
                span = displayed - thumb
                top = int(_cursor_visual_row() / max(1, total - 1) * span + 0.5)
                parts: list[tuple[str, str]] = []
                for i in range(displayed):
                    style = "class:sbar-thumb" if top <= i < top + thumb else "class:bdr"
                    parts.append((style, "│"))
                    if i < displayed - 1:
                        parts.append(("", "\n"))
                return parts

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

            @kb.add("escape", "enter")   # Alt+Enter — insert literal newline
            def _alt_enter(event) -> None:
                buf.insert_text("\n")

            @kb.add("up")    # scroll within wrapped text; only switch history at the very top
            def _up(event) -> None:
                if not _move_visual_row(-1):
                    buf.history_backward()

            @kb.add("down")  # scroll within wrapped text; only switch history at the very bottom
            def _down(event) -> None:
                if not _move_visual_row(1):
                    buf.history_forward()

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
                _FC(
                    content=_HS([
                        _W(_FTC(_top), height=1, dont_extend_height=True),
                        _W(_FTC(_info), height=1, dont_extend_height=True),
                        _W(_FTC(_mid), height=1, dont_extend_height=True),
                        _VS(
                            [
                                _W(
                                    _BC(buffer=buf),
                                    wrap_lines=True,
                                    get_line_prefix=_line_prefix,
                                ),
                                _W(_FTC(_scrollbar), width=1, dont_extend_width=True),
                            ],
                            # Grows 1→5 rows with content (incl. line-wrap); right edge becomes
                            # a slider (tinted '│' segment) once content overflows and scrolls.
                            height=_input_height,
                        ),
                        _W(_FTC(_bot), height=1, dont_extend_height=True),
                    ]),
                    floats=[
                        _Fl(
                            xcursor=True, ycursor=True,
                            content=_CM(max_height=8, scroll_offset=2),
                        ),
                    ],
                ),
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
                "sbar-thumb":                               "ansicyan bold",
            })

            # Save cursor position, then push it to bottom with plain newlines
            # (plain \n always scrolls; ANSI cursor sequences are unreliable on ConPTY).
            sys.stdout.write("\033[s")
            _rows = shutil.get_terminal_size((80, 24)).lines
            # Reserve 9 lines for the full frame (top+info+mid+up to 5×input+bot).
            sys.stdout.write("\n" * max(0, (_rows - 9) // 2))
            sys.stdout.flush()

            app = _App(
                layout=layout,
                key_bindings=_mkb([_lkb(), kb]),
                style=style,
                erase_when_done=True,
                full_screen=False,
                mouse_support=False,
            )
            result = app.run()

            # Restore cursor to pre-push position and clear to end of screen.
            sys.stdout.write("\033[u\033[0J")
            sys.stdout.flush()
            return (result or "").strip()

else:
    class _BottomPanel:  # type: ignore[no-redef]
        def __init__(self, get_info, history_path=None, completer=None):
            self._get_info = get_info

        def prompt(self) -> str:
            return input("kratos ❯ ").strip()


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

class _LineFilter:
    """Buffer streamed chunks and emit complete lines only.

    Per-token `console.print(..., end="")` cannot coexist with a pinned
    bottom region (rich.Live or prompt_toolkit alike — both redraw the live
    area on every write, and a redraw mid-partial-line mangles the stream
    and injects stray control characters). Buffering to whole lines — the
    same approach `_CoderFilter` already uses for coder output — sidesteps
    that entirely: each `console.print(line, ...)` is a complete, atomic
    write that Live can safely interleave with its own redraws.
    """

    def __init__(self, style: str = "") -> None:
        self._buf = ""
        self._style = style

    def feed(self, chunk: str, style: str = "") -> None:
        if style != self._style:
            self.flush()
            self._style = style
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            console.print(line, style=self._style or None, highlight=False)

    def flush(self) -> None:
        if self._buf:
            console.print(self._buf, style=self._style or None, highlight=False)
            self._buf = ""


class _LiveBar:
    """Renderable wrapper so `rich.Live` re-evaluates the status bar on every
    refresh tick — the idiomatic way to keep a `Live` display showing
    *current* values rather than a snapshot taken at construction time.
    """

    def __init__(self, build: "Callable[[], object]") -> None:
        self._build = build

    def __rich__(self) -> object:
        return self._build()


def _stream_agent(
    agent: KratosAgent, task: str, logger: "SessionLogger",
    _ctx_state: "dict | None" = None,
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
    planner_filter = _LineFilter()
    verify_filter = _LineFilter()
    coder_think_filter = _LineFilter(style="dim italic")

    # Running completion-token estimate — Ollama only reports real counts at
    # the *end* of each model call (the "usage"/"ctx_info" events), so without
    # this the bar's ∑ and "%→compose" numbers would sit frozen mid-stream.
    # ~4 chars/token is the standard rough estimate; good enough for a live
    # indicator that the real end-of-call numbers immediately correct.
    running_completion = 0
    session_completion_base = 0

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    _ROLE_BY_SOURCE = {"planner": "planner", "coder": "coder", "verify": "verifier"}

    def _build_bar():
        role = _ROLE_BY_SOURCE.get(current_section, "")
        state = dict(ctx_live)
        if role and running_completion:
            used, total = state.get(role, (0, 0))
            state[role] = (used + running_completion, total)
        usage = agent.session_usage
        sess_tok = (
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0) + session_completion_base + running_completion,
        )
        coder_used, coder_total = state.get("coder", (0, windows["coder"]))
        threshold = float(getattr(agent.config, "compress_threshold", 0.75))
        compose_at = max(1, threshold * coder_total)
        compose_pct = min(100, int(coder_used * 100 / compose_at))
        return status_bar(
            scope=agent.config.scope or "project",
            permission=agent.config.permission,
            ctx_state=state,
            elapsed_s=time.monotonic() - task_start,
            project_name=agent._indexer.root.name,
            current_section=role,
            session_tokens=sess_tok,
            goal=agent.config.goal,
            hint="Ctrl+C to stop",
            compose_pct=compose_pct,
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
                session_completion_base += running_completion
                running_completion = 0
                current_section = content
                section_start = time.monotonic()
                # Dedup across stepwise re-entries too: print the role banner only when
                # the role changes (e.g. coder -> verify -> coder), not on every step's
                # repeated "coder" header — the "Step N/M" info line is the per-step marker.
                if content != last_banner_section:
                    last_banner_section = content
                    console.print()
                    console.print(section_banner(content, _smodel(content)))

            elif source == "planner":
                running_completion += _est_tokens(content)
                style = "dim italic" if kind == "think" else ""
                planner_filter.feed(content, style=style)
                if kind != "think":
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
                    d = _j.loads(content); logger._write("token_usage", **d)
                except Exception:
                    pass

            elif source == "end":
                sec_elapsed = time.monotonic() - section_start
                if current_section == "planner":
                    planner_filter.flush()
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
                session_completion_base += running_completion
                running_completion = 0
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
    finally:
        planner_filter.flush()
        verify_filter.flush()
        coder_think_filter.flush()
        live.stop()

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

    # Live ctx tracking: updated by _stream_agent as model calls report token counts.
    _windows = role_context_windows(config)
    _ctx_live: dict[str, tuple[int, int]] = {
        "planner": (0, _windows["planner"]),
        "coder":   (0, _windows["coder"]),
        "verifier": (0, _windows["verifier"]),
    }
    _last_task_s: float | None = None

    # ── live info content for the panel frame ─────────────────────────────────
    def _make_toolbar() -> list[tuple[str, str]]:
        """Builds prompt_toolkit bottom-toolbar: ⏱ (grows left on m/h) + tokens + ctx + %→compose + project + perm.
        Both % and the Uhr/time are always shown together with the other live stats.
        """
        SEP = "   │   "
        out: list[tuple[str, str]] = [("", "  ")]

        # Time (Uhr) early in the line: its string growth (s→m→h) makes the clock "move further left"
        # and pushes the following stats right in a controlled way inside the frame.
        _time_val = _last_task_s if _last_task_s is not None else (time.time() - _session_start)
        _t_str = elapsed_str(_time_val) if _time_val and _time_val > 0 else "0s"
        out += [("", "⏱ "), ("ansiyellow", _t_str), ("", SEP)]

        # Cumulative token usage — prefer Ollama eval counts; fall back to ctx_info sizes
        _tok = agent.session_usage
        _total = _tok.get("prompt_tokens", 0) + _tok.get("completion_tokens", 0)
        tok_str = "--" if _total == 0 else f"{_total:,}"
        out += [("", "∑ "), ("ansicyan", tok_str), ("", SEP)]

        _windows_now = role_context_windows(config)
        for _role, _abbr in (("planner", "P"), ("coder", "C"), ("verifier", "V")):
            _used = _ctx_live.get(_role, (0, _windows_now[_role]))[0]
            out += [("", f"{_abbr} {_used // 1024}k/{_windows_now[_role] // 1024}k"), ("", SEP)]

        # % until auto-compose (always present together with the time)
        _coder_total = _windows_now["coder"]
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
    elif "--tui" in sys.argv:
        import time as _t
        from pathlib import Path as _Path
        from kratos.config import KratosConfig as _KC, _project_dir as _pd
        from kratos.bridge import OllamaBridge as _OB
        from kratos.agent import KratosAgent as _KA
        from kratos.logger import SessionLogger as _SL
        from kratos.prompts import load_prompts as _lp
        from kratos.tui import run_tui as _run_tui
        _cfg    = _KC.load()
        _scope  = _cfg.scope or "project"
        _bridge = _OB(_cfg.ollama_host)
        _prm    = _lp()
        _agent  = _KA(_cfg, _bridge, prompts=_prm)
        _logger = _SL(_pd())
        _run_tui(
            config=_cfg,
            bridge=_bridge,
            agent=_agent,
            logger=_logger,
            project_root=_Path.cwd(),
            session_start=_t.time(),
            scope=_scope,
        )
    else:
        main()
