"""Classic-REPL input frame and stream-rendering helpers.

``_BottomPanel`` is the framed, non-fullscreen prompt_toolkit ``Application``
that draws the bordered input box + live info bar (falls back to plain
``input()`` when prompt_toolkit isn't installed). ``_PlannerFilter``,
``_CoderFilter``, ``_LineFilter`` and ``_LiveBar`` buffer/format the agent's
streamed output for ``console.print`` + ``rich.Live`` in ``app/cli.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from kratos.planning import parse_execution_plan, render_checklist
from kratos.ui import console

try:
    from prompt_toolkit.completion import Completer
    _HAS_PT = True
except ImportError:
    _HAS_PT = False


# ── prompt input + slash-command completion ──────────────────────────────────

if _HAS_PT:
    class _BottomPanel:
        """Non-fullscreen prompt_toolkit Application — framed input + live info bar.

        Terminal layout while waiting for input (input expands up to 5 lines):
          ╭────────────────────────────────────────────────────────╮
          │  ⏱ 3m  P 2k/40k  │  C 11k/64k  │  V 3k/40k  project  mid │
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


# ── coder output filter ───────────────────────────────────────────────────────

class _PlannerFilter:
    """Buffer planner output and render only the short checklist on flush."""

    def __init__(self, emit: Callable[[str, str], None] | None = None) -> None:
        self._buf: str = ""
        self._emit = emit or (lambda text, style="": console.print(text, style=style, highlight=False))

    def feed(self, token: str, style: str = "") -> None:
        del style
        self._buf += token

    def flush(self) -> None:
        text = self._buf.strip()
        self._buf = ""
        if not text:
            return
        plan = parse_execution_plan(text)
        checklist = render_checklist(plan.items, compact=True)
        if checklist.strip():
            heading = "PLAN CHECKLIST"
            try:
                from kratos.prompts import get_snippet
                heading = get_snippet("planner_checklist_heading") or heading
            except Exception:
                pass
            self._emit(heading, "bold")
            for line in checklist.splitlines():
                self._emit(f"  {line}", "")
        else:
            self._emit("PLAN CHECKLIST: (no checklist parsed)", "dim")


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
            # Do NOT echo the file marker here. The authoritative line is the
            # actual do_write tool event ("write_file('x') -> N bytes (+a/-r lines)")
            # which the stream prints once the write lands on disk. Echoing the
            # marker too produced the duplicate the user reported:
            #   ↳ write_file('numstats.py')
            #   ↳ WRITE  write_file('numstats.py') -> 404 bytes (-5 +0 lines)
            # Keep only the state updates so the following code body stays hidden.
            path = s[len(fm):].strip()
            if path:
                self._seen.add(path)
            self._in_summary = False
            self._in_code = False
        elif s.startswith(dm):
            # Same as above — the do_delete tool event ("delete_file('x') -> ok")
            # is the single authoritative line; don't pre-echo the marker.
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
        else:
            # Print coder action commands (### VERIFY / RUN / READ / INSPECT / DONE) so they are visible
            # to the user ("Befehle wirklich gedruckt"). Use tolerant check against current markers.
            try:
                from kratos.prompts import get_marker
                vm = (get_marker("verify") or "### VERIFY:").rstrip(":") + ":"
                rm = (get_marker("run") or "### RUN:").rstrip(":") + ":"
                im = (get_marker("inspect") or "### INSPECT:").rstrip(":") + ":"
                rdm = (get_marker("read") or "### READ:").rstrip(":") + ":"
                donem = (get_marker("done") or "### DONE").rstrip(":")
            except Exception:
                vm, rm, im, rdm, donem = "### VERIFY:", "### RUN:", "### INSPECT:", "### READ:", "### DONE"
            low = s.lower()
            if low.startswith(vm.lower()) or low.startswith(rm.lower()):
                console.print(f"  [cyan]↳[/cyan] {s}")
            elif low.startswith(im.lower()) or low.startswith(rdm.lower()):
                console.print(f"  [dim cyan]↳[/dim cyan] {s}")
            elif low.startswith(donem.lower()):
                console.print(f"  [green]↳[/green] {s}")


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
