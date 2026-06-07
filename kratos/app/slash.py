"""Shared slash-command metadata and autocomplete — used by both the classic
REPL (``app/cli.py``) and the TUI (``app/tui.py``) so the command tree lives
in exactly one place.
"""

from __future__ import annotations

try:
    from prompt_toolkit.completion import Completer, Completion
    _HAS_PT = True
except ImportError:
    _HAS_PT = False

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


def slash_completions(text: str) -> list[tuple[str, str, str]]:
    """Return (value, display, meta) completion tuples for a partial slash-command text.

    `text` must start with "/". Used by the TUI autocomplete overlay and in tests.
    """
    if not text.startswith("/"):
        return []
    after = text[1:]
    space = after.find(" ")
    if space == -1:
        partial = after.lower()
        return [
            ("/" + name, "/" + name, desc)
            for name, (_, desc) in sorted(_SLASH_TREE.items())
            if name.startswith(partial)
        ]
    cmd = after[:space].lower()
    partial_sub = after[space + 1:].lower()
    if cmd in _SLASH_TREE:
        subs, _ = _SLASH_TREE[cmd]
        if subs:
            return [
                (sub, sub, desc)
                for sub, desc in sorted(subs.items())
                if sub.startswith(partial_sub)
            ]
    return []


# ── prompt_toolkit completer for the classic REPL ────────────────────────────

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

    def make_completer() -> "Completer | None":
        return _SlashCompleter()
else:
    def make_completer() -> "Completer | None":
        return None
