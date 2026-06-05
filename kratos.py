#!/usr/bin/env python3
"""Kratos — abliterated dual-model CLI AI agent.

Planner: huihui_ai/qwen3-abliterated:8b  (chain-of-thought planning, ~4.8 GB)
Coder:   NeuralDaredevil-8B-abliterated Q4_K_M  (code generation, ~4.7 GB)

Both models are abliterated (safety-filter removed).
Ollama runs natively on Windows; GPU offload via RTX 4050 Laptop (4 GB VRAM).

Usage:
    python kratos.py            # start interactive REPL
    python kratos.py --setup    # run model setup wizard
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable
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

from kratos.config import KratosConfig, GLOBAL_DIR
from kratos.bridge import OllamaBridge
from kratos.agent import KratosAgent
from kratos.commands import handle
from kratos.logger import SessionLogger
from kratos.ui import (
    console,
    print_banner, print_error, print_info, print_warn, print_success,
    planner_header, coder_header, verify_header, section_end,
    route_info, direct_header, tool_call,
)

_HISTORY_FILE = GLOBAL_DIR / "history.txt"
_PT_STYLE = PTStyle.from_dict({
    "prompt":                "ansicyan bold",
    "completion-menu.completion":          "bg:#1e2a35 #aaaaaa",
    "completion-menu.completion.current":  "bg:#0078d4 #ffffff bold",
    "completion-menu.meta.completion":     "bg:#1e2a35 #666666",
    "completion-menu.meta.completion.current": "bg:#005fa3 #cccccc",
}) if _HAS_PT else None

# ── slash-command completer ───────────────────────────────────────────────────
#
# Structure: { "cmd": ( {sub: description, ...} | None, "short description" ) }
_SLASH_TREE: dict[str, tuple[dict[str, str] | None, str]] = {
    "exit":       (None,                                          "Quit Kratos"),
    "quit":       (None,                                          "Quit Kratos"),
    "q":          (None,                                          "Quit Kratos"),
    "help":       (None,                                          "Show all commands"),
    "clear":      (None,                                          "Clear screen"),
    "status":     (None,                                          "Show status bar"),
    "setup":      (None,                                          "Model setup info"),
    "goal":       ({"clear": "Clear goal"},                       "Set or show goal"),
    "scope":      ({"global": "Machine-wide config",
                    "project": "Per-project config",
                    "info":    "Show paths"},                     "Config file scope"),
    "permission": ({"low":  "Read only",
                    "mid":  "Read + write  (default)",
                    "high": "Read + write + delete"},             "Coder permission level"),
    "models":     ({"planner": "Change planner model",
                    "coder":   "Change coder model"},             "Model config"),
    "index":      ({"rebuild": "Rescan project files"},           "Project file index"),
    "memory":     ({"list":    "Show all entries",
                    "clear":   "Clear session/project/all"},      "Persistent memory"),
    "history":    ({"clear": "Reset conversation"},               "Conversation history"),
    "build":      ({"clear": "Remove build command"},             "Build command for test loop"),
    "test":       ({"clear": "Remove test command"},              "Test command for diagnostic loop"),
    "logging":    ({"on": "Start logging session", "off": "Stop logging"}, "Session logging"),
}

if _HAS_PT:
    class _SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return

            # Split into command part and (optional) subcommand part
            after_slash = text[1:]
            space_idx = after_slash.find(" ")

            if space_idx == -1:
                # Still typing the command name  e.g. "/per"
                partial_cmd = after_slash.lower()
                for name, (_, desc) in sorted(_SLASH_TREE.items()):
                    if name.startswith(partial_cmd):
                        yield Completion(
                            "/" + name,
                            start_position=-len(text),
                            display=f"/{name}",
                            display_meta=desc,
                        )
            else:
                # Command complete, completing subcommand  e.g. "/permission m"
                cmd = after_slash[:space_idx].lower()
                partial_sub = after_slash[space_idx + 1:].lower()
                if cmd in _SLASH_TREE:
                    subcmds, _ = _SLASH_TREE[cmd]
                    if subcmds:
                        for sub, sub_desc in sorted(subcmds.items()):
                            if sub.startswith(partial_sub):
                                yield Completion(
                                    sub,
                                    start_position=-len(partial_sub),
                                    display=sub,
                                    display_meta=sub_desc,
                                )

    _COMPLETER: "Completer | None" = _SlashCompleter()
else:
    _COMPLETER = None


# ── input helper ─────────────────────────────────────────────────────────────

def _input(session: "PromptSession | None") -> str:
    if session and _HAS_PT:
        return session.prompt([("class:prompt", "kratos ❯ ")], style=_PT_STYLE)
    return input("kratos ❯ ")


# ── file operations (auto — no confirmation needed within project scope) ──────

def _apply_file_ops(
    agent: KratosAgent, project_root: "Path", logger: "SessionLogger | None" = None
) -> None:
    """Apply all file writes and deletes that the coder produced.

    Scope rule: operations are only executed within project_root.
    Anything referencing paths outside is skipped with a warning.
    No user confirmation is required — the user opened this directory as
    their project and the coder operates inside it by design.
    """
    from pathlib import Path as _Path

    changes = agent.pending_file_changes
    deletions = agent.pending_file_deletions

    if not changes and not deletions:
        return

    console.print()

    perm = agent.config.permission

    # ── writes ────────────────────────────────────────────────────────────────
    written = 0
    if changes:
        if not agent.config.can_write():
            print_warn(f"Write blocked (permission={perm}). Use /permission mid or high.")
        else:
            for rel_path, content in changes:
                target = (project_root / rel_path).resolve()
                try:
                    target.relative_to(project_root.resolve())
                except ValueError:
                    print_warn(f"Skipped write (outside project): {rel_path}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                size_str = f"{len(content.encode()) / 1024:.1f} KB"
                tool_call(f"write_file({rel_path!r}) → {size_str}  [written]")
                if logger:
                    logger.log_file_write(rel_path, content)
                written += 1

    # ── deletes ───────────────────────────────────────────────────────────────
    deleted = 0
    if deletions:
        if not agent.config.can_delete():
            print_warn(f"Delete blocked (permission={perm}). Use /permission high.")
        else:
            for rel_path in deletions:
                target = (project_root / rel_path).resolve()
                try:
                    target.relative_to(project_root.resolve())
                except ValueError:
                    print_warn(f"Skipped delete (outside project): {rel_path}")
                    continue
                if target.exists():
                    target.unlink()
                    tool_call(f"delete_file({rel_path!r})  [deleted]")
                    if logger:
                        logger.log_file_delete(rel_path)
                    deleted += 1
                else:
                    print_warn(f"Delete skipped (not found): {rel_path}")

    if written or deleted:
        console.print()


# ── coder output filter ───────────────────────────────────────────────────────

class _CoderFilter:
    """Line-buffer that shows only file-operations and the summary section.

    The actual code content is suppressed — only these markers are printed:
      ### FILE: path      → "Writing: path"
      ### DELETE: path    → "Deleting: path"
      ### SUMMARY         → show the following lines as summary
    """
    def __init__(self) -> None:
        self._buf = ""
        self._in_summary = False
        self._in_code_block = False
        self._seen: set[str] = set()

    def feed(self, token: str) -> None:
        self._buf += token
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._process_line(line)

    def flush(self) -> None:
        if self._buf.strip():
            self._process_line(self._buf)
        self._buf = ""

    def _process_line(self, line: str) -> None:
        stripped = line.strip()

        if stripped.startswith("### FILE:"):
            path = stripped[9:].strip()
            if path and path not in self._seen:
                self._seen.add(path)
                tool_call(f"write_file({path!r})")
            self._in_summary = False
            self._in_code_block = False
            return

        if stripped.startswith("### DELETE:"):
            path = stripped[11:].strip()
            if path:
                tool_call(f"delete_file({path!r})")
            self._in_summary = False
            self._in_code_block = False
            return

        if stripped.startswith("### SUMMARY"):
            self._in_summary = True
            self._in_code_block = False
            console.print()
            console.print("[dim]  Summary:[/dim]")
            return

        if stripped.startswith("```"):
            self._in_code_block = not self._in_code_block

        if self._in_summary and not self._in_code_block and stripped:
            console.print(f"  [dim]{stripped}[/dim]", highlight=False)


# ── agent streaming ───────────────────────────────────────────────────────────

def _stream_agent(
    agent: KratosAgent, task: str, logger: "SessionLogger"
) -> None:
    """Run intelligent pipeline and print streaming output with Rich UI."""

    coder_filter = _CoderFilter()
    current_section: str | None = None
    planner_buf = ""
    coder_buf = ""

    try:
        for source, content, kind in agent.process(task):

            if source == "log":
                # Internal log event from agent — write to file only, no display
                if logger.enabled:
                    try:
                        import json as _json
                        data = _json.loads(content)
                        event_type = data.pop("type", "unknown")
                        logger._write(event_type, **data)
                    except Exception:
                        pass

            elif source == "router":
                route_info(content)
                parts = dict(p.split("=", 1) for p in content.split("  ") if "=" in p)
                logger.log_route(
                    intent=parts.get("intent", ""),
                    route=parts.get("route", ""),
                )

            elif source == "tool":
                tool_call(content)
                # Parse name(args) format for structured logging
                if "(" in content:
                    name = content[:content.index("(")]
                    args_str = content[content.index("(")+1:content.rindex(")")]
                    logger.log_tool(name=name, args={"raw": args_str},
                                    result=content.split("→")[-1].strip() if "→" in content else "")
                else:
                    logger.log_tool(name=content, args={})

            elif source == "header":
                current_section = content
                if content == "planner":
                    planner_header(agent.config.planner_model)
                elif content == "coder":
                    coder_header(agent.config.coder_model)
                elif content == "verify":
                    verify_header(agent.config.planner_model)

            elif source == "planner":
                if kind == "think":
                    console.print(content, end="", style="dim italic", highlight=False)
                else:
                    planner_buf += content
                    console.print(content, end="", highlight=False)

            elif source == "verify":
                # Show verifier output — it's short (VERIFIED / NEEDS_REVISION)
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
                direct_header()
                console.print(content, highlight=False)
                section_end()

            elif source == "end":
                if current_section == "planner" and planner_buf:
                    logger.log_model_output("planner", agent.config.planner_model, planner_buf)
                    planner_buf = ""
                elif current_section == "coder":
                    coder_filter.flush()
                    if coder_buf:
                        logger.log_model_output("coder", agent.config.coder_model, coder_buf)
                    coder_buf = ""
                elif current_section == "verify":
                    console.print()   # newline after VERIFIED / NEEDS_REVISION
                section_end()
                current_section = None

            elif source == "info":
                print_info(content)
                logger.log_info(content)

            elif source == "warn":
                print_warn(content)

            elif source == "error":
                print_error(content)
                logger.log_error(content)

            elif source == "question":
                console.print()
                console.print(f"[cyan]Kratos:[/cyan] {content}")

    except KeyboardInterrupt:
        console.print()
        coder_filter.flush()
        print_warn("Interrupted.")
        return

    # Auto-apply all file operations after streaming completes
    _apply_file_ops(agent, agent.indexer.root, logger)


# ── startup checks ────────────────────────────────────────────────────────────

def _ensure_ready(bridge: OllamaBridge, config: KratosConfig) -> bool:
    print_info("Connecting to Ollama (WSL + CUDA)…")
    if bridge.is_running():
        print_success("Ollama ready.")
    else:
        print_info("Starting Ollama in WSL…")
        if bridge.start():
            print_success("Ollama started.")
        else:
            print_warn(
                "Cannot reach Ollama. Start it manually:\n"
                "  wsl -e bash -c 'OLLAMA_HOST=0.0.0.0 ollama serve &'\n"
                "Then re-run Kratos."
            )
            return False

    missing = []
    if not bridge.model_exists(config.planner_model):
        missing.append(f"[cyan]{config.planner_model}[/cyan] (planner)")
    if not bridge.model_exists(config.coder_model):
        missing.append(f"[green]{config.coder_model}[/green] (coder)")
    if missing:
        print_warn("Missing models: " + ", ".join(missing))
        print_info("Run [cyan]python setup_models.py[/cyan] to install them.")
        return False
    return True


# ── main REPL ─────────────────────────────────────────────────────────────────

def main() -> None:
    config = KratosConfig.load()
    scope = config.scope or "project"

    bridge = OllamaBridge(config.ollama_host)
    agent = KratosAgent(config, bridge)

    from pathlib import Path as _Path
    from kratos.config import _project_dir
    project_root = _Path.cwd()
    logger = SessionLogger(_project_dir())

    print_banner(config.planner_model, config.coder_model, scope, config.permission)
    print_info(f"Project root: [bold]{project_root}[/bold]")
    _ensure_ready(bridge, config)
    console.print()
    print_info("Enter your task, or [cyan]/help[/cyan] for commands.  [dim]/exit[/dim] to quit.")
    console.print()

    # Setup prompt_toolkit session with history + slash-command autocomplete
    session = None
    if _HAS_PT:
        GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
        session = PromptSession(
            history=FileHistory(str(_HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=_COMPLETER,
            complete_while_typing=True,  # dropdown opens the moment "/" is typed
        )

    while True:
        try:
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
            continue

        # Normal task → run agent pipeline
        logger.log_input(line)
        try:
            _stream_agent(agent, line, logger)
        except Exception as exc:
            print_error(f"Agent error: {escape(str(exc))}")
            logger.log_error(str(exc))


if __name__ == "__main__":
    if "--setup" in sys.argv:
        import setup_models as sm
        sm.setup()
    else:
        main()
