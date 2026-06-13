"""Slash command handlers for the Kratos REPL.

``handle(line, config, scope)`` parses one slash-command line and returns
``(config, scope, signal)`` where signal is one of:

  None            → continue normally
  "exit"          → exit the REPL
  "clear_history" → caller should call agent.clear_history()
  "clear_screen"  → caller should call console.clear()

Config changes are persisted immediately via ``config.save(scope)``.
"""

from __future__ import annotations

from pathlib import Path
from .config import KratosConfig, GLOBAL_DIR, _project_dir
from .llm.tokens import role_context_windows
from .ui import (
    console,
    print_error, print_info, print_success, print_warn, print_help,
    show_permission_level, show_models, refresh_status,
)


def _ctx_display(config: KratosConfig) -> dict[str, int]:
    d = role_context_windows(config)
    if getattr(config, "always_max_ctx", True):
        d["max_policy"] = 1  # signal in show
    return d


def handle(
    line: str,
    config: KratosConfig,
    scope: str,
    agent=None,    # KratosAgent | None
    logger=None,   # SessionLogger | None
) -> tuple[KratosConfig, str, str | None]:
    """Dispatch a slash command.

    Returns (config, scope, signal) where signal is one of:
        None            → continue normally
        "exit"          → exit the REPL
        "clear_history" → clear agent history
        "clear_screen"  → clear terminal
    """
    tokens = line.strip().lstrip("/").split(maxsplit=3)
    if not tokens:
        return config, scope, None
    cmd = tokens[0].lower()
    args = tokens[1:]

    # ── exit ─────────────────────────────────────────────────────────────────
    if cmd in ("exit", "quit", "q"):
        return config, scope, "exit"

    # ── help ─────────────────────────────────────────────────────────────────
    elif cmd == "help":
        print_help()

    # ── clear ─────────────────────────────────────────────────────────────────
    elif cmd == "clear":
        return config, scope, "clear_screen"

    # ── goal ──────────────────────────────────────────────────────────────────
    elif cmd == "goal":
        if not args:
            if config.goal:
                print_info(f"Current goal: [yellow]{config.goal}[/yellow]")
            else:
                print_info("No goal set. Use [cyan]/goal <text>[/cyan].")
        elif args[0] == "clear":
            config.goal = None
            config.save(scope)
            print_success("Goal cleared.")
        else:
            config.goal = " ".join(args)
            config.save(scope)
            print_success(f"Goal set: [yellow]{config.goal}[/yellow]")

    # ── scope ─────────────────────────────────────────────────────────────────
    elif cmd == "scope":
        if not args or args[0] == "info":
            print_info(f"Current scope: [bold]{scope}[/bold]")
            print_info(f"  [cyan]global[/cyan]   → {GLOBAL_DIR / 'config.json'}")
            print_info(f"  [cyan]project[/cyan]  → {_project_dir() / 'config.json'}")
        elif args[0] in ("global", "project"):
            scope = args[0]
            print_success(f"Scope → [bold]{scope}[/bold]")
        else:
            print_error(f"Unknown scope '{args[0]}'. Use: global | project")

    # ── permission ────────────────────────────────────────────────────────────
    elif cmd == "permission":
        if not args:
            show_permission_level(config.permission)
        else:
            level = args[0].lower()
            if level in ("low", "mid", "high"):
                config.permission = level  # type: ignore[assignment]
                config.save(scope)
                show_permission_level(config.permission)
            else:
                print_error("Usage: /permission [low|mid|high]")

    # ── models ────────────────────────────────────────────────────────────────
    elif cmd == "models":
        if not args:
            show_models(
                config.planner_model,
                config.coder_model,
                verifier=config.verifier_model,
                compressor=config.compressor_model,
                ctx=_ctx_display(config),
            )
        elif args[0] == "planner" and len(args) >= 2:
            config.planner_model = args[1]
            config.save(scope)
            print_success(f"Planner → [cyan]{config.planner_model}[/cyan]")
            refresh_status(config.planner_model, config.coder_model, scope, config.permission, config.goal, verifier=config.verifier_model, compressor=config.compressor_model, ctx=_ctx_display(config))
        elif args[0] == "coder" and len(args) >= 2:
            config.coder_model = args[1]
            config.save(scope)
            print_success(f"Coder → [green]{config.coder_model}[/green]")
            refresh_status(config.planner_model, config.coder_model, scope, config.permission, config.goal, verifier=config.verifier_model, compressor=config.compressor_model, ctx=_ctx_display(config))
        elif args[0] == "verifier" and len(args) >= 2:
            config.verifier_model = args[1]
            config.save(scope)
            print_success(f"Verifier → [yellow]{config.verifier_model}[/yellow]")
            refresh_status(config.planner_model, config.coder_model, scope, config.permission, config.goal, verifier=config.verifier_model, compressor=config.compressor_model, ctx=_ctx_display(config))
        elif args[0] == "compressor" and len(args) >= 2:
            config.compressor_model = args[1]
            config.save(scope)
            print_success(f"Compressor → [magenta]{config.compressor_model}[/magenta]")
            refresh_status(config.planner_model, config.coder_model, scope, config.permission, config.goal, verifier=config.verifier_model, compressor=config.compressor_model, ctx=_ctx_display(config))
        else:
            print_error("Usage: /models  |  /models planner|coder|verifier|compressor <name>")

    # ── prompts (JSON externalized systems + snippets; /prompts reload after edit) ─
    elif cmd == "prompts":
        from .prompts import load_prompts, reload_prompts
        pm = load_prompts()
        if not args or args[0] in ("list", "show"):
            from .ui import show_prompts
            show_prompts(pm)
            print_info("Override files checked: ~/.kratos/prompts.json then ./.kratos/prompts.json")
            print_info("Use /prompts dump or /prompts reload")
        elif args[0] == "reload":
            reload_prompts()
            if agent is not None:
                # Rebind so this agent instance sees the fresh manager (with cached overrides)
                try:
                    agent.prompts = load_prompts()
                except Exception:
                    pass
            print_success("Prompts reloaded from disk (next LLM calls will use updates).")
        elif args[0] == "dump":
            target = args[1] if len(args) > 1 else ".kratos/prompts.json"
            from pathlib import Path as _P
            out = pm.dump_defaults(_P(target))
            print_success(f"Dumped defaults to {out} — edit and /prompts reload")
        else:
            print_error("Usage: /prompts [list|reload|dump [path]]")

    # ── status ────────────────────────────────────────────────────────────────
    elif cmd == "status":
        refresh_status(
            config.planner_model,
            config.coder_model,
            scope,
            config.permission,
            config.goal,
            verifier=config.verifier_model,
            compressor=config.compressor_model,
            ctx=_ctx_display(config),
        )

    # ── history ───────────────────────────────────────────────────────────────
    elif cmd == "history":
        if args and args[0] == "clear":
            return config, scope, "clear_history"
        print_info("Use [cyan]/history clear[/cyan] to reset conversation context.")

    # ── setup ─────────────────────────────────────────────────────────────────
    elif cmd == "setup":
        print_info("Run: [cyan]python setup_models.py[/cyan]")
        print_info("Or in WSL: [cyan]bash setup_wsl.sh[/cyan]  to install Ollama with CUDA.")

    # ── index / knowledge (vector DB + continuous "gets") ─────────────────────
    elif cmd == "index":
        if agent is None:
            print_warn("Agent not available.")
        elif args and args[0] == "rebuild":
            n = agent.rebuild_index()
            print_success(f"Index rebuilt: {n} files.")
        else:
            index = agent.indexer.index
            from rich.table import Table
            from rich import box as _box
            table = Table(box=_box.SIMPLE, show_header=True)
            table.add_column("File", style="cyan")
            table.add_column("Pri", width=4)
            for e in index[:30]:
                table.add_row(e.rel_path, str(e.priority))
            if len(index) > 30:
                table.add_row(f"… and {len(index) - 30} more", "")
            console.print(table)

    # ── knowledge (the new vector DB / continuous retrieval surface) ──────────
    elif cmd == "knowledge":
        if agent is None or agent.knowledge is None:
            print_warn("Knowledge base not available (no embed model or lancedb?). Falling back to classic index/memory.")
        elif not args or args[0] in ("status", "info"):
            st = agent.knowledge.status()
            print_info(f"Knowledge base: backend={st.get('backend')}  chunks={st.get('chunks')}  embed_model={st.get('embed_model')}")
            print_info(f"  location: {st.get('kb_dir')}")
        elif args[0] == "rebuild":
            force = "force" in " ".join(args[1:]).lower()
            n = agent.rebuild_knowledge(force=force)
            print_success(f"Knowledge base rebuilt: {n} chunks (embed model: {getattr(agent.config, 'embed_model', '?')}).")
        else:
            print_info("Usage: /knowledge [status|rebuild [force]]")

    # ── memory ────────────────────────────────────────────────────────────────
    elif cmd == "memory":
        if agent is None:
            print_warn("Agent not available.")
        else:
            sub = args[0].lower() if args else "list"
            if sub == "list":
                all_mem = agent.memory.list_all()
                from rich.table import Table
                from rich import box as _box
                table = Table(box=_box.SIMPLE, show_header=True)
                table.add_column("Tier", width=10)
                table.add_column("Category", width=12)
                table.add_column("Content")
                for tier, entries in all_mem.items():
                    for e in entries:
                        table.add_row(tier, e.category, e.content[:80])
                if not any(all_mem.values()):
                    print_info("Memory is empty.")
                else:
                    console.print(table)
            elif sub == "clear":
                target = args[1].lower() if len(args) >= 2 else "session"
                if target == "all":
                    agent.memory.clear_task()
                    agent.memory.clear_session()
                    agent.memory.clear_project()
                    print_success("All memory cleared.")
                elif target == "project":
                    agent.memory.clear_project()
                    print_success("Project memory cleared.")
                elif target == "session":
                    agent.memory.clear_session()
                    agent.memory.clear_task()
                    print_success("Session/task memory cleared.")
                else:
                    print_error("Usage: /memory clear [session|project|all]")
            else:
                print_error("Usage: /memory [list|clear [session|project|all]]")

    # ── build / test commands ─────────────────────────────────────────────────
    elif cmd == "build":
        if not args:
            val = config.build_cmd or "(not set)"
            print_info(f"build_cmd: [cyan]{val}[/cyan]")
        elif args[0] == "clear":
            config.build_cmd = None
            config.save(scope)
            print_success("build_cmd cleared.")
        else:
            config.build_cmd = " ".join(args)
            config.save(scope)
            print_success(f"build_cmd → [cyan]{config.build_cmd}[/cyan]")

    elif cmd == "test":
        if not args:
            val = config.test_cmd or "(not set)"
            print_info(f"test_cmd: [cyan]{val}[/cyan]")
        elif args[0] == "clear":
            config.test_cmd = None
            config.save(scope)
            print_success("test_cmd cleared.")
        else:
            config.test_cmd = " ".join(args)
            config.save(scope)
            print_success(f"test_cmd → [cyan]{config.test_cmd}[/cyan]")

    # ── logging ───────────────────────────────────────────────────────────────
    elif cmd == "logging":
        if logger is None:
            print_warn("Logger not available.")
        else:
            sub = args[0].lower() if args else "status"
            if sub == "on":
                if not logger.enabled:
                    path = logger.enable()
                    print_success(f"Logging enabled → [cyan]{path}[/cyan]")
                else:
                    print_info(f"Already logging to [cyan]{logger.log_path}[/cyan]")
            elif sub == "off":
                if logger.enabled:
                    logger.disable()
                    print_success("Logging disabled.")
                else:
                    print_info("Logging is already off.")
            else:
                state = "[green]on[/green]" if logger.enabled else "[red]off[/red]"
                print_info(f"Logging: {state}")
                if logger.log_path:
                    print_info(f"Log file: [cyan]{logger.log_path}[/cyan]")

    # ── tokens ────────────────────────────────────────────────────────────────
    elif cmd == "tokens":
        if agent is None:
            print_warn("Agent not available.")
        else:
            return config, scope, "show_tokens"

    # ── unknown ───────────────────────────────────────────────────────────────
    else:
        print_error(f"Unknown command: /{cmd}   — type [cyan]/help[/cyan]")

    return config, scope, None
