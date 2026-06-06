"""Kratos agent — dual-model pipeline with dynamic reasoning and auto-compression.

Token-stream convention:
  All generators yield (source: str, content: str, kind: str) where source is:
    "router"   routing decision info           kind="info"
    "header"   section start                   kind="planner"|"coder"|"verify"|"relay"
    "planner"  planner token stream            kind="think"|"text"
    "coder"    coder token stream              kind="think"|"text"
    "verify"   verifier token stream           kind="think"|"text"
    "relay"    relay pre-processor (no display) kind="text"
    "tool"     tool call display               kind="tool"
    "direct"   direct answer (no LLM)          kind="text"
    "info"     informational message            kind="info"
    "warn"     warning                          kind="warn"
    "error"    error                            kind="error"
    "question" clarification request            kind="question"
    "usage"    token usage JSON                 kind="usage"
    "end"      section end                      kind="end"

Pipeline:
  Input → Analyze → Classify → Route → Build Context
    → [Relay if huge] → Planner → Coder → Verifier
    → if NEEDS_REVISION: re-plan → re-code → re-verify  (up to max_verify_iterations)
    → VERIFIED | UNSOLVABLE

Auto-compression: before each model call, if estimated prompt tokens >
compress_threshold × num_ctx → compress_history() via the compressor model.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from .analyzer import InputAnalyzer
from .bridge import OllamaBridge
from .classifier import IntentClassifier, Intent
from .compress import Compressor
from .config import KratosConfig, _project_dir, GLOBAL_DIR
from .context import ContextBuilder, ContextPackage, ProjectIndexer, ScopeType
from .memory import MemoryEntry, MemoryManager
from .router import Route, Router
from .tokens import (
    estimate, estimate_messages, choose_num_ctx,
    fit_to_budget, relay_needed, model_max_ctx,
)

# Prompts are fully externalized to JSON — all AI-facing strings, patterns, and pipeline config
# live in prompts_default.json (user-editable). Python code contains only flow logic.
from .prompts import (
    load_prompts,
    get_system,
    get_snippet,
    get_predict,
    get_marker,
    get_toolchain,
    get_plan_config,
    reload_prompts,
)

# ── token predict limits (now sourced from prompts at runtime for full configurability) ──
# (kept as module fallbacks only for very early import paths; prefer get_predict)

_PLAN_PREDICT        = 2048
_PLAN_PREDICT_HEAVY  = 3072
_PLAN_PREDICT_RETRY  = 6144
_CODE_PREDICT        = 16384
_VERIFY_PREDICT      = 512
_RELAY_PREDICT       = 1200


# ── output parsers (markers sourced from prompts at runtime for perfect sync with JSON) ──

def _get_file_change_re():
    pm = load_prompts()
    fm = re.escape(pm.get_marker("file") or "### FILE:")
    return re.compile(rf'{fm}\s*(.+?)\s*\n```(?:\w+)?\n(.*?)```', re.S)

def _get_file_delete_re():
    pm = load_prompts()
    dm = re.escape(pm.get_marker("delete") or "### DELETE:")
    return re.compile(rf'{dm}\s*(.+?)\s*$', re.M)

# Fallback module-level compiled (using defaults at import time)
_FILE_CHANGE_RE = re.compile(
    r'###\s+FILE:\s*(.+?)\s*\n```(?:\w+)?\n(.*?)```',
    re.S,
)
_FILE_DELETE_RE = re.compile(r'###\s+DELETE:\s*(.+?)\s*$', re.M)


def _parse_file_changes(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).strip(), m.group(2)) for m in _get_file_change_re().finditer(text)]


def _parse_file_deletions(text: str) -> list[str]:
    return [m.group(1).strip() for m in _get_file_delete_re().finditer(text)]


@dataclass(frozen=True)
class VerificationCommand:
    cmd: str
    purpose: str
    source: str
    is_test: bool


@dataclass
class ProvenWork:
    iteration: int
    files_changed: list[str] = field(default_factory=list)
    file_checks: list[dict] = field(default_factory=list)
    commands: list[dict] = field(default_factory=list)
    commands_planned: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "files_changed": self.files_changed,
            "file_checks": self.file_checks,
            "commands_planned": self.commands_planned,
            "commands": self.commands,
        }


# Fallback constants — the JSON toolchain section is the authoritative source.
# _is_safe_verification_command() always reads from JSON (cached), these are emergency fallbacks.
_VERIFY_META_CHARS_FALLBACK: tuple[str, ...] = ("&&", "||", ";", "|", ">", "<", "`")
_SAFE_VERIFY_PREFIXES_FALLBACK: tuple[str, ...] = (
    "python -m pytest", "py -m pytest", "pytest",
    "python -m unittest", "py -m unittest",
    "python tests/", "python test",
    "dotnet build", "dotnet test", "dotnet run --project",
    "npm test", "npm run test", "npm run build",
    "pnpm test", "pnpm run test", "pnpm run build",
    "yarn test", "yarn build",
    "cargo test", "cargo build",
    "go test", "go build",
    "mvn test", "gradle test", "./gradlew test",
    "npx tsc", "tsc --noEmit",
)


def _compile_check_cmds(toolchains: set[str], root: "Path") -> "list[VerificationCommand]":
    """Quick compile-only commands to run BEFORE the full test suite.

    Commands sourced from prompts JSON toolchain.compile_commands so they can be
    customized without touching Python code.
    """
    compile_cfgs: dict = get_toolchain("compile_commands") or {
        "dotnet": "dotnet build --nologo -q",
        "node_tsconfig": "npx tsc --noEmit --pretty false",
    }
    cmds: list[VerificationCommand] = []
    if "dotnet" in toolchains:
        cmd = compile_cfgs.get("dotnet", "dotnet build --nologo -q")
        cmds.append(VerificationCommand(cmd=cmd, purpose="compile check", source="auto-compile", is_test=False))
    if "node" in toolchains and (root / "tsconfig.json").exists():
        cmd = compile_cfgs.get("node_tsconfig", "npx tsc --noEmit --pretty false")
        cmds.append(VerificationCommand(cmd=cmd, purpose="TypeScript compile check", source="auto-compile", is_test=False))
    return cmds


def _clean_command_line(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^(?:PS>|\$|>|#)\s*", "", s)
    s = s.split(" #", 1)[0].strip()
    return s


def _is_safe_verification_command(cmd: str) -> bool:
    # Read from JSON on each call (load_prompts() is cached — O(1) after first load).
    meta_chars = tuple(get_toolchain("blocked_verify_chars") or _VERIFY_META_CHARS_FALLBACK)
    safe_prefixes = tuple(get_toolchain("safe_verify_prefixes") or _SAFE_VERIFY_PREFIXES_FALLBACK)
    normalized = " ".join(cmd.strip().split()).lower()
    if not normalized or any(meta in normalized for meta in meta_chars):
        return False
    if normalized.startswith("dotnet run --project"):
        return _is_test_verification_command(normalized)
    return normalized.startswith(safe_prefixes)


def _is_test_verification_command(cmd: str) -> bool:
    normalized = " ".join(cmd.strip().split()).lower().replace("\\", "/")
    if normalized.startswith(("dotnet build", "npm run build", "pnpm run build", "yarn build")):
        return False
    if normalized.startswith(("python -m pytest", "py -m pytest", "pytest")):
        return True
    if normalized.startswith(("python -m unittest", "py -m unittest")):
        return True
    if normalized.startswith("python ") and "/test" in normalized:
        return True
    if normalized.startswith("dotnet test"):
        return True
    if normalized.startswith("dotnet run --project") and "test" in normalized:
        return True
    if normalized.startswith(("npm test", "npm run test", "pnpm test", "pnpm run test", "yarn test")):
        return True
    return normalized.startswith(("cargo test", "go test", "mvn test", "gradle test", "./gradlew test"))


def _purpose_for_command(cmd: str, source: str) -> str:
    return f"{source} {'test' if _is_test_verification_command(cmd) else 'build'} verification"


def _quote_rel(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return f'"{rel}"' if " " in rel else rel


def _is_noise_path(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return bool(parts & {".git", ".svn", ".hg", "node_modules", ".venv", "venv", "bin", "obj", "dist", "build", "target"})


def _detect_project_toolchains(root: Path) -> set[str]:
    """Return the build/test toolchains actually present in the project root."""
    toolchains: set[str] = set()
    # Python
    if any((root / n).exists() for n in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini")):
        toolchains.add("python")
    elif (root / "tests").is_dir() or any(root.glob("test_*.py")):
        toolchains.add("python")
    # .NET
    if any(root.glob("*.sln")) or any(root.rglob("*.csproj")):
        toolchains.add("dotnet")
    # Node
    if (root / "package.json").exists():
        toolchains.add("node")
    # Rust
    if (root / "Cargo.toml").exists():
        toolchains.add("cargo")
    # Go
    if (root / "go.mod").exists():
        toolchains.add("go")
    # Maven / Gradle
    if (root / "pom.xml").exists():
        toolchains.add("maven")
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        toolchains.add("gradle")
    return toolchains


def _command_toolchain(cmd: str) -> str | None:
    """Return the toolchain name a verify command belongs to, or None."""
    n = " ".join(cmd.strip().split()).lower()
    if n.startswith(("pytest", "python -m pytest", "py -m pytest",
                     "python -m unittest", "py -m unittest")):
        return "python"
    if n.startswith("python ") and "test" in n:
        return "python"
    if n.startswith("dotnet"):
        return "dotnet"
    if n.startswith(("npm ", "pnpm ", "yarn ")):
        return "node"
    if n.startswith("cargo "):
        return "cargo"
    if n.startswith("go "):
        return "go"
    if n.startswith("mvn "):
        return "maven"
    if n.startswith(("gradle ", "./gradlew ")):
        return "gradle"
    return None


def _dedupe_verification_commands(commands: list[VerificationCommand]) -> list[VerificationCommand]:
    seen: set[str] = set()
    unique: list[VerificationCommand] = []
    for item in commands:
        key = " ".join(item.cmd.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _extract_readme_verification_commands(root: Path) -> list[VerificationCommand]:
    commands: list[VerificationCommand] = []
    for readme in sorted(root.glob("README*")):
        if not readme.is_file():
            continue
        try:
            text = readme.read_text("utf-8", errors="replace")
        except OSError:
            continue
        in_fence = False
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped.startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            cmd = _clean_command_line(stripped)
            if not in_fence and not _is_safe_verification_command(cmd):
                continue
            if _is_safe_verification_command(cmd):
                commands.append(VerificationCommand(
                    cmd=cmd,
                    purpose=_purpose_for_command(cmd, readme.name),
                    source=readme.name,
                    is_test=_is_test_verification_command(cmd),
                ))
    # Bare "dotnet build" / "dotnet test" without a project path only work when a
    # .sln or .csproj lives in the root.  Drop them when they don't — the inferred
    # commands (_infer_project_verification_commands) will provide explicit-path
    # equivalents that actually work.
    has_root_project = any(root.glob("*.sln")) or any(root.glob("*.csproj"))
    if not has_root_project:
        import re as _re
        commands = [
            c for c in commands
            if not _re.match(r"^dotnet\s+(build|test)\s*$", c.cmd.strip(), _re.I)
        ]

    return _dedupe_verification_commands(commands)


def _infer_project_verification_commands(root: Path) -> list[VerificationCommand]:
    commands: list[VerificationCommand] = []

    if (root / "package.json").exists():
        try:
            package = json.loads((root / "package.json").read_text("utf-8"))
            scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
        except (json.JSONDecodeError, OSError):
            scripts = {}
        if "build" in scripts:
            commands.append(VerificationCommand("npm run build", "package.json build verification", "package.json", False))
        if "test" in scripts:
            commands.append(VerificationCommand("npm test", "package.json test verification", "package.json", True))

    has_python_project = any((root / name).exists() for name in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini"))
    has_python_tests = (root / "tests").exists() or any(root.glob("test_*.py"))
    if has_python_project and has_python_tests:
        commands.append(VerificationCommand("python -m pytest tests", "python pytest verification", "python", True))

    if (root / "Cargo.toml").exists():
        commands.append(VerificationCommand("cargo test", "cargo test verification", "cargo", True))

    if (root / "go.mod").exists():
        commands.append(VerificationCommand("go test ./...", "go test verification", "go", True))

    sln_files = [p for p in root.glob("*.sln") if not _is_noise_path(p)]
    if sln_files:
        rel = _quote_rel(sln_files[0], root)
        commands.append(VerificationCommand(f"dotnet build {rel}", "dotnet solution build verification", "dotnet", False))
        commands.append(VerificationCommand(f"dotnet test {rel} --no-build", "dotnet solution test verification", "dotnet", True))
        return _dedupe_verification_commands(commands)

    csprojs = [p for p in root.rglob("*.csproj") if not _is_noise_path(p)]
    test_projects: list[Path] = []
    source_projects: list[Path] = []
    for csproj in csprojs:
        try:
            text = csproj.read_text("utf-8", errors="replace").lower()
        except OSError:
            text = ""
        rel_lower = csproj.relative_to(root).as_posix().lower()
        if "test" in rel_lower or "microsoft.net.test.sdk" in text or "xunit" in text or "nunit" in text:
            test_projects.append(csproj)
        else:
            source_projects.append(csproj)

    for csproj in test_projects:
        rel = _quote_rel(csproj, root)
        commands.append(VerificationCommand(f"dotnet build {rel}", "dotnet test project build verification", "dotnet", False))
        try:
            text = csproj.read_text("utf-8", errors="replace").lower()
        except OSError:
            text = ""
        if "microsoft.net.test.sdk" in text or "xunit" in text or "nunit" in text:
            commands.append(VerificationCommand(f"dotnet test {rel} --no-build", "dotnet test project verification", "dotnet", True))
        else:
            commands.append(VerificationCommand(f"dotnet run --project {rel}", "dotnet executable test project verification", "dotnet", True))

    if not test_projects and source_projects:
        rel = _quote_rel(source_projects[0], root)
        commands.append(VerificationCommand(f"dotnet build {rel}", "dotnet project build verification", "dotnet", False))

    return _dedupe_verification_commands(commands)


def _proven_work_satisfied(proof: ProvenWork, require_test: bool = True) -> bool:
    if not proof.commands:
        return False
    # Per-step intermediate failures are allowed (other stubs may not be done yet).
    # What matters is the FINAL state: the last test command that ran must have passed.
    if require_test:
        test_cmds = [c for c in proof.commands if c.get("is_test")]
        if not test_cmds:
            return False
        return bool(test_cmds[-1].get("exit_code") == 0)
    # Build-only case: all commands must pass.
    return all(item.get("exit_code") == 0 for item in proof.commands)


def _format_proven_work_feedback(proof: ProvenWork, require_test: bool = True) -> str:
    if not proof.commands_planned:
        return "PROVEN_WORK missing: no safe build/test command was configured, found in README, or inferred from project files."
    if not proof.commands:
        return "PROVEN_WORK missing: verification commands were planned but not executed."
    failed = [item for item in proof.commands if item.get("exit_code") != 0]
    if failed:
        item = failed[-1]  # last failure is most recent and most relevant
        return (
            f"PROVEN_WORK failed: command `{item.get('cmd')}` exited with {item.get('exit_code')}.\n"
            f"{str(item.get('output', ''))[-3000:]}"
        )
    if require_test and not any(item.get("is_test") for item in proof.commands):
        return "PROVEN_WORK incomplete: build commands passed, but no test command actually ran."
    return "PROVEN_WORK incomplete: verification evidence did not satisfy the proof gate."


def _extract_plan_steps(plan: str) -> list[str]:
    """Extract NUMBERED actionable steps from planner output.

    Patterns sourced from JSON plan.step_regex and plan.step_skip_prefixes.
    Only digit-prefixed lines count as steps — bullets explicitly excluded to
    prevent the 15-item explosion where file lists / risk bullets become steps.
    Falls back to double-newline paragraph splitting if no numbered items found.
    """
    if not plan or not plan.strip():
        return []
    # Read config from JSON (cached)
    step_regex = get_plan_config("step_regex") or r'^(?:\d+[\.\)\-:]\s*|Step\s+\d+[:\.\)]?\s*)(.+)$'
    raw_skip = get_plan_config("step_skip_prefixes") or ["what ", "which ", "potential", "verification", "final ", "note:"]
    min_len = int(get_plan_config("min_step_length") or 8)
    _skip_prefixes = tuple(raw_skip)
    steps: list[str] = []
    for raw in plan.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(step_regex, line, re.I)
        if m:
            txt = m.group(1).strip()
            if len(txt) > min_len and not txt.lower().startswith(_skip_prefixes):
                steps.append(txt)
    if steps:
        return steps
    # Fallback: double-newline paragraphs containing action verbs
    parts = [p.strip() for p in re.split(r'\n\s*\n', plan) if p.strip()]
    filtered = [p for p in parts if any(k in p.lower() for k in (
        "edit", "change", "implement", "add ", "fix ", "update ", "create ", "remove "
    ))]
    return filtered[:12] or [plan.strip()[:400]]


def _extract_step_file_refs(step_text: str) -> list[str]:
    """Extract source file paths referenced in a single step description.

    Used to pre-read the CURRENT disk state of those files and inject it
    into the coder's context so it can see what's there before overwriting.
    Caps at 6 to avoid flooding the prompt.
    """
    patterns = [
        r'(?:File:|file:)\s*([^\s,\n`"\']+)',           # File: src/Foo.cs
        r'`([^`]+\.[a-zA-Z]{1,8})`',                    # `src/Foo.cs`
        r'"([^"]+\.[a-zA-Z]{1,8})"',                    # "src/Foo.cs"
    ]
    found: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, step_text):
            p = m.group(1).strip().strip('`"\'').replace("\\", "/")
            if "." in p and ("/" in p or "\\" in p) and p not in seen:
                seen.add(p)
                found.append(p)
    return found[:6]


# ── STEP_TEST: temp test parsing + .NET runner patching ──────────────────────

def _parse_step_tests(text: str) -> list[tuple[str, str]]:
    """Parse ### STEP_TEST: path blocks from coder output.

    Same format as ### FILE: blocks. Returns (rel_path, content) pairs.
    Marker sourced from JSON markers.step_test.
    """
    marker = get_marker("step_test") or "### STEP_TEST:"
    escaped = re.escape(marker)
    pattern = re.compile(rf'{escaped}\s*(.+?)\s*\n```(?:\w+)?\n(.*?)```', re.S)
    return [(m.group(1).strip(), m.group(2)) for m in pattern.finditer(text)]


def _patch_dotnet_test_runner(test_dir: Path, class_name: str) -> tuple[Path | None, str | None]:
    """Temporarily add class_name.RunAll() to the test runner's Program.cs.

    Handles two common .NET test runner formats:

    Format A — top-level call style (most common):
        TaskParserTests.RunAll();
        TaskFormatterTests.RunAll();
      → inserts ClassName.RunAll(); after the last existing call.

    Format B — tuple/array style (e.g. probe runner):
        var tests = new (string Name, Action Run)[]
        {
            ("TaskParserTests",  TaskParserTests.RunAll),
            ("TaskFormatterTests", TaskFormatterTests.RunAll)   ← no comma on last
        };
      → adds a new entry after the last tuple, fixing up trailing commas.

    Returns (prog_path, original_content) so callers can restore after the test run.
    Returns (None, None) if Program.cs not found or cannot be patched.
    """
    prog = test_dir / "Program.cs"
    if not prog.exists():
        return None, None
    original = prog.read_text("utf-8")
    lines = original.rstrip().split("\n")

    # ── Format A: lines containing RunAll() (with parentheses = direct call) ──
    last_call_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if "RunAll()" in s and not s.startswith("//"):
            last_call_idx = i
    if last_call_idx >= 0:
        lines.insert(last_call_idx + 1, f"{class_name}.RunAll();")
        prog.write_text("\n".join(lines) + "\n", "utf-8")
        return prog, original

    # ── Format B: tuple/array style — lines ending with .RunAll) or .RunAll), ─
    last_tuple_idx = -1
    indent = "    "
    for i, line in enumerate(lines):
        s = line.strip()
        if (".RunAll)" in s or ".RunAll)," in s) and not s.startswith("//"):
            last_tuple_idx = i
            indent = line[: len(line) - len(line.lstrip())]
    if last_tuple_idx >= 0:
        # Ensure the previous last entry has a trailing comma
        prev = lines[last_tuple_idx].rstrip()
        if not prev.endswith(","):
            lines[last_tuple_idx] = prev + ","
        lines.insert(last_tuple_idx + 1, f'{indent}("{class_name}", {class_name}.RunAll)')
        prog.write_text("\n".join(lines) + "\n", "utf-8")
        return prog, original

    # ── Fallback: just append (may not compile, but best-effort) ──────────────
    lines.append(f"{class_name}.RunAll();")
    prog.write_text("\n".join(lines) + "\n", "utf-8")
    return prog, original


# ── command registry ──────────────────────────────────────────────────────────

class CommandRegistry:
    """Centralized manager for all project build/test commands.

    Discovers commands once per run from README, project structure, and config.
    Provides structured access for the coder prompt and the verify pipeline.
    """

    def __init__(self, config: "KratosConfig", root: Path) -> None:
        self._config = config
        self._root = root
        self.toolchains: set[str] = set()
        self.commands: list[VerificationCommand] = []
        self.compile_commands: list[VerificationCommand] = []

    def discover(self) -> "CommandRegistry":
        self.toolchains = _detect_project_toolchains(self._root)
        raw: list[VerificationCommand] = []
        # Configured commands first (highest priority)
        if self._config.build_cmd:
            raw.append(VerificationCommand(self._config.build_cmd, "configured build", "/build", False))
        if self._config.test_cmd:
            raw.append(VerificationCommand(self._config.test_cmd, "configured test", "/test", True))
        # Auto-discovered
        raw += _extract_readme_verification_commands(self._root)
        raw += _infer_project_verification_commands(self._root)
        self.commands = _dedupe_verification_commands([
            c for c in raw if _is_safe_verification_command(c.cmd)
        ])
        self.compile_commands = _compile_check_cmds(self.toolchains, self._root)
        return self

    def verify_hint(self) -> str:
        """One-block description for injection into planner/coder prompts."""
        tc_str = ", ".join(sorted(self.toolchains)) if self.toolchains else "unknown"
        if self.commands:
            lines = [
                f"PROJECT TOOLCHAIN: {tc_str}",
                "STEP_VERIFY MUST use only the commands below — do NOT invent a runner "
                "(e.g. never use pytest for a dotnet project, never use dotnet for Python):",
            ]
            lines += [f"  `{c.cmd}`" for c in self.commands[:4]]
            return "\n".join(lines)
        elif self.toolchains:
            return (
                f"PROJECT TOOLCHAIN: {tc_str}. "
                "Use only the correct runner for this toolchain in every STEP_VERIFY."
            )
        return ""

    def format_for_prompt(self) -> str:
        """Compact command registry block for coder context injection."""
        tc_str = ", ".join(sorted(self.toolchains)) if self.toolchains else "unknown"
        lines = [f"COMMAND REGISTRY  (toolchain: {tc_str})"]
        if self.compile_commands:
            for c in self.compile_commands:
                lines.append(f"  [COMPILE]  `{c.cmd}` — {c.purpose}")
        for c in self.commands[:5]:
            kind = "[TEST]  " if c.is_test else "[BUILD] "
            lines.append(f"  {kind} `{c.cmd}` — {c.purpose}")
        if not self.compile_commands and not self.commands:
            lines.append("  (no commands detected — ensure README or project files describe how to build/test)")
        return "\n".join(lines)

    def test_commands(self) -> list[VerificationCommand]:
        return [c for c in self.commands if c.is_test]

    def is_toolchain_mismatch(self, cmd: str) -> bool:
        if not self.toolchains:
            return False
        tc = _command_toolchain(cmd)
        return bool(tc and tc not in self.toolchains)


# ── scope / num_ctx selection ─────────────────────────────────────────────────

def _scope_for(route: Route, intent: Intent) -> ScopeType:
    if route == Route.DIRECT_ANSWER:
        return "none"
    if route == Route.PLANNER_ONLY:
        if intent in (Intent.QUESTION, Intent.EXPLAIN):
            return "minimal"
        return "architecture"
    if route == Route.CODER_ONLY:
        if intent == Intent.FOLLOWUP:
            return "patch_context"
        if intent == Intent.SHELL_GIT:
            return "none"
        return "targeted"
    if route == Route.DIAGNOSTIC_LOOP:
        return "diagnostic"
    if route == Route.PLANNER_THEN_CODER:
        return "architecture"
    return "minimal"


def _coder_scope_for(intent: Intent) -> ScopeType:
    if intent == Intent.FOLLOWUP:
        return "patch_context"
    return "expanded"


# ── dynamic reasoning ─────────────────────────────────────────────────────────

def _needs_thinking(
    task: str, scope: ScopeType, route: Route, is_retry: bool, n_files: int
) -> bool:
    """CoT is disabled — the 8b model takes 20+ minutes with think=True on a 6 GB laptop
    and causes Ollama server timeouts.  Context-rich prompts + PROVEN_WORK feedback give
    the planner all the signal it needs without chain-of-thought."""
    return False


# ── message builders ──────────────────────────────────────────────────────────

def _coder_context_block(ctx: ContextPackage, pm, step_mode: bool = False) -> str:
    """Build the file-context section used in coder prompts.

    Returns a string with test-file headers, stub-file headers, and done-source
    excerpts — or an empty string if ctx has no files.

    step_mode=True additionally classifies C# // TODO stubs as stubs (not "done").
    """
    if not ctx.files:
        return ""
    noise = ("README", "ARCH", "REQUIRE", "docs/", "CHANGELOG", "LICENSE")
    test_files = [f for f in ctx.files
                  if any(x in f.rel_path for x in ("test_", "_test.", ".spec.", ".test.",
                                                     "Tests.", "Tests/", "tests/"))]
    src_files  = [f for f in ctx.files
                  if f not in test_files and not any(x in f.rel_path for x in noise)]

    def _is_stub(f) -> bool:
        c = f.content or ""
        if "NotImplementedError" in c:
            return True
        # C# stubs: always detect TODO regardless of step_mode.
        # Without this, the non-stepwise coder path classifies C# stubs as
        # "done" files (do-not-rewrite), hiding them from the implementation prompt.
        if "// TODO" in c or "/* TODO" in c:
            return True
        return False

    stub_files = [f for f in src_files if _is_stub(f)]
    done_files = [f for f in src_files if f not in stub_files]

    parts: list[str] = []
    if test_files:
        parts.append(pm.get_snippet("test_files_header") or
                     "TEST FILES — these define the exact API. Match every signature exactly:")
        for f in test_files:
            # Test files define the exact contract — give them full content (they're small).
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(4000)}")
    if stub_files:
        names = ", ".join(f.rel_path for f in stub_files)
        parts.append((pm.get_snippet("stub_files_header") or "STUB FILES — IMPLEMENT ALL: ") + names)
        for f in stub_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(2500)}")
    if done_files:
        parts.append(pm.get_snippet("done_files_header") or
                     "Already-implemented (reference only — do NOT rewrite unless plan says to):")
        for f in done_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1000)}")
    return "\n\n".join(p for p in parts if p)


def _planner_msg(task: str, ctx: ContextPackage, all_files: list | None = None, verify_hint: str = "") -> str:
    pm = load_prompts()
    parts: list[str] = []
    if ctx.project_description:
        parts.append(ctx.project_description)
    if all_files and not ctx.project_description:
        listing = "\n".join(f"  {e.rel_path}" for e in all_files[:100])
        parts.append((pm.get_snippet("all_project_files_header") or "All project files:\n") + listing)
    if ctx.memory_summary:
        parts.append(ctx.memory_summary)
    if ctx.files:
        parts.append(pm.get_snippet("file_contents_header") or "File contents (most relevant):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1500)}")
    if ctx.error_lines:
        parts.append((pm.get_snippet("errors_header") or "Errors/logs:\n") + "\n".join(ctx.error_lines[:15]))
    if verify_hint:
        parts.append(verify_hint)
    parts.append(f"{pm.get_snippet('task_label') or 'Task: '}{task}")
    return "\n\n".join(p for p in parts if p)


def _coder_msg(task: str, ctx: ContextPackage, plan: str) -> str:
    pm = load_prompts()
    parts: list[str] = []
    if plan:
        parts.append(f"{pm.get_snippet('plan_label') or 'Plan:\\n'}{plan}")
    ctx_block = _coder_context_block(ctx, pm, step_mode=False)
    if ctx_block:
        parts.append(ctx_block)
    if ctx.error_lines:
        parts.append((pm.get_snippet("errors_short_header") or "Errors:\n") + "\n".join(ctx.error_lines[:10]))
    parts.append(f"{pm.get_snippet('task_label') or 'Task: '}{task}")
    return "\n\n".join(p for p in parts if p)


def _planner_retry_msg(
    task: str, prev_plan: str, verify_feedback: str, ctx: ContextPackage | None = None
) -> str:
    pm = load_prompts()
    parts = [f"{pm.get_snippet('task_label') or 'Task: '}{task}"]
    if ctx and ctx.files:
        parts.append(pm.get_snippet("current_file_state_header") or "Current file state (updated since last iteration):")
        for f in ctx.files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(800)}")
    parts.append(f"{pm.get_snippet('previous_plan_label') or 'Previous plan:\\n'}{prev_plan[:600]}")
    fb_intro = pm.get_snippet("verify_feedback_intro") or "Verifier / test feedback — what still needs to be fixed:\n"
    fb_action = pm.get_snippet("verify_feedback_action") or (
        "Produce a precise plan for what the coder must implement to fix all issues. "
        "List each file and function. For circular imports name the exact import line to remove."
    )
    parts.append(f"{fb_intro}{verify_feedback[:2000]}\n\n{fb_action}")
    return "\n\n".join(parts)


def _coder_retry_msg(
    task: str, ctx: ContextPackage, plan: str, verify_feedback: str
) -> str:
    pm = load_prompts()
    parts: list[str] = []
    if ctx.files:
        noise = ("README", "ARCH", "REQUIRE", "docs/", "CHANGELOG", "LICENSE")
        test_files = [f for f in ctx.files
                      if any(x in f.rel_path for x in ("test_", "_test.", ".spec.", ".test."))]
        src_files  = [f for f in ctx.files
                      if f not in test_files and not any(x in f.rel_path for x in noise)]
        if test_files:
            parts.append(pm.get_snippet("test_files_header_short") or "TEST FILES — exact API you must satisfy:")
            for f in test_files:
                parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1000)}")
        parts.append("Current source files:")
        for f in src_files:
            parts.append(f"--- {f.rel_path} ---\n{f.excerpt(1000)}")
    parts.append(f"{pm.get_snippet('revised_plan_label') or 'Revised plan:\\n'}{plan}")
    rf_intro = pm.get_snippet("required_fixes_intro") or "Required fixes:\n"
    rf_action = pm.get_snippet("required_fixes_action") or "Fix ALL issues. Every NotImplementedError must be replaced."
    parts.append(f"{rf_intro}{verify_feedback[:2000]}\n\n{rf_action}")
    parts.append(f"{pm.get_snippet('task_label') or 'Task: '}{task}")
    return "\n\n".join(p for p in parts if p)


def _verify_msg(task: str, plan: str, coder_output: str, proof: ProvenWork | None = None) -> str:
    pm = load_prompts()
    proof_text = json.dumps(proof.to_dict(), ensure_ascii=False, indent=2) if proof else "{}"
    tl = pm.get_snippet("task_label") or "Task: "
    pl = pm.get_snippet("plan_label") or "Plan given to coder:\n"
    cl = pm.get_snippet("proven_work_label") or "PROVEN_WORK evidence from Kratos runtime:\n"
    return (
        f"{tl}{task}\n\n"
        f"{pl}{plan[:600]}\n\n"
        f"Coder output:\n{coder_output[:2000]}\n\n"
        f"{cl}{proof_text[:4000]}"
    )


def _clarification_msg(analysis, intent: Intent) -> str:
    pm = load_prompts()
    kw_str = ", ".join(analysis.keywords[:6]) if analysis.keywords else "none"
    tmpl = pm.get_snippet("clarification_template") or (
        "Your request is unclear. Detected keywords: [{kw}]. "
        "Please specify: what should change, which file(s), and what the expected result is."
    )
    return tmpl.format(kw=kw_str)


# ── direct search (no LLM) ────────────────────────────────────────────────────

def _direct_file_search(indexer: ProjectIndexer, keywords: list[str], file_paths: list[str]) -> str:
    results: list[str] = []
    index = indexer.index
    if not index:
        return "No project files indexed."
    for fp in file_paths:
        fp_l = fp.lower().replace("\\", "/")
        for e in index:
            if fp_l in e.rel_path.lower():
                results.append(f"  {e.rel_path}  [pri={e.priority}]")
    for kw in keywords[:5]:
        for e in index:
            line = f"  {e.rel_path}  [pri={e.priority}]"
            if kw.lower() in e.rel_path.lower() and line not in results:
                results.append(line)
    if not results:
        top = [f"  {e.rel_path}" for e in index[:20]]
        return "No direct matches. Top project files:\n" + "\n".join(top)
    return "Matching files:\n" + "\n".join(results[:20])


def _direct_code_search(indexer: ProjectIndexer, keywords: list[str]) -> str:
    lines: list[str] = []
    for kw in keywords[:3]:
        hits = indexer.search_content(kw, max_results=8)
        if hits:
            lines.append(f"\nResults for '{kw}':")
            for entry, lineno, text in hits:
                lines.append(f"  {entry.rel_path}:{lineno}  {text}")
    return "\n".join(lines) if lines else "No matches found in project files."


# ── main agent ────────────────────────────────────────────────────────────────

class KratosAgent:
    def __init__(self, config: KratosConfig, bridge: OllamaBridge, prompts=None) -> None:
        self.config = config
        self.bridge = bridge
        self.prompts = prompts or load_prompts()

        self._planner_history: list[dict] = []
        self._coder_history:   list[dict] = []

        self._analyzer   = InputAnalyzer()
        self._classifier = IntentClassifier()
        self._router     = Router()

        project_root      = Path.cwd()
        self._indexer     = ProjectIndexer(project_root)
        self._ctx_builder = ContextBuilder(self._indexer)
        self._memory      = MemoryManager(_project_dir(), GLOBAL_DIR)
        self._compressor  = Compressor(bridge, config)

        # Accumulated token usage for this session
        self._session_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0
        }

        # File operations extracted from last coder run — applied by caller
        self.pending_file_changes:   list[tuple[str, str]] = []
        self.pending_file_deletions: list[str] = []

    # ── session usage ─────────────────────────────────────────────────────────

    @property
    def session_usage(self) -> dict[str, int]:
        return dict(self._session_usage)

    def _record_usage(self, usage_json: str) -> None:
        try:
            d = json.loads(usage_json)
            self._session_usage["prompt_tokens"]     += d.get("prompt_tokens", 0)
            self._session_usage["completion_tokens"] += d.get("completion_tokens", 0)
        except Exception:
            pass

    # ── intelligent pipeline ─────────────────────────────────────────────────

    def process(self, task: str) -> Generator[tuple[str, str, str], None, None]:
        """Full pipeline. Yields (source, content, kind) events."""
        self.pending_file_changes.clear()
        self.pending_file_deletions.clear()

        analysis = self._analyzer.analyze(task)
        intent   = self._classifier.classify(analysis)
        route    = self._router.route(intent)

        yield ("router", f"intent={intent.value}  route={route.value}", "info")

        # ── memory retrieval ──────────────────────────────────────────────────
        mem_entries    = self._memory.get_relevant(
            analysis.keywords, categories=["solution", "file_role", "decision", "convention"]
        )
        memory_summary = self._memory.format_for_prompt(mem_entries)

        # ── context build ─────────────────────────────────────────────────────
        scope = _scope_for(route, intent)
        n_indexed = len(self._indexer.index)
        yield ("tool", f"index_project({self._indexer.root.name!r}) → {n_indexed} files", "tool")

        ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary=memory_summary, scope=scope,
            token_budget=self.config.planner_num_ctx,
        )
        coder_scope = _coder_scope_for(intent)
        coder_ctx = self._ctx_builder.build(
            analysis, intent=intent.value, route=route.value,
            memory_summary="", scope=coder_scope,
            token_budget=self.config.coder_num_ctx,
        )
        for f in ctx.files:
            sz = f"{f.size / 1024:.1f} KB" if f.size > 0 else "n/a"
            yield ("tool", f"read_file({f.rel_path!r}) → {sz}", "tool")

        # ── log context ───────────────────────────────────────────────────────
        yield ("log", json.dumps({
            "type": "index_project",
            "project": self._indexer.root.name,
            "file_count": n_indexed,
            "files": [{"rel_path": e.rel_path, "size": e.size, "priority": e.priority}
                      for e in self._indexer.index],
        }), "log")

        # ── command registry (once per run) ──────────────────────────────────
        # Discovers toolchains, build commands, test runners from project structure.
        # All three consumers (planner hint, coder context, stepwise verify) share this.
        _cmd_registry = CommandRegistry(self.config, self._indexer.root).discover()
        _project_toolchains = _cmd_registry.toolchains
        verify_hint = _cmd_registry.verify_hint()

        # ── terminal routes ───────────────────────────────────────────────────
        if route == Route.DIRECT_ANSWER:
            if intent == Intent.FILE_SEARCH:
                result = _direct_file_search(self._indexer, analysis.keywords, analysis.file_paths)
            else:
                result = _direct_code_search(self._indexer, analysis.keywords)
            yield ("direct", result, "text")
            return

        if route == Route.ASK_CLARIFICATION:
            yield ("question", _clarification_msg(analysis, intent), "question")
            return

        if route == Route.PLANNER_ONLY:
            yield ("header", "planner", "header")
            plan_full = yield from self._run_planner(
                _planner_msg(task, ctx, all_files=self._indexer.index, verify_hint=verify_hint),
                route, keep_alive="5m", scope=scope, task=task, is_retry=False,
            )
            yield ("end", "planner", "end")
            # Async memory extraction
            mem_entries_new = self._compressor.generate_memory(task, plan_full, "", [])
            self._memory.add_from_compress(mem_entries_new)
            return

        if route == Route.CODER_ONLY:
            # Git / shell / followup continuation — do not force the heavy "### FILE" format.
            # Still goes through coder (so it benefits from max ctx + history), but prompt is light.
            yield ("header", "coder", "header")
            pm = load_prompts()
            light = (
                f"{pm.get_snippet('task_label') or 'Task: '}{task}\n\n"
                + (pm.get_snippet("coder_only_light") or
                   "If this is a git/shell command or a small continuation, reply with the precise commands or "
                   "the minimal code patch. Use normal text or ``` blocks. Only emit ### FILE: blocks if the task "
                   "actually requires writing source files to disk.")
            )
            cfull = yield from self._run_coder(light)
            yield ("end", "coder", "end")
            # If it happened to emit files (follow-up coding), still apply them (rare for pure git)
            chg = _parse_file_changes(cfull)
            if chg:
                self.pending_file_changes = chg
                self.pending_file_deletions = _parse_file_deletions(cfull)
                # (apply happens in caller via _show_file_ops + the agent already wrote? no — for this path we apply here for consistency)
                root = self._indexer.root
                for rp, ct in chg:
                    try:
                        (root / rp).parent.mkdir(parents=True, exist_ok=True)
                        (root / rp).write_text(ct, encoding="utf-8")
                        yield ("tool", f"apply_file({rp!r}) [from CODER_ONLY]", "tool")
                    except Exception:
                        pass
            return

        # ── plan → code → verify loop ─────────────────────────────────────────
        max_iter   = self.config.max_verify_iterations
        plan_text  = ""
        verify_feedback = ""
        needs_plan = route in (Route.PLANNER_THEN_CODER, Route.DIAGNOSTIC_LOOP)
        _accumulated_changes: dict[str, str] = {}
        _accumulated_deletions: set[str] = set()
        # Snapshot of files before any writes (for rollback on UNSOLVABLE)
        _original_snapshots: dict[str, str | None] = {}

        for attempt in range(max_iter):
            is_retry   = attempt > 0
            iter_label = f"iteration {attempt + 1}/{max_iter}"

            if is_retry:
                yield ("info", f"Revising — {iter_label}", "info")
                self._indexer.build_index()
                ctx = self._ctx_builder.build(
                    analysis, intent=intent.value, route=route.value,
                    memory_summary=memory_summary, scope=scope,
                    token_budget=self.config.planner_num_ctx,
                )
                coder_ctx = self._ctx_builder.build(
                    analysis, intent=intent.value, route=route.value,
                    memory_summary="", scope=coder_scope,
                    token_budget=self.config.coder_num_ctx,
                )

            # ── 1. Large-input relay (before planner if needed) ───────────────
            planner_input_raw = (
                _planner_retry_msg(task, plan_text, verify_feedback, ctx=ctx)
                if is_retry
                else _planner_msg(task, ctx, all_files=self._indexer.index, verify_hint=verify_hint)
            )
            raw_tokens = estimate(planner_input_raw)
            if relay_needed(raw_tokens, self.config.planner_num_ctx, self.config.relay_threshold):
                yield ("info", f"Large input ({raw_tokens} est. tokens) → relay pre-process", "info")
                yield ("header", "relay", "header")
                relayed = self._compressor.relay_large_input(task, planner_input_raw)
                planner_input = relayed
                yield ("end", "relay", "end")
            else:
                planner_input = planner_input_raw

            # ── 2. Planner ────────────────────────────────────────────────────
            if needs_plan or is_retry:
                yield ("header", "planner", "header")
                plan_full = yield from self._run_planner(
                    planner_input, route, keep_alive="0",
                    scope=scope, task=task, is_retry=is_retry,
                )
                yield ("end", "planner", "end")
                plan_text = plan_full

            # ── 3. Coder — STEPWISE per plan item (with inline test verify) ───
            # This is the core "coder must think every step from plan, show command,
            # implement, verify with test, then continue every step".
            # Verifier later sees the full per-step PROVEN_WORK evidence.
            project_root = self._indexer.root
            proof = ProvenWork(iteration=attempt + 1)

            steps = _extract_plan_steps(plan_text) if (plan_text and needs_plan) else []
            use_stepwise = len(steps) >= 1 and route in (Route.PLANNER_THEN_CODER, Route.DIAGNOSTIC_LOOP)

            coder_full_for_verify = ""   # collect last coder output for the LLM verifier msg

            if use_stepwise:
                yield ("info", f"Stepwise execution: {len(steps)} plan steps (coder + test per step)", "info")
                for sidx, step in enumerate(steps):
                    step_num = sidx + 1
                    yield ("info", f"Step {step_num}/{len(steps)}: {step[:80]}", "info")

                    # fresh context for this micro-task (files may have changed from prev steps)
                    step_coder_ctx = self._ctx_builder.build(
                        analysis, intent=intent.value, route=route.value,
                        memory_summary="", scope="expanded",
                        token_budget=self.config.coder_num_ctx,
                    )

                    pm = load_prompts()
                    forced_prefix = pm.get_snippet("coder_step_forced_prefix") or (
                        "CRITICAL: Implement ONLY the SINGLE step below. Begin response with '### FILE:'.\n"
                        "Think: how exactly, risks, exact verify command for THIS step.\n\n"
                    )
                    cur_label = (pm.get_snippet("step_input_current_step_label") or "CURRENT STEP TO IMPLEMENT NOW ({num}/{total}):\n").format(num=step_num, total=len(steps))

                    # Build the project-file context block for this step so the coder
                    # sees the real types, signatures, and test contracts before writing.
                    ctx_block = _coder_context_block(step_coder_ctx, pm, step_mode=True)

                    # Pre-step disk reads: inject the CURRENT on-disk state of files
                    # referenced by this step + files touched in previous steps.
                    # This is critical: after step N writes a broken file, step N+1 coder
                    # needs to SEE that broken file to know what to fix.
                    _step_refs = _extract_step_file_refs(step)
                    _prev_touched = [f for f in _accumulated_changes if f not in _step_refs]
                    _to_read = list(dict.fromkeys(_step_refs + _prev_touched))[:8]
                    _disk_reads: list[tuple[str, str]] = []
                    for _ref in _to_read:
                        _target = (project_root / _ref).resolve()
                        try:
                            if _target.exists():
                                _raw = _target.read_text("utf-8", errors="replace")
                                _excerpt = _raw[-2500:] if len(_raw) > 2500 else _raw
                                _disk_reads.append((_ref, _excerpt))
                                yield ("tool", f"read_file({_ref!r}) -> {len(_raw)} chars [pre-step {step_num}]", "tool")
                        except OSError:
                            pass

                    disk_section = ""
                    if _disk_reads:
                        _ds_parts = [
                            "CURRENT FILE STATE ON DISK (actual content right now — "
                            "fix any errors you see, use these types/signatures exactly):"
                        ]
                        for _ref, _excerpt in _disk_reads:
                            _ds_parts.append(f"\n### FILE: {_ref}\n```\n{_excerpt}\n```")
                        disk_section = "\n".join(_ds_parts) + "\n\n"

                    # Explicit per-step file constraint
                    _step_file_constraint = ""
                    if _step_refs:
                        _step_file_constraint = (
                            f"FILES FOR THIS STEP ONLY: {', '.join(_step_refs)}\n"
                            "DO NOT modify any other file. One step = one file (or the files listed above).\n\n"
                        )

                    # Command registry section injected into coder prompt
                    _cmd_block = _cmd_registry.format_for_prompt()

                    step_input = forced_prefix + (
                        f"{pm.get_snippet('step_input_full_task_label') or 'Full task: '}{task}\n\n"
                        f"{pm.get_snippet('step_input_overall_plan_label') or 'OVERALL PLAN (for reference):\\n'}{plan_text[:1500]}\n\n"
                        + (f"{_cmd_block}\n\n" if _cmd_block else "")
                        + (f"PROJECT FILES (types, test contracts — match signatures exactly):\n{ctx_block}\n\n" if ctx_block else "")
                        + disk_section
                        + _step_file_constraint
                        + f"{cur_label}{step}\n\n"
                        f"{pm.get_snippet('step_input_prev_steps_label') or 'Previous steps completed in this attempt: '}{sidx}\n"
                        + (pm.get_snippet("step_input_after_change") or
                           "After your change the runtime will run your suggested STEP_VERIFY command and other tests.\n"
                           "Output the file(s) + ### STEP_VERIFY: <cmd>")
                    )
                    if verify_feedback:
                        step_input += f"\n\n{pm.get_snippet('step_input_feedback_suffix') or 'Previous verifier feedback to address: '}{verify_feedback[:1200]}"

                    yield ("header", "coder", "header")
                    coder_step_out = yield from self._run_coder(step_input)
                    yield ("end", "coder", "end")
                    coder_full_for_verify = coder_step_out  # last one wins for the final LLM verify msg

                    # Parse + apply only the changes from this step
                    step_changes = _parse_file_changes(coder_step_out)
                    step_deletes = _parse_file_deletions(coder_step_out)
                    step_tests   = _parse_step_tests(coder_step_out)
                    for rel_path, content in step_changes:
                        _accumulated_changes[rel_path] = content
                    _accumulated_deletions.update(step_deletes)
                    self.pending_file_changes = list(_accumulated_changes.items())
                    self.pending_file_deletions = list(_accumulated_deletions)
                    self._memory.track_files([p for p, _ in step_changes])

                    # Apply writes for this step immediately (so next step + tests see it).
                    # No SHA256 read-back — write success is proven by compile+test, not byte comparison.
                    for rel_path, content in step_changes:
                        target = (project_root / rel_path).resolve()
                        try:
                            target.relative_to(project_root.resolve())
                            if attempt == 0 and rel_path not in _original_snapshots:
                                _original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(content, encoding="utf-8")
                            file_bytes = target.stat().st_size
                            yield ("tool", f"write_file({rel_path!r}) -> {file_bytes} bytes [step {step_num}]", "tool")
                            proof.file_checks.append({"path": rel_path, "operation": "write", "ok": True, "bytes": file_bytes, "step": step_num})
                        except (ValueError, OSError) as exc:
                            proof.file_checks.append({"path": rel_path, "operation": "write", "ok": False, "error": str(exc), "step": step_num})
                            yield ("error", f"File write failed for {rel_path}: {exc}", "error")

                    for rel_path in step_deletes:
                        target = (project_root / rel_path).resolve()
                        try:
                            target.relative_to(project_root.resolve())
                            if attempt == 0 and rel_path not in _original_snapshots:
                                _original_snapshots[rel_path] = target.read_text("utf-8") if target.exists() else None
                            if target.exists():
                                target.unlink()
                            ok = not target.exists()
                            yield ("tool", f"delete_file({rel_path!r}) -> {'ok' if ok else 'FAILED'} [step {step_num}]", "tool")
                            proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": ok, "step": step_num})
                        except (ValueError, OSError) as exc:
                            proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": False, "error": str(exc), "step": step_num})
                            yield ("error", f"File delete failed for {rel_path}: {exc}", "error")

                    # STEP_TEST: write temp test files, patch test runner, run, then delete.
                    # This is the "proven work" concept: the coder writes a targeted test for
                    # the current step's logic and we execute it before the full suite.
                    _temp_test_paths: list[Path] = []
                    _patched_runner: tuple[Path | None, str | None] = (None, None)
                    _do_patch = bool(get_toolchain("step_test_dotnet_patch_program", True))
                    _do_delete = bool(get_toolchain("step_test_delete_after", True))
                    for rel_test_path, test_content in step_tests:
                        test_target = (project_root / rel_test_path).resolve()
                        try:
                            test_target.relative_to(project_root.resolve())
                            test_target.parent.mkdir(parents=True, exist_ok=True)
                            test_target.write_text(test_content, "utf-8")
                            _temp_test_paths.append(test_target)
                            yield ("tool", f"write_step_test({rel_test_path!r}) [temp, step {step_num}]", "tool")
                            # For .NET: patch the test runner's Program.cs so the class gets called
                            if _do_patch and "dotnet" in _project_toolchains and test_target.suffix.lower() == ".cs":
                                class_name = test_target.stem
                                _patched_runner = _patch_dotnet_test_runner(test_target.parent, class_name)
                                if _patched_runner[0]:
                                    yield ("tool", f"patch_test_runner({_patched_runner[0].name!r}) -> add {class_name}.RunAll()", "tool")
                        except (ValueError, OSError) as exc:
                            yield ("warn", f"STEP_TEST write failed ({rel_test_path}): {exc}", "warn")

                    # Per-step verification: compile gate + project commands + coder STEP_VERIFY
                    step_verif_cmds: list[VerificationCommand] = []
                    step_verify_match = re.search(
                        rf"{re.escape(get_marker('step_verify') or '### STEP_VERIFY:')}\s*(.+)$",
                        coder_step_out, re.M | re.I,
                    )
                    if step_verify_match:
                        raw_cmd = _clean_command_line(step_verify_match.group(1))
                        if _is_safe_verification_command(raw_cmd) and not _cmd_registry.is_toolchain_mismatch(raw_cmd):
                            step_verif_cmds.append(VerificationCommand(
                                cmd=raw_cmd,
                                purpose=f"step {step_num} suggested verify",
                                source="coder-step",
                                is_test=_is_test_verification_command(raw_cmd),
                            ))
                        elif _cmd_registry.is_toolchain_mismatch(raw_cmd):
                            tc = _command_toolchain(raw_cmd)
                            yield ("warn",
                                   f"Step {step_num}: ignored STEP_VERIFY `{raw_cmd}` "
                                   f"(toolchain `{tc}` not in project: "
                                   f"{', '.join(sorted(_project_toolchains))}). "
                                   "Using project commands instead.",
                                   "warn")
                    # Compile gate first (fast syntax check), then project test commands
                    step_verif_cmds.extend(self._verification_commands(route))
                    step_verif_cmds = _dedupe_verification_commands(step_verif_cmds)
                    if step_changes and _cmd_registry.compile_commands:
                        _existing = {c.cmd for c in step_verif_cmds}
                        step_verif_cmds = [
                            c for c in _cmd_registry.compile_commands if c.cmd not in _existing
                        ] + step_verif_cmds

                    _step_failed = False
                    if step_verif_cmds:
                        for vcmd in step_verif_cmds:
                            yield ("tool", f"run_command({vcmd.cmd!r}) -> {vcmd.purpose} [step {step_num}]", "tool")
                            res = self._run_verification_command(vcmd)
                            proof.commands.append(res)
                            proof.commands_planned.append({
                                "cmd": vcmd.cmd, "purpose": vcmd.purpose, "source": vcmd.source,
                                "is_test": vcmd.is_test, "step": step_num,
                            })
                            st = "ok" if res["exit_code"] == 0 else "FAILED"
                            yield ("tool", f"verify_command({vcmd.cmd!r}) -> {st} exit={res['exit_code']} [step {step_num}]", "tool")
                            yield ("log", json.dumps({
                                "type": "build_test", "cmd": vcmd.cmd, "purpose": vcmd.purpose,
                                "source": vcmd.source, "is_test": vcmd.is_test,
                                "exit_code": res["exit_code"], "step": step_num,
                                "duration_seconds": res["duration_seconds"], "output": res["output"][-2000:],
                            }), "log")
                            if res["exit_code"] != 0:
                                _tail = res.get("output", "")[-600:].strip()
                                verify_feedback = (
                                    f"Step {step_num} verification failed: `{vcmd.cmd}`"
                                    + (f"\nTest output:\n{_tail}" if _tail else "")
                                )
                                _step_failed = True
                                break
                        if not _step_failed:
                            yield ("info", f"Step {step_num}/{len(steps)} verified.", "info")
                    else:
                        yield ("warn", f"Step {step_num}: no verification command found. Continuing.", "warn")

                    # Clean up temp test files and restore patched runner (regardless of outcome)
                    if _do_delete:
                        for _tmp in _temp_test_paths:
                            try:
                                _tmp.unlink(missing_ok=True)
                                yield ("tool", f"delete_step_test({_tmp.name!r}) [cleanup step {step_num}]", "tool")
                            except OSError:
                                pass
                    if _patched_runner[0] and _patched_runner[1] is not None:
                        try:
                            _patched_runner[0].write_text(_patched_runner[1], "utf-8")
                            yield ("tool", f"restore_test_runner({_patched_runner[0].name!r})", "tool")
                        except OSError:
                            pass

                    # Step failed — continue so all stubs get implemented before outer retry
                    if _step_failed:
                        yield ("warn", f"Step {step_num} failed — continuing with remaining steps", "warn")
                        continue

            else:
                # ── Legacy / fallback one-shot coder (CODER_ONLY, unclear plans, retries) ──
                yield ("header", "coder", "header")
                pm = load_prompts()
                forced_prefix = pm.get_snippet("coder_forced_prefix_legacy") or (
                    "CRITICAL: Your response MUST begin with '### FILE:' on the very first line.\n"
                    "Output ONLY source files as specified in the system prompt.\n\n"
                )
                if is_retry:
                    coder_input = forced_prefix + _coder_retry_msg(task, coder_ctx, plan_text, verify_feedback)
                else:
                    coder_input = forced_prefix + _coder_msg(task, coder_ctx, plan=plan_text)

                coder_full = yield from self._run_coder(coder_input)
                yield ("end", "coder", "end")
                coder_full_for_verify = coder_full

                # Accumulate + apply exactly as before (one big batch)
                new_changes   = _parse_file_changes(coder_full)
                new_deletions = _parse_file_deletions(coder_full)
                for rel_path, content in new_changes:
                    _accumulated_changes[rel_path] = content
                _accumulated_deletions.update(new_deletions)
                self.pending_file_changes   = list(_accumulated_changes.items())
                self.pending_file_deletions = list(_accumulated_deletions)
                self._memory.track_files([p for p, _ in self.pending_file_changes])

                for rel_path, content in self.pending_file_changes:
                    target = (project_root / rel_path).resolve()
                    try:
                        target.relative_to(project_root.resolve())
                        if attempt == 0 and rel_path not in _original_snapshots:
                            _original_snapshots[rel_path] = (
                                target.read_text("utf-8") if target.exists() else None
                            )
                        size = len(content.encode("utf-8"))
                        yield ("tool", f"apply_file({rel_path!r}) -> write {size} bytes", "tool")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(content, encoding="utf-8")
                        actual = target.read_text("utf-8")
                        digest = hashlib.sha256(actual.encode("utf-8")).hexdigest()[:12]
                        ok = actual == content
                        proof.file_checks.append({
                            "path": rel_path, "operation": "write", "ok": ok,
                            "bytes": len(actual.encode("utf-8")), "sha256": digest,
                        })
                        yield ("tool", f"verify_file_write({rel_path!r}) -> {'ok' if ok else 'FAILED'} sha256={digest}", "tool")
                    except (ValueError, OSError) as exc:
                        proof.file_checks.append({"path": rel_path, "operation": "write", "ok": False, "error": str(exc)})
                        yield ("error", f"File write failed for {rel_path}: {exc}", "error")

                for rel_path in self.pending_file_deletions:
                    target = (project_root / rel_path).resolve()
                    try:
                        target.relative_to(project_root.resolve())
                        if attempt == 0 and rel_path not in _original_snapshots:
                            _original_snapshots[rel_path] = (
                                target.read_text("utf-8") if target.exists() else None
                            )
                        yield ("tool", f"delete_file({rel_path!r}) -> apply deletion", "tool")
                        if target.exists():
                            target.unlink()
                        ok = not target.exists()
                        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": ok})
                        yield ("tool", f"verify_file_delete({rel_path!r}) -> {'ok' if ok else 'FAILED'}", "tool")
                    except (ValueError, OSError) as exc:
                        proof.file_checks.append({"path": rel_path, "operation": "delete", "ok": False, "error": str(exc)})
                        yield ("error", f"File delete failed for {rel_path}: {exc}", "error")

            failed_file_checks = [item for item in proof.file_checks if not item.get("ok")]
            if failed_file_checks:
                verify_feedback = f"File application failed: {failed_file_checks[0]}"
                yield ("log", json.dumps({"type": "proven_work", **proof.to_dict()}), "log")
                yield ("log", json.dumps({"type": "verify_decision",
                                          "decision": "NEEDS_REVISION",
                                          "feedback": verify_feedback,
                                          "iteration": attempt + 1}), "log")
                if attempt < max_iter - 1:
                    yield ("warn", verify_feedback, "warn")
                    continue
                yield ("warn", f"Safety cap ({max_iter} iterations) reached.", "warn")
                break

            # ── 4a. PROVEN_WORK command evidence (final sweep + per-step already recorded) ─
            # In stepwise the per-step tests are already in proof.commands.
            # We still run the configured/full suite here for the final gate + LLM verifier.
            verification_commands = self._verification_commands(route)
            extra_planned = [
                {"cmd": item.cmd, "purpose": item.purpose, "source": item.source, "is_test": item.is_test}
                for item in verification_commands
            ]
            proof.commands_planned.extend(extra_planned)

            if verification_commands:
                yield ("log", json.dumps({"type": "proven_work_plan",
                                          "commands": extra_planned,
                                          "iteration": attempt + 1}), "log")
                for verify_cmd in verification_commands:
                    # avoid re-running identical that just passed in the last step
                    already_ran_ok = any(
                        c.get("cmd") == verify_cmd.cmd and c.get("exit_code") == 0
                        for c in proof.commands
                    )
                    if already_ran_ok:
                        continue
                    yield ("tool", f"run_command({verify_cmd.cmd!r}) -> {verify_cmd.purpose} [final]", "tool")
                    result = self._run_verification_command(verify_cmd)
                    proof.commands.append(result)
                    status = "ok" if result["exit_code"] == 0 else "FAILED"
                    yield ("tool", f"verify_command({verify_cmd.cmd!r}) -> {status} exit_code={result['exit_code']}", "tool")
                    yield ("log", json.dumps({"type": "build_test",
                                              "cmd": verify_cmd.cmd, "purpose": verify_cmd.purpose,
                                              "source": verify_cmd.source, "is_test": verify_cmd.is_test,
                                              "exit_code": result["exit_code"],
                                              "duration_seconds": result["duration_seconds"],
                                              "output": result["output"][-3000:]}), "log")
                    if result["exit_code"] != 0:
                        break
            else:
                yield ("warn", "PROVEN_WORK: no safe verification command found. Set /test <cmd> or add test instructions.", "warn")

            yield ("log", json.dumps({"type": "proven_work", **proof.to_dict()}), "log")
            self._save_iteration_state(attempt + 1, plan_text, verify_feedback, proof)

            proof_required = self.config.require_proven_work
            require_test = self.config.require_test_for_verified
            if proof_required and not _proven_work_satisfied(proof, require_test=require_test):
                verify_feedback = _format_proven_work_feedback(proof, require_test=require_test)
                yield ("warn", verify_feedback[:500], "warn")
                yield ("log", json.dumps({"type": "verify_decision",
                                          "decision": "NEEDS_REVISION",
                                          "feedback": verify_feedback,
                                          "iteration": attempt + 1}), "log")
                if attempt < max_iter - 1:
                    continue
                yield ("warn", f"Safety cap ({max_iter} iterations) reached.", "warn")
                break

            # ── 4b. LLM verifier, gated by PROVEN_WORK (now sees per-step evidence) ─
            yield ("header", "verify", "header")
            verify_full = yield from self._run_verifier(
                _verify_msg(task, plan_text, coder_full_for_verify or plan_text, proof)
            )
            yield ("end", "verify", "end")

            vf_upper = verify_full.upper()

            if "UNSOLVABLE" in vf_upper:
                reason = verify_full.replace("UNSOLVABLE:", "").replace("UNSOLVABLE", "").strip()
                yield ("warn", f"Task cannot be solved — {reason[:300]}", "warn")
                yield ("log", json.dumps({"type": "verify_decision", "decision": "UNSOLVABLE",
                                          "feedback": verify_full, "iteration": attempt + 1}), "log")
                # Rollback files written during this run
                self._rollback(_original_snapshots, project_root)
                self.pending_file_changes.clear()
                self.pending_file_deletions.clear()
                break

            if "VERIFIED" in vf_upper and "NEEDS_REVISION" not in vf_upper:
                yield ("log", json.dumps({"type": "verify_decision", "decision": "VERIFIED",
                                          "feedback": "", "iteration": attempt + 1}), "log")
                self._record_solution([p for p, _ in self.pending_file_changes],
                                      attempt + 1, task, plan_text, coder_full_for_verify or plan_text, proof)
                yield ("info", "PROVEN_WORK accepted — verification commands ran and passed.", "info")
                if attempt > 0:
                    yield ("info", f"Verified after {attempt + 1} iteration(s).", "info")

                # Final extra verification pass — verifier "really does the tests"
                final_cmds = self._verification_commands(route)
                if final_cmds:
                    yield ("info", "Final full verification sweep...", "info")
                    all_ok = True
                    for fcmd in final_cmds:
                        yield ("tool", f"run_command({fcmd.cmd!r}) -> final full check", "tool")
                        fres = self._run_verification_command(fcmd)
                        proof.commands.append(fres)  # append for record
                        st = "ok" if fres["exit_code"] == 0 else "FAILED"
                        yield ("tool", f"verify_command({fcmd.cmd!r}) -> {st} (final)", "tool")
                        if fres["exit_code"] != 0:
                            all_ok = False
                            break
                    if not all_ok:
                        # demote to revision even if LLM said verified (strict)
                        verify_feedback = "Final verification sweep failed — one or more tests did not pass after VERIFIED."
                        yield ("warn", verify_feedback, "warn")
                        if attempt < max_iter - 1:
                            continue
                break

            # NEEDS_REVISION
            verify_feedback = verify_full.replace("NEEDS_REVISION:", "").strip()
            yield ("log", json.dumps({"type": "verify_decision", "decision": "NEEDS_REVISION",
                                      "feedback": verify_full, "iteration": attempt + 1}), "log")
            if attempt < max_iter - 1:
                yield ("warn", f"Needs revision ({attempt + 1}/{max_iter}) — {verify_feedback[:200]}", "warn")
            else:
                yield ("warn", f"Safety cap ({max_iter} iterations) reached — review manually.", "warn")

    # ── model runners ─────────────────────────────────────────────────────────

    def _auto_compress_if_needed(
        self, history: list[dict], model: str, num_ctx: int,
        _pending_events: list | None = None,
    ) -> bool:
        """Compress history in-place if it's approaching the context limit.

        If _pending_events is provided, appends a (source, content, kind) tuple
        so the caller can yield it — making compression visible in the UI.
        """
        if not self.config.auto_compress:
            return False
        tok = estimate_messages(history)
        threshold = int(num_ctx * self.config.compress_threshold)
        if tok > threshold or len(history) > self.config.max_history_pairs * 2:
            compressed = self._compressor.compress_history(history, keep_pairs=4)
            if compressed and _pending_events is not None:
                short_model = model.split("/")[-1].split(":")[0]
                _pending_events.append(
                    ("compress",
                     f"Auto-compressed {short_model} history  {tok:,} tokens → keep 4 pairs",
                     "info")
                )
            return compressed
        return False

    def _run_planner(
        self, msg: str, route: Route, keep_alive: str = "0",
        scope: ScopeType = "targeted", task: str = "", is_retry: bool = False,
    ) -> Generator:
        needs_thinking = _needs_thinking(task, scope, route, is_retry, 0)
        planner_think: bool | None = None if needs_thinking else False
        p = load_prompts()

        # Larger budget on retry+CoT so thinking doesn't crowd out the actual plan
        if needs_thinking and is_retry:
            num_predict = p.get_predict("plan_retry")
        elif needs_thinking:
            num_predict = p.get_predict("plan_heavy")
        else:
            num_predict = p.get_predict("plan")

        prompt_msgs = [
            {"role": "system", "content": get_system("planner")},
            *self._planner_history,
            {"role": "user", "content": msg},
        ]
        prompt_tok = estimate_messages(prompt_msgs)
        force_max = getattr(self.config, "always_max_ctx", True)
        num_ctx = choose_num_ctx(
            model=self.config.planner_model,
            prompt_tokens=prompt_tok,
            max_new_tokens=num_predict,
            vram_ceiling=min(self.config.vram_ctx_ceiling, self.config.planner_num_ctx),
            force_max_context=force_max,
        )
        num_ctx = min(num_ctx, model_max_ctx(self.config.planner_model))

        _compress_events: list = []
        self._auto_compress_if_needed(self._planner_history, self.config.planner_model, num_ctx,
                                      _pending_events=_compress_events)
        for _ev in _compress_events:
            yield _ev
        yield ("ctx_info", f"planner|{prompt_tok}|{num_ctx}", "info")

        yield ("log", json.dumps({
            "type": "model_input", "role": "planner",
            "model": self.config.planner_model,
            "num_ctx": num_ctx, "prompt_tokens_est": prompt_tok,
            "think": needs_thinking,
        }), "log")

        thinking = ""
        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.planner_model,
                messages=prompt_msgs,
                temperature=self.config.planner_temp,
                num_predict=num_predict,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                think=planner_think,
            ):
                if kind == "think":
                    thinking += token
                    yield ("planner", token, "think")   # stream so user sees progress
                elif kind == "usage":
                    self._record_usage(token)
                    yield ("usage", token, "usage")
                else:
                    full += token
                    yield ("planner", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Planner interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Planner failed: {exc}", "error")

        # Guard: if CoT exhausted the token budget before producing any output,
        # re-run immediately with think=False using the same prompt.
        if not full.strip() and needs_thinking:
            yield ("warn", "CoT used all token budget — retrying without chain-of-thought", "warn")
            try:
                for token, kind in self.bridge.chat(
                    model=self.config.planner_model,
                    messages=prompt_msgs,
                    temperature=self.config.planner_temp,
                    num_predict=p.get_predict("plan"),
                    num_ctx=num_ctx,
                    keep_alive=keep_alive,
                    think=False,
                ):
                    if kind == "usage":
                        self._record_usage(token)
                        yield ("usage", token, "usage")
                    elif kind != "think":
                        full += token
                        yield ("planner", token, kind)
            except Exception as exc:
                yield ("error", f"Planner no-think retry failed: {exc}", "error")

        if thinking:
            yield ("log", json.dumps({"type": "model_thinking", "role": "planner",
                                       "chars": len(thinking)}), "log")
        yield ("log", json.dumps({"type": "model_output", "role": "planner",
                                   "chars": len(full)}), "log")

        self._planner_history.append({"role": "user",      "content": msg})
        self._planner_history.append({"role": "assistant", "content": full})
        return full

    def _run_coder(self, msg: str) -> Generator:
        prompt_msgs = [
            {"role": "system", "content": get_system("coder")},
            *self._coder_history,
            {"role": "user", "content": msg},
        ]
        prompt_tok = estimate_messages(prompt_msgs)
        force_max = getattr(self.config, "always_max_ctx", True)
        p = load_prompts()
        num_ctx = choose_num_ctx(
            model=self.config.coder_model,
            prompt_tokens=prompt_tok,
            max_new_tokens=p.get_predict("code"),
            vram_ceiling=min(self.config.vram_ctx_ceiling, self.config.coder_num_ctx),
            force_max_context=force_max,
        )
        num_ctx = min(num_ctx, model_max_ctx(self.config.coder_model))

        _compress_events: list = []
        self._auto_compress_if_needed(self._coder_history, self.config.coder_model, num_ctx,
                                      _pending_events=_compress_events)
        for _ev in _compress_events:
            yield _ev
        yield ("ctx_info", f"coder|{prompt_tok}|{num_ctx}", "info")

        yield ("log", json.dumps({
            "type": "model_input", "role": "coder",
            "model": self.config.coder_model,
            "num_ctx": num_ctx, "prompt_tokens_est": prompt_tok,
        }), "log")

        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.coder_model,
                messages=prompt_msgs,
                temperature=self.config.coder_temp,
                num_predict=p.get_predict("code"),
                num_ctx=num_ctx,
                think=False,
            ):
                if kind == "usage":
                    self._record_usage(token)
                    yield ("usage", token, "usage")
                elif kind != "think":
                    full += token
                    yield ("coder", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Coder interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Coder failed: {exc}", "error")

        yield ("log", json.dumps({"type": "model_output", "role": "coder",
                                   "chars": len(full)}), "log")

        self._coder_history.append({"role": "user",      "content": msg})
        self._coder_history.append({"role": "assistant", "content": full})
        return full

    def _run_verifier(self, msg: str) -> Generator:
        prompt_msgs = [
            {"role": "system", "content": get_system("verifier")},
            {"role": "user",   "content": msg},
        ]
        prompt_tok = estimate_messages(prompt_msgs)
        force_max = getattr(self.config, "always_max_ctx", True)
        p = load_prompts()
        num_ctx = choose_num_ctx(
            model=self.config.verifier_model,
            prompt_tokens=prompt_tok,
            max_new_tokens=p.get_predict("verify"),
            vram_ceiling=min(self.config.vram_ctx_ceiling, self.config.verifier_num_ctx),
            force_max_context=force_max,
        )
        num_ctx = min(num_ctx, model_max_ctx(self.config.verifier_model))

        yield ("ctx_info", f"verifier|{prompt_tok}|{num_ctx}", "info")
        yield ("log", json.dumps({
            "type": "model_input", "role": "verifier",
            "model": self.config.verifier_model, "num_ctx": num_ctx,
        }), "log")

        full = ""
        try:
            for token, kind in self.bridge.chat(
                model=self.config.verifier_model,
                messages=prompt_msgs,
                temperature=self.config.verifier_temp,
                num_predict=p.get_predict("verify"),
                num_ctx=num_ctx,
                keep_alive="0",
                think=False,
            ):
                if kind == "usage":
                    self._record_usage(token)
                elif kind != "think":
                    full += token
                    yield ("verify", token, kind)
        except KeyboardInterrupt:
            yield ("warn", "Verifier interrupted.", "warn")
        except Exception as exc:
            yield ("error", f"Verifier failed: {exc}", "error")
            full = "NEEDS_REVISION: verifier error — treat as unverified"

        yield ("log", json.dumps({"type": "model_output", "role": "verifier",
                                   "chars": len(full)}), "log")
        return full

    # ── build/test runner ─────────────────────────────────────────────────────

    def _verification_commands(self, route: Route) -> list[VerificationCommand]:
        commands: list[VerificationCommand] = []

        if self.config.build_cmd:
            commands.append(VerificationCommand(
                cmd=self.config.build_cmd,
                purpose="configured build verification",
                source="/build",
                is_test=_is_test_verification_command(self.config.build_cmd),
            ))
        if self.config.test_cmd:
            commands.append(VerificationCommand(
                cmd=self.config.test_cmd,
                purpose="configured test verification",
                source="/test",
                is_test=True,
            ))

        if commands:
            return _dedupe_verification_commands(commands)

        if not self.config.auto_discover_verification:
            return []

        root = self._indexer.root
        commands.extend(_extract_readme_verification_commands(root))
        commands.extend(_infer_project_verification_commands(root))

        safe = [item for item in commands if _is_safe_verification_command(item.cmd)]
        return _dedupe_verification_commands(safe)

    def _run_verification_command(self, command: VerificationCommand) -> dict:
        started = time.monotonic()
        try:
            result = subprocess.run(
                command.cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.verification_timeout_seconds,
                cwd=str(self._indexer.root),
            )
            output = (result.stdout or "") + (result.stderr or "")
            exit_code = int(result.returncode)
        except Exception as exc:
            output = str(exc)
            exit_code = 1

        return {
            "cmd": command.cmd,
            "purpose": command.purpose,
            "source": command.source,
            "is_test": command.is_test,
            "exit_code": exit_code,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output": output,
        }

    def _run_build_test(self) -> tuple[bool, str, str] | None:
        commands = self._verification_commands(Route.DIAGNOSTIC_LOOP)
        if not commands:
            return None
        result = self._run_verification_command(commands[0])
        return (result["exit_code"] == 0, result["output"], result["cmd"])

    # ── rollback ──────────────────────────────────────────────────────────────

    def _rollback(
        self, snapshots: dict[str, str | None], project_root: Path
    ) -> None:
        """Restore files to their state before this run (called on UNSOLVABLE)."""
        for rel_path, original_content in snapshots.items():
            target = (project_root / rel_path).resolve()
            try:
                target.relative_to(project_root.resolve())
                if original_content is None:
                    if target.exists():
                        target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(original_content, encoding="utf-8")
            except (ValueError, OSError):
                pass

    # ── session persistence ────────────────────────────────────────────────────

    def _save_iteration_state(
        self,
        iteration: int,
        plan: str,
        feedback: str,
        proof: ProvenWork | None = None,
    ) -> None:
        state_dir = _project_dir()
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "iteration": iteration,
                "plan": plan[:2000],
                "last_feedback": feedback[:2000],
                "pending_files": [p for p, _ in self.pending_file_changes],
                "proven_work": proof.to_dict() if proof else None,
                "session_usage": self._session_usage,
            }
            (state_dir / "session.json").write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def _record_solution(
        self,
        files_changed: list[str],
        iteration: int,
        task: str,
        plan: str,
        coder_output: str,
        proof: ProvenWork | None = None,
    ) -> None:
        if not files_changed:
            return
        # Semantic memory extraction via compressor
        mem_entries = self._compressor.generate_memory(task, plan, coder_output, files_changed)
        self._memory.add_from_compress(mem_entries, tier="project")
        # Also record the basic solution fact
        proof_cmds = []
        if proof:
            proof_cmds = [item["cmd"] for item in proof.commands if item.get("exit_code") == 0]
        suffix = f"; PROVEN_WORK: {', '.join(proof_cmds[:3])}" if proof_cmds else ""
        self._memory.add(MemoryEntry(
            category="solution",
            content=f"Solved in {iteration} iteration(s). Files: {', '.join(files_changed[:6])}{suffix}",
            tags=["verified", "proven_work"] if proof_cmds else ["verified"],
        ), "project")

    # ── public API ────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self._planner_history.clear()
        self._coder_history.clear()
        self._memory.clear_task()
        self._memory.clear_session()

    def rebuild_index(self) -> int:
        self._indexer.invalidate()
        return len(self._indexer.build_index())

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def indexer(self) -> ProjectIndexer:
        return self._indexer
