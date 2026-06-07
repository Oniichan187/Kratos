"""Verification infrastructure: dataclasses, command discovery, proven-work logic."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .prompts import get_marker, get_plan_config, get_toolchain, load_prompts

if TYPE_CHECKING:
    from .config import KratosConfig


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


def _compile_check_cmds(toolchains: set[str], root: Path) -> list[VerificationCommand]:
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
        sln_files = [p for p in root.glob("*.sln") if not _is_noise_path(p)]
        root_projects = [p for p in root.glob("*.csproj") if not _is_noise_path(p)]
        all_projects = [p for p in root.rglob("*.csproj") if not _is_noise_path(p)]
        test_projects = [
            p for p in all_projects
            if "test" in p.relative_to(root).as_posix().lower()
        ]
        target = (
            sln_files[0]
            if sln_files else
            root_projects[0]
            if root_projects else
            test_projects[0]
            if test_projects else
            all_projects[0]
            if all_projects else
            None
        )
        if target is not None:
            cmd = f"dotnet build {_quote_rel(target, root)} --nologo -v:minimal"
        else:
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


_CMD_PATH_TOKEN_RE = re.compile(r'[^\s"\']+[/\\][^\s"\']*\.[a-zA-Z][a-zA-Z0-9]{0,8}')


def _missing_command_paths(cmd: str, root: Path) -> list[str]:
    """Return file-path tokens referenced in *cmd* that don't exist under *root*.

    Catches verify commands the coder points at not-yet-created files (e.g.
    `python -m pytest tests/test_todo_store.py` before that file was written),
    so the runtime can skip them instead of executing a guaranteed failure.
    Bare directory/package targets (e.g. `pytest tests`) have no extension and
    are left alone.
    """
    missing: list[str] = []
    for token in _CMD_PATH_TOKEN_RE.findall(cmd):
        rel = token.strip("'\"").replace("\\", "/")
        if (root / rel).exists():
            continue
        if rel not in missing:
            missing.append(rel)
    return missing


def _is_safe_verification_command(cmd: str) -> bool:
    # Read from JSON on each call (load_prompts() is cached — O(1) after first load).
    meta_chars = tuple(get_toolchain("blocked_verify_chars") or _VERIFY_META_CHARS_FALLBACK)
    safe_prefixes = tuple(get_toolchain("safe_verify_prefixes") or _SAFE_VERIFY_PREFIXES_FALLBACK)
    normalized = " ".join(cmd.strip().split()).lower()
    if not normalized or any(meta in normalized for meta in meta_chars):
        return False
    if normalized.startswith("dotnet run --project"):
        return True
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
    if any((root / n).exists() for n in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini")):
        toolchains.add("python")
    elif (root / "tests").is_dir() or any(root.glob("test_*.py")):
        toolchains.add("python")
    if any(root.glob("*.sln")) or any(root.rglob("*.csproj")):
        toolchains.add("dotnet")
    if (root / "package.json").exists():
        toolchains.add("node")
    if (root / "Cargo.toml").exists():
        toolchains.add("cargo")
    if (root / "go.mod").exists():
        toolchains.add("go")
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
        commands = [
            c for c in commands
            if not re.match(r"^dotnet\s+(build|test)\s*$", c.cmd.strip(), re.I)
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
    if require_test:
        test_cmds = [c for c in proof.commands if c.get("is_test")]
        if not test_cmds:
            return False
        return bool(test_cmds[-1].get("exit_code") == 0)
    return all(item.get("exit_code") == 0 for item in proof.commands)


def _format_proven_work_feedback(proof: ProvenWork, require_test: bool = True) -> str:
    pm = load_prompts()
    if not proof.commands_planned:
        return pm.get_snippet("proven_work_missing_command")
    if not proof.commands:
        return "PROVEN_WORK missing: verification commands were planned but not executed."  # keep one for now, or add another snippet
    failed = [item for item in proof.commands if item.get("exit_code") != 0]
    if failed:
        item = failed[-1]
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
        r'(?:File:|file:)\s*([^\s,\n`"\']+)',
        r'`([^`]+\.[a-zA-Z]{1,8})`',
        r'"([^"]+\.[a-zA-Z]{1,8})"',
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

    def __init__(self, config: KratosConfig, root: Path) -> None:
        self._config = config
        self._root = root
        self.toolchains: set[str] = set()
        self.commands: list[VerificationCommand] = []
        self.compile_commands: list[VerificationCommand] = []

    def discover(self) -> CommandRegistry:
        self.toolchains = _detect_project_toolchains(self._root)
        raw: list[VerificationCommand] = []
        if self._config.build_cmd:
            raw.append(VerificationCommand(self._config.build_cmd, "configured build", "/build", False))
        if self._config.test_cmd:
            raw.append(VerificationCommand(self._config.test_cmd, "configured test", "/test", True))
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
