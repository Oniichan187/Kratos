"""Verification infrastructure: dataclasses, command discovery, proven-work logic."""

from __future__ import annotations

import json
import re
import shutil
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
    # Optional working directory RELATIVE to the project root. Needed for
    # nested project layouts (e.g. the actual Python package lives in
    # `starter_project/` below the folder Kratos was started in).
    cwd: str | None = None


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


def _normalize_runner_command(cmd: str) -> str:
    """Rewrite bare runner shims to their ``python -m`` equivalents.

    On many Windows setups the console-script shim (e.g. ``pytest.exe``) is
    not on PATH even though the package is installed, while ``python -m pytest``
    works perfectly. This normalisation is applied centrally in
    ``_run_verification_command`` so it covers every invocation path (coder-
    emitted ``### VERIFY``, auto-verify, README-discovered, inferred).

    Transformations:
      ``pytest …``      → ``python -m pytest …``
      ``py -m pytest …`` → left alone (already correct)
      ``pip …``         → ``python -m pip …``
      Everything else   → unchanged
    """
    stripped = " ".join(cmd.strip().split())
    lower = stripped.lower()

    # Already in module form — don't double-wrap.
    if lower.startswith(("python -m ", "py -m ")):
        return stripped

    # pytest shim (bare `pytest` or `pytest.exe`)
    if re.match(r"^pytest(?:\.exe)?(?:\s|$)", lower):
        rest = stripped[stripped.lower().index("pytest") + len("pytest"):].lstrip()
        # Strip stray .exe suffix if present on the bare shim token
        rest = re.sub(r"^\.exe\s*", "", rest, flags=re.I)
        return f"python -m pytest {rest}".strip()

    # pip shim
    if re.match(r"^pip(?:\.exe)?(?:\s|$)", lower):
        rest = stripped[stripped.lower().index("pip") + len("pip"):].lstrip()
        rest = re.sub(r"^\.exe\s*", "", rest, flags=re.I)
        return f"python -m pip {rest}".strip()

    return stripped


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
    # strip markdown decoration: `cmd`, **cmd**, *(comment)* suffixes
    s = s.strip("`").strip()
    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        s = s[2:-2].strip()
    s = re.sub(r"^(?:PS>|\$|>|#)\s*", "", s)
    s = s.split(" #", 1)[0].strip()
    return s.strip("`").strip()


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


def _is_safe_verification_command(cmd: str, root: Path | None = None) -> bool:
    # Read from JSON on each call (load_prompts() is cached — O(1) after first load).
    meta_chars = tuple(get_toolchain("blocked_verify_chars") or _VERIFY_META_CHARS_FALLBACK)
    safe_prefixes = tuple(get_toolchain("safe_verify_prefixes") or _SAFE_VERIFY_PREFIXES_FALLBACK)
    normalized = " ".join(cmd.strip().split()).lower()
    if not normalized or any(meta in normalized for meta in meta_chars):
        return False
    if root is not None:
        parts = normalized.split()
        if len(parts) == 2 and parts[0] in {"python", "py"}:
            script = parts[1].replace("\\", "/")
            if script.endswith(".py") and (root / script).exists():
                return True
    if normalized.startswith("dotnet run --project"):
        return True
    return normalized.startswith(safe_prefixes)


_INSPECT_BLOCKED_RE = re.compile(
    r"(?i)(?:\b("
    r"set-content|add-content|out-file|remove-item|move-item|copy-item|new-item|"
    r"set-item|clear-content|rename-item|new-itemproperty|remove-itemproperty|"
    r"invoke-expression|iex|start-process|start-job|invoke-webrequest|"
    r"invoke-restmethod|irm|iwr|curl|wget|invoke-command"
    r")\b|>>|<|`|&)"
)

_INSPECT_ALLOWED_RE = re.compile(
    r"(?i)(?:^|[;|]\s*)(?:"
    r"(?:\$\w+(?:\.\w+)?\s*=\s*[^;|]+;\s*)*"
    r"(?:"
    r"rg\b|"
    r"Get-Content\b|gc\b|"
    r"Get-ChildItem\b|gci\b|"
    r"Get-Item\b|gi\b|"
    r"Resolve-Path\b|"
    r"Test-Path\b|"
    r"git\s+(?:diff|status|show|log)\b|"
    r"type\b|cat\b|ls\b|dir\b|findstr\b|"
    r"Select-Object\b|Select-String\b|"
    r"ForEach-Object\b|Where-Object\b|Sort-Object\b|Measure-Object\b|"
    r"Format-Table\b|Format-List\b"
    r")"
    r")"
)


def _is_safe_inspect_command(cmd: str, root: Path | None = None) -> bool:
    """Return True for read-only shell inspection commands."""
    normalized = " ".join(cmd.strip().split())
    if not normalized:
        return False
    if _INSPECT_BLOCKED_RE.search(normalized):
        return False
    pathish_cmds = (
        "get-content", "gc", "get-childitem", "gci", "get-item", "gi",
        "resolve-path", "test-path", "rg", "findstr", "type", "cat", "ls", "dir",
    )
    if any(token in normalized.lower() for token in pathish_cmds):
        if re.search(r"(?<!\.)\.\.[\\/]", normalized) or re.search(r"\b[A-Za-z]:[\\/]", normalized) or "\\\\" in normalized:
            return False
    return bool(_INSPECT_ALLOWED_RE.search(normalized))


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


def _candidate_project_dirs(root: Path) -> list[Path]:
    """Root plus its immediate non-noise subdirectories.

    Real-world layouts often nest the actual project one level below the
    folder the agent was started in (e.g. ``starter_project/pyproject.toml``).
    Without this, command discovery finds nothing and no test ever runs.
    """
    dirs: list[Path] = [root]
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and not _is_noise_path(child):
                dirs.append(child)
    except OSError:
        pass
    return dirs


def _detect_project_toolchains(root: Path) -> set[str]:
    """Return the build/test toolchains present in the project root or one
    level below it (nested starter-project layouts)."""
    toolchains: set[str] = set()
    for d in _candidate_project_dirs(root):
        if any((d / n).exists() for n in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini")):
            toolchains.add("python")
        elif (d / "tests").is_dir() or any(d.glob("test_*.py")):
            toolchains.add("python")
        if any(d.glob("*.sln")) or (d is root and any(d.rglob("*.csproj"))):
            toolchains.add("dotnet")
        if (d / "package.json").exists():
            toolchains.add("node")
        if (d / "Cargo.toml").exists():
            toolchains.add("cargo")
        if (d / "go.mod").exists():
            toolchains.add("go")
        if (d / "pom.xml").exists():
            toolchains.add("maven")
        if (d / "build.gradle").exists() or (d / "build.gradle.kts").exists():
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
    seen: set[tuple[str, str]] = set()
    unique: list[VerificationCommand] = []
    for item in commands:
        key = (" ".join(item.cmd.split()).lower(), (getattr(item, "cwd", None) or ""))
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
    else:
        # Nested layout: the actual Python project lives one level below the
        # agent's working root (e.g. starter_project/pyproject.toml). Run
        # pytest with that subdirectory as working directory so package
        # imports resolve exactly like `cd <subdir>; python -m pytest`.
        for d in _candidate_project_dirs(root):
            if d == root:
                continue
            sub_proj = any((d / n).exists() for n in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini"))
            sub_tests = (d / "tests").exists() or any(d.glob("test_*.py"))
            if sub_proj and sub_tests:
                rel = d.relative_to(root).as_posix()
                commands.append(VerificationCommand(
                    "python -m pytest", f"python pytest verification (in {rel}/)",
                    rel, True, cwd=rel,
                ))
                break

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
        # Turn the raw traceback into an explicit, actionable instruction. Weak
        # local models cannot reliably act on a bare traceback (the 94×-pytest
        # session proved that) — a named diagnosis + required fix converges them.
        from .execution.diagnostics import diagnose_command
        diag = diagnose_command(item)
        diag_block = f"\n\n{diag.as_feedback()}" if diag else ""
        return (
            f"PROVEN_WORK failed: command `{item.get('cmd')}` exited with {item.get('exit_code')}."
            f"{diag_block}\n\n--- raw output (tail) ---\n"
            f"{str(item.get('output', ''))[-2500:]}"
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
        r'(?<![A-Za-z0-9_])([A-Za-z0-9_.-]+(?:[/\\][A-Za-z0-9_.-]+)+\.[a-zA-Z]{1,8})',
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

    def cwd_for(self, cmd: str) -> str | None:
        """Working directory (relative to project root) for a coder-emitted or
        auto-verify command in NESTED project layouts.

        Root cause this fixes (real session 2026-06-12_20-27): the checklist
        said `VERIFY: python -m pytest`, the runtime executed it in the parent
        root instead of `starter_project/` → every run failed with
        `ModuleNotFoundError: No module named 'mini_agent_check'` and the model
        could never converge. Discovered commands carry the correct cwd —
        reuse it for any command of the same toolchain.
        """
        tc = _command_toolchain(cmd)
        if tc is None:
            return None
        for c in self.commands:
            if getattr(c, "cwd", None) and _command_toolchain(c.cmd) == tc:
                return c.cwd
        return None

    def is_toolchain_mismatch(self, cmd: str) -> bool:
        if not self.toolchains:
            return False
        tc = _command_toolchain(cmd)
        return bool(tc and tc not in self.toolchains)
