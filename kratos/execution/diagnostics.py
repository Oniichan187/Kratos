"""FailureDiagnoser + RepairTracker — turn raw test/build output into a
concrete, actionable fix instruction, and stop the loop from re-running the
same failing command forever.

Sessions that shaped this module:
  * 00-57-15: circular import re-run 94× with only a raw traceback as feedback.
  * 13-12-22: 4B coder wrote a wrong override signature (handle_starttag) →
    TypeError reported only as a downstream AssertionError.
  * 19-59-46: 7B coder GUESSED the HTML format (<div>/<span>) instead of reading
    the fixture (<section>/<p>) → re.search(...).group() on None.

Pure stdlib, no project imports → trivially unit-testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "Diagnosis",
    "FailureDiagnoser",
    "RepairTracker",
    "diagnose_command",
]


@dataclass
class Diagnosis:
    category: str
    signature: str
    summary: str
    fix_instruction: str
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "signature": self.signature,
            "summary": self.summary,
            "fix_instruction": self.fix_instruction,
            "files": self.files,
        }

    def as_feedback(self) -> str:
        lines = [f"DIAGNOSIS [{self.category}]: {self.summary}"]
        if self.files:
            lines.append("Involved files: " + ", ".join(self.files))
        lines.append("REQUIRED FIX: " + self.fix_instruction)
        return "\n".join(lines)


# ── regex probes (ordered most-specific → least-specific) ───────────────────

_CIRCULAR_RE = re.compile(
    r"cannot import name ['\"]?(?P<name>\w+)['\"]?\s+from\s+partially initialized module\s+"
    r"['\"]?(?P<module>[\w.]+)['\"]?",
    re.I,
)
_CIRCULAR_HINT_RE = re.compile(r"most likely due to a circular import", re.I)
_CANNOT_IMPORT_RE = re.compile(
    r"cannot import name ['\"]?(?P<name>\w+)['\"]?\s+from\s+['\"]?(?P<module>[\w.]+)['\"]?",
    re.I,
)
_MODULE_NOT_FOUND_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError): No module named ['\"]?(?P<module>[\w.]+)['\"]?",
    re.I,
)
_SYNTAX_RE = re.compile(
    r"(?P<kind>SyntaxError|IndentationError|TabError):\s*(?P<msg>.+)", re.I,
)
_FILE_LINE_RE = re.compile(r'File "(?P<file>[^"]+\.py)", line (?P<line>\d+)')
_PYTEST_FILE_RE = re.compile(r"(?P<file>(?:[A-Za-z]:)?[\w./\\-]+\.py):(?P<line>\d+):")
_NOT_IMPLEMENTED_RE = re.compile(r"\bNotImplementedError\b", re.I)
_ATTRIBUTE_RE = re.compile(
    r"AttributeError: (?:module|'[\w.]+' object) .*?has no attribute ['\"]?(?P<attr>\w+)['\"]?",
    re.I,
)
_NAME_ERROR_RE = re.compile(r"NameError: name ['\"]?(?P<name>\w+)['\"]? is not defined", re.I)
# A lookup/parse returned None and the code used it (re.search(...).group() with no
# match, etc.). With weak coder models this almost always means the INPUT FORMAT was
# guessed wrong (session 19-59-46) → steer it to READ the real input.
_NONETYPE_RE = re.compile(
    r"(?:AttributeError: 'NoneType' object has no attribute '(?P<attr>\w+)'"
    r"|'NoneType' object is not subscriptable"
    r"|'NoneType' object is not iterable)",
    re.I,
)
# Wrong method/function signature — the #1 weak-model mistake when overriding a
# base-class method (e.g. html.parser.HTMLParser.handle_starttag(self, tag, attrs)).
_ARG_ERR_RE = re.compile(
    r"TypeError:\s*(?P<fn>[\w.]+)\(\)\s*(?P<detail>"
    r"missing \d+ required positional argument[^\n]*|"
    r"takes \d+ positional arguments? but \d+ (?:was|were) given|"
    r"got an unexpected keyword argument[^\n]*|"
    r"missing \d+ required keyword-only argument[^\n]*)",
    re.I,
)
_TYPE_ERROR_RE = re.compile(r"TypeError: (?P<msg>.+)", re.I)
_ASSERT_RE = re.compile(r"^E\s+(?:assert|AssertionError)\b.*", re.I | re.M)
_PYTEST_SUMMARY_RE = re.compile(r"(?P<failed>\d+) failed", re.I)
_COLLECT_ERROR_RE = re.compile(r"errors? during collection|ERROR collecting", re.I)


def _modules_to_files(*modules: str) -> list[str]:
    out: list[str] = []
    for mod in modules:
        if not mod:
            continue
        rel = mod.replace(".", "/") + ".py"
        if rel not in out:
            out.append(rel)
    return out


def _tail(text: str, n: int = 4000) -> str:
    return text[-n:] if len(text) > n else text


def _shorten_path(path: str) -> str:
    p = path.replace("\\", "/")
    is_absolute = p.startswith("/") or re.match(r"^[A-Za-z]:", p) is not None
    if not is_absolute:
        return p
    parts = [seg for seg in p.split("/") if seg and not re.match(r"^[A-Za-z]:$", seg)]
    return "/".join(parts[-3:]) if len(parts) > 3 else "/".join(parts)


def diagnose_command(result: dict) -> "Diagnosis | None":
    if result is None:
        return None
    exit_code = result.get("exit_code")
    if exit_code == 0:
        return None
    cmd = result.get("cmd", "")
    output = (result.get("output")
              or ((result.get("stdout", "") or "") + "\n" + (result.get("stderr", "") or "")))
    return FailureDiagnoser().diagnose(output, cmd=cmd, exit_code=exit_code,
                                       timed_out=bool(result.get("timed_out")),
                                       blocked=bool(result.get("blocked")),
                                       block_reason=result.get("block_reason", ""))


class FailureDiagnoser:
    """Stateless analyzer: command output → :class:`Diagnosis`."""

    def diagnose(self, output: str, cmd: str = "", exit_code: "int | None" = None,
                 timed_out: bool = False, blocked: bool = False,
                 block_reason: str = "") -> "Diagnosis | None":
        text = output or ""

        if blocked:
            return Diagnosis(
                category="blocked_command",
                signature=f"blocked:{block_reason}",
                summary=f"Command was blocked by the safety guard: {block_reason}",
                fix_instruction=(
                    "Do NOT attempt this command. Choose a safe, non-destructive "
                    "alternative that stays inside the project."
                ),
            )

        if timed_out:
            return Diagnosis(
                category="timeout",
                signature=f"timeout:{cmd}",
                summary=f"`{cmd}` timed out.",
                fix_instruction=(
                    "The command did not finish in time. Look for an infinite loop, "
                    "a blocking input() call, or a hung network request and remove it."
                ),
            )

        # ── 1. circular import ──────────────────────────────────────────────
        m = _CIRCULAR_RE.search(text)
        if m or _CIRCULAR_HINT_RE.search(text):
            name = m.group("name") if m else ""
            module = m.group("module") if m else ""
            files = self._collect_py_files(text) or _modules_to_files(module)
            file_hint = " and ".join(files[:2]) if files else "the two modules involved"
            return Diagnosis(
                category="circular_import",
                signature=f"circular_import:{module}:{name}",
                summary=(
                    f"Circular import: module {module!r} is only partially "
                    f"initialized when {name!r} is imported from it."
                    if module else "Circular import between two modules."
                ),
                fix_instruction=(
                    f"Two modules import from each other at module top level ({file_hint}). "
                    "REORDERING the import lines or adding `# noqa` will NOT fix this — the "
                    "cycle must be BROKEN. Do ONE of: (a) remove the cross-import entirely and "
                    "keep each symbol in a single owning module, (b) move the shared symbol "
                    "into a third module both can import, or (c) convert one side to a lazy "
                    "import done *inside* the function that needs it. After the change, neither "
                    "module may import the other at top level."
                ),
                files=files,
            )

        # ── 2. cannot import name (non-circular) ────────────────────────────
        m = _CANNOT_IMPORT_RE.search(text)
        if m:
            name, module = m.group("name"), m.group("module")
            files = _modules_to_files(module)
            return Diagnosis(
                category="import_name_error",
                signature=f"cannot_import:{module}:{name}",
                summary=f"{name!r} cannot be imported from {module!r} — it is not defined there.",
                fix_instruction=(
                    f"Define {name!r} in {module!r}, or fix the import to point at the module "
                    f"that actually defines {name!r}. Check for typos in the symbol name."
                ),
                files=files,
            )

        # ── 3. module not found ─────────────────────────────────────────────
        m = _MODULE_NOT_FOUND_RE.search(text)
        if m:
            module = m.group("module")
            return Diagnosis(
                category="module_not_found",
                signature=f"module_not_found:{module}",
                summary=f"No module named {module!r}.",
                fix_instruction=(
                    f"Either the module {module!r} does not exist (create it / fix the import "
                    "path), or the test is being run from the wrong working directory. For a "
                    "nested package layout, run the test command from the directory that "
                    "contains the package (the folder with pyproject.toml/setup.py)."
                ),
                files=_modules_to_files(module),
            )

        # ── 4. syntax / indentation ─────────────────────────────────────────
        m = _SYNTAX_RE.search(text)
        if m:
            files = self._collect_py_files(text)
            loc = files[0] if files else "the reported file"
            return Diagnosis(
                category="syntax_error",
                signature=f"syntax:{m.group('kind')}:{loc}",
                summary=f"{m.group('kind')}: {m.group('msg').strip()[:120]}",
                fix_instruction=(
                    f"Fix the {m.group('kind')} in {loc}. Read the exact reported line and the "
                    "lines around it; this is a hard parse error, so the file cannot run at all "
                    "until it is valid Python."
                ),
                files=files,
            )

        # ── 5. NotImplementedError / empty stub ─────────────────────────────
        if _NOT_IMPLEMENTED_RE.search(text):
            files = self._collect_py_files(text)
            return Diagnosis(
                category="not_implemented",
                signature="not_implemented:" + (files[0] if files else "?"),
                summary="A function still raises NotImplementedError (it is an unfinished stub).",
                fix_instruction=(
                    "Implement the function body for real. A docstring or `pass` or `...` is not "
                    "an implementation — write the logic the test expects and return the value."
                ),
                files=files,
            )

        # ── 6. NameError ────────────────────────────────────────────────────
        m = _NAME_ERROR_RE.search(text)
        if m:
            files = self._collect_py_files(text)
            return Diagnosis(
                category="name_error",
                signature=f"name_error:{m.group('name')}",
                summary=f"name {m.group('name')!r} is not defined.",
                fix_instruction=(
                    f"Define or import {m.group('name')!r} before it is used, or fix the typo."
                ),
                files=files,
            )

        # ── 6b. parse/lookup returned None (wrong input-format assumption) ───
        m = _NONETYPE_RE.search(text)
        if m:
            files = self._collect_py_files(text)
            return Diagnosis(
                category="parse_returned_none",
                signature="parse_none:" + (files[0] if files else "?"),
                summary="A lookup/parse returned None and was then used (e.g. re.search(...).group() with no match).",
                fix_instruction=(
                    "Your code looked for something and found nothing, then used the None result. "
                    "This almost always means the INPUT FORMAT you assumed is wrong. Do NOT guess "
                    "the structure: open and READ the actual input/fixture file first "
                    "(### READ: <path>), look at its real tags/keys/columns, and base your parsing "
                    "on what is actually there. Also guard each lookup (check for None) before use."
                ),
                files=files,
            )

        # ── 6c. generic AttributeError ──────────────────────────────────────
        m = _ATTRIBUTE_RE.search(text)
        if m:
            files = self._collect_py_files(text)
            return Diagnosis(
                category="attribute_error",
                signature=f"attribute_error:{m.group('attr')}",
                summary=f"missing attribute {m.group('attr')!r}.",
                fix_instruction=(
                    f"The object does not have {m.group('attr')!r}. Check the type you are "
                    "calling it on — you may be treating a str like a Path (or vice versa). "
                    "Make the parameter type and its usage consistent."
                ),
                files=files,
            )

        # ── 7. wrong signature (root cause, ranked ABOVE downstream assertion) ─
        m = _ARG_ERR_RE.search(text)
        if m:
            files = self._collect_py_files(text)
            fn = m.group("fn")
            detail = m.group("detail").strip()
            return Diagnosis(
                category="signature_mismatch",
                signature=f"signature_mismatch:{fn}:{detail[:40]}",
                summary=f"{fn}() is called with a signature it does not accept ({detail[:80]}).",
                fix_instruction=(
                    f"The definition of {fn}() does not match how it is called. If you are "
                    "OVERRIDING a base-class or stdlib method (e.g. html.parser.HTMLParser."
                    "handle_starttag(self, tag, attrs)), the override MUST use the EXACT same "
                    "parameters as the base method — do not add, remove, or rename parameters. "
                    "Otherwise fix the call site to pass the right arguments. Match the "
                    "parameter list exactly."
                ),
                files=files,
            )

        # ── 8. generic TypeError (runtime — ranked above assertion) ──────────
        m = _TYPE_ERROR_RE.search(text)
        if m:
            files = self._collect_py_files(text)
            return Diagnosis(
                category="type_error",
                signature="type_error:" + m.group("msg").strip()[:80],
                summary=f"TypeError: {m.group('msg').strip()[:120]}",
                fix_instruction=(
                    "A value has the wrong type for how it is used (a common case: a parameter "
                    "annotated as Path but called with a str and used with str methods). Make the "
                    "declared type and the actual usage agree."
                ),
                files=files,
            )

        # ── 9. assertion failure (logic wrong, code runs, no exception) ──────
        am = _ASSERT_RE.search(text)
        if am:
            files = self._collect_py_files(text)
            return Diagnosis(
                category="assertion_failure",
                signature="assertion:" + am.group(0).strip()[:80],
                summary="A test assertion failed — the code runs but produces the wrong value.",
                fix_instruction=(
                    "Read the assertion's expected-vs-actual values in the output and correct the "
                    "logic so the produced value matches what the test expects. Do not change the "
                    "test."
                ),
                files=files,
            )

        # ── 10. pytest collection error / generic non-zero ──────────────────
        if exit_code == 2 and _COLLECT_ERROR_RE.search(text):
            files = self._collect_py_files(text)
            return Diagnosis(
                category="collection_error",
                signature="collection_error:" + (files[0] if files else "?"),
                summary="pytest could not even collect the tests (exit code 2).",
                fix_instruction=(
                    "This is an import-time error, not a failing assertion — the test module "
                    "cannot be imported. Fix the import error in the module(s) listed in the "
                    "traceback before expecting any test to run."
                ),
                files=files,
            )

        sm = _PYTEST_SUMMARY_RE.search(text)
        if sm:
            files = self._collect_py_files(text)
            return Diagnosis(
                category="tests_failed",
                signature="tests_failed:" + sm.group("failed"),
                summary=f"{sm.group('failed')} test(s) failed.",
                fix_instruction=(
                    "Read the failing test names and their assertion output, then fix the "
                    "implementation so each expectation holds."
                ),
                files=files,
            )

        return Diagnosis(
            category="unknown_failure",
            signature=f"exit:{exit_code}:" + _tail(text, 60).strip()[:60],
            summary=f"`{cmd}` failed with exit code {exit_code}.",
            fix_instruction=(
                "Read the command output carefully, find the first concrete error line, and fix "
                "its root cause before re-running."
            ),
            files=self._collect_py_files(text),
        )

    @staticmethod
    def _collect_py_files(text: str) -> list[str]:
        source: list[str] = []
        tests: list[str] = []
        seen: set[str] = set()
        for rx in (_FILE_LINE_RE, _PYTEST_FILE_RE):
            for m in rx.finditer(text):
                f = m.group("file").replace("\\", "/")
                low = f.lower()
                if (f.startswith("<") or "site-packages" in low
                        or "/lib/python" in low or "/lib/" in low
                        or re.search(r"python\d", low) or "/dist-packages/" in low):
                    continue
                f = _shorten_path(f)
                if f in seen:
                    continue
                seen.add(f)
                base = f.rsplit("/", 1)[-1]
                if base.startswith("test_") or "/tests/" in f or f.startswith("tests/"):
                    tests.append(f)
                else:
                    source.append(f)
        return (source + tests)[:5]


class RepairTracker:
    """Tracks failure signatures across repair iterations to detect a stall."""

    def __init__(self, stall_threshold: int = 2) -> None:
        self.stall_threshold = max(1, stall_threshold)
        self._counts: dict[str, int] = {}
        self._sequence: list[str] = []

    def register(self, signature: str) -> int:
        if not signature:
            signature = "<empty>"
        self._counts[signature] = self._counts.get(signature, 0) + 1
        self._sequence.append(signature)
        return self._counts[signature]

    def count(self, signature: str) -> int:
        return self._counts.get(signature, 0)

    def is_stalled(self, signature: str) -> bool:
        return self.count(signature) >= self.stall_threshold

    def repeated_immediately(self, signature: str) -> bool:
        return len(self._sequence) >= 2 and self._sequence[-2] == signature

    def escalation_note(self, signature: str) -> str:
        n = self.count(signature)
        return (
            f"STALL WARNING: this exact failure has now occurred {n} times. The previous "
            "fix attempt did NOT change the outcome — stop repeating it. Re-read the involved "
            "files in full, change your APPROACH (not just a line), and address the ROOT CAUSE "
            "named in the diagnosis above before running the command again."
        )

    def reset(self) -> None:
        self._counts.clear()
        self._sequence.clear()
