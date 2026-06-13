"""Reporter — final answers built ONLY from verified evidence.

This module is the hard gate against the worst agent failure mode Kratos
showed in practice: *sounding* successful while ``files_changed`` was empty.

Rules enforced here (and tested in tests/test_reporter_gate.py):

  1. If the task required code changes and no REAL file change is verifiable
     on disk → status can never be SUCCESS. The report says
     "Keine echten Dateiänderungen erkannt".
  2. "Tests bestanden" may only appear when a test command actually ran
     (evidence: recorded command with is_test + exit_code). Otherwise the
     report says "Tests nicht ausgeführt" with the reason.
  3. The diff summary comes from real before/after content comparison
     (snapshots vs disk) — never invented. No changes → "Keine Dateien
     geändert."
  4. No-op writes (identical content re-written) do NOT count as changes.

The Reporter never receives free-form model text as a fact source — only the
structured ``ProvenWork`` evidence and the snapshot map maintained by the
runtime.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .verification import ProvenWork
from .execution.diffing import diff_stats, git_changed_files, git_diff_stat

__all__ = ["FileChangeEvidence", "FinalReport", "verify_files_changed", "build_final_report"]

NO_REAL_CHANGES_MSG = "Keine echten Dateiänderungen erkannt"
TESTS_NOT_RUN_MSG = "Tests nicht ausgeführt"
# (status gating + data-driven facts; see docs/agent_verification.md)


@dataclass
class FileChangeEvidence:
    path: str
    kind: str            # "created" | "modified" | "deleted" | "unchanged" | "missing"
    before_hash: str = ""
    after_hash: str = ""
    lines_added: int = 0
    lines_removed: int = 0

    @property
    def is_real_change(self) -> bool:
        return self.kind in ("created", "modified", "deleted")


@dataclass
class FinalReport:
    status: str                      # "SUCCESS" | "PARTIAL" | "FAILED"
    changed_files: list[FileChangeEvidence] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    commands: list[dict] = field(default_factory=list)
    tests_ran: bool = False
    tests_passed: bool | None = None  # None == not run
    tests_detail: str = ""
    web_requested: bool = False
    web_research_used: bool = False
    web_sources: list[str] = field(default_factory=list)
    diff_summary: str = ""
    git_changed: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    diagnosis: str = ""

    def to_markdown(self) -> str:
        lines = ["## Ergebnis", f"Status:\n- {self.status}", "", "## Geänderte Dateien"]
        real = [c for c in self.changed_files if c.is_real_change]
        if real:
            for c in real:
                delta = ""
                if c.lines_added or c.lines_removed:
                    delta = f" (+{c.lines_added}/-{c.lines_removed} Zeilen)"
                lines.append(f"- {c.path}: {c.kind}{delta}")
        else:
            lines.append("- Keine Dateien geändert.")
        evidence_paths = {c.path for c in real}
        extra_git = [f for f in self.git_changed if f not in evidence_paths]
        if extra_git:
            lines.append(
                "- (git-Querprüfung meldet zusätzliche geänderte Dateien außerhalb "
                f"der Agent-Evidenz: {', '.join(extra_git[:10])})"
            )
        lines += ["", "## Gefundene Probleme"]
        lines += [f"- {p}" for p in self.problems] or ["- (keine dokumentiert)"]
        lines += ["", "## Ausgeführte Befehle", "| Befehl | Shell | Exitcode | Ergebnis |", "|---|---|---:|---|"]
        if self.commands:
            for c in self.commands:
                ec = c.get("exit_code", "?")
                ok = "ok" if ec == 0 else ("BLOCKIERT" if c.get("blocked") else "FEHLER")
                lines.append(f"| `{c.get('cmd', '?')}` | {c.get('shell', '?')} | {ec} | {ok} |")
        else:
            lines.append("| (keine Befehle ausgeführt) | – | – | – |")
        lines += ["", "## Tests"]
        lines.append(f"- Ausgeführt: {'Ja' if self.tests_ran else 'Nein'}")
        if self.tests_ran:
            lines.append(f"- Ergebnis: {'bestanden' if self.tests_passed else 'FEHLGESCHLAGEN'}")
            if self.tests_detail:
                lines.append(f"- Detail: {self.tests_detail}")
        else:
            lines.append(
                f"- {TESTS_NOT_RUN_MSG} (Grund: "
                f"{self.tests_detail or 'kein Testbefehl ausgeführt'})"
            )
        if self.diagnosis:
            lines += ["", "## Diagnose des letzten Fehlers", f"- {self.diagnosis}"]
        lines += ["", "## Websuche/Webscraping",
                  f"- Verlangt: {'Ja' if self.web_requested else 'Nein'}",
                  f"- Durchgeführt: {'Ja' if self.web_research_used else 'Nein'}"]
        if self.web_sources:
            lines.append("- Quellen:")
            for src in self.web_sources:
                lines.append(f"  - {src}")
        elif self.web_requested and not self.web_research_used:
            lines.append("- Quellen: keine — Web search provider not configured / nicht durchgeführt")
        lines += ["", "## Diff-Zusammenfassung",
                  f"- {self.diff_summary or 'Kein Diff vorhanden.'}",
                  "", "## Offene Einschränkungen"]
        lines += [f"- {l}" for l in self.limitations] or ["- keine"]
        return "\n".join(lines)


def _hash_text(text: str | None) -> str:
    if text is None:
        return ""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def verify_files_changed(
    project_root: Path,
    claimed_files: list[str],
    original_snapshots: dict[str, str | None],
) -> list[FileChangeEvidence]:
    """Compare claimed changes against REAL disk state.

    For each claimed path, read the file from disk and compare with the
    pre-run snapshot. Identical content (no-op write) is classified
    ``unchanged`` and does not count as a change. This is the ground truth
    the final gate uses — never the model's claims.
    """
    evidence: list[FileChangeEvidence] = []
    root = project_root.resolve()
    for rel in dict.fromkeys(claimed_files):
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            evidence.append(FileChangeEvidence(rel, "missing"))
            continue
        before = original_snapshots.get(rel)
        after: str | None
        if target.exists() and target.is_file():
            try:
                after = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                after = None
        else:
            after = None

        if before is None and after is not None:
            kind = "created"
        elif before is not None and after is None:
            kind = "deleted"
        elif before is None and after is None:
            kind = "missing"
        elif before == after:
            kind = "unchanged"
        else:
            kind = "modified"

        stats = diff_stats(before, after)
        evidence.append(FileChangeEvidence(
            path=rel, kind=kind,
            before_hash=_hash_text(before), after_hash=_hash_text(after),
            lines_added=stats.added,
            lines_removed=stats.removed,
        ))
    return evidence


def _git_diff_stat(project_root: Path, timeout: int = 15) -> str:
    """Real `git diff HEAD --stat` (staged + unstaged), or '' when unavailable."""
    return git_diff_stat(project_root, timeout)


def _evaluate_tests(proof: ProvenWork) -> tuple[bool, bool | None, str]:
    """(tests_ran, tests_passed, detail) — derived ONLY from command evidence."""
    test_cmds = [c for c in proof.commands if c.get("is_test") and not c.get("skipped")]
    if not test_cmds:
        return False, None, "kein Testbefehl wurde tatsächlich ausgeführt"
    last = test_cmds[-1]
    passed = last.get("exit_code") == 0
    return True, passed, f"`{last.get('cmd')}` → exit={last.get('exit_code')}"


def build_final_report(
    project_root: Path,
    proof: ProvenWork,
    original_snapshots: dict[str, str | None],
    task_requires_changes: bool,
    problems: list[str] | None = None,
    web_sources: list[str] | None = None,
    limitations: list[str] | None = None,
    verifier_accepted: bool = False,
    web_requested: bool = False,
) -> FinalReport:
    """Build the final report. This function IS the anti-hallucination gate:
    status is computed from evidence, not from any claim."""
    evidence = verify_files_changed(project_root, proof.files_changed, original_snapshots)
    real_changes = [e for e in evidence if e.is_real_change]
    tests_ran, tests_passed, tests_detail = _evaluate_tests(proof)

    # Surface a concrete diagnosis for the last failing command (never invented —
    # parsed from the recorded command output).
    diagnosis_text = ""
    if tests_ran and tests_passed is False:
        from .execution.diagnostics import diagnose_command
        failed = [c for c in proof.commands if c.get("exit_code") not in (0, None)]
        if failed:
            diag = diagnose_command(failed[-1])
            if diag is not None:
                diagnosis_text = f"{diag.summary} → {diag.fix_instruction}"

    problems = list(problems or [])
    limitations = list(limitations or [])

    # ── the hard rules ────────────────────────────────────────────────────────
    if task_requires_changes and not real_changes:
        status = "FAILED"
        problems.insert(0, NO_REAL_CHANGES_MSG)
        limitations.insert(0,
            f"{NO_REAL_CHANGES_MSG} — die Aufgabe verlangte Codeänderungen, "
            "aber auf der Festplatte ist keine inhaltliche Änderung nachweisbar.")
    elif tests_ran and tests_passed and verifier_accepted:
        status = "SUCCESS"
    elif real_changes or not task_requires_changes:
        status = "PARTIAL"
        if not tests_ran:
            limitations.append("Tests wurden nicht ausgeführt — Status kann nicht SUCCESS sein.")
        elif not tests_passed:
            limitations.append("Letzter Testlauf fehlgeschlagen.")
        elif not verifier_accepted:
            limitations.append("Verifier hat das Ergebnis nicht bestätigt.")
    else:
        status = "FAILED"

    diff_summary = _git_diff_stat(project_root)
    if not diff_summary and real_changes:
        diff_summary = "; ".join(
            f"{e.path} ({e.kind}, +{e.lines_added}/-{e.lines_removed})" for e in real_changes
        )
    # never invent a diff:
    if not real_changes and not diff_summary:
        diff_summary = "Kein Diff vorhanden."

    return FinalReport(
        status=status,
        changed_files=evidence,
        problems=problems,
        commands=[{k: c.get(k) for k in ("cmd", "shell", "exit_code", "is_test", "blocked")} for c in proof.commands],
        tests_ran=tests_ran,
        tests_passed=tests_passed,
        tests_detail=tests_detail,
        web_requested=web_requested,
        web_research_used=bool(web_sources),
        web_sources=list(web_sources or []),
        diff_summary=diff_summary,
        git_changed=git_changed_files(project_root),
        limitations=limitations,
        diagnosis=diagnosis_text,
    )
