"""Retry/session-bookkeeping mixin — rollback, iteration-state persistence,
and post-success memory recording.

Split out of ``KratosAgent`` (mixed in via
``class KratosAgent(_RoleRunnerMixin, _RetryMixin)``); these methods operate
on the same ``self`` (config, project_dir helpers, compressor, memory).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from ..config import _project_dir
from ..memory import MemoryEntry
from ..verification import ProvenWork
from ..planning import ExecutionPlan


class _RetryMixin:
    """Provides ``_rollback``, ``_save_iteration_state``, ``_record_solution``."""

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

    def _record_planner_artifact(
        self,
        task: str,
        route: str,
        iteration: int,
        plan_markdown: str,
        plan_state: ExecutionPlan | None = None,
    ) -> Path | None:
        """Persist the exact planner Markdown and a short project-memory summary."""
        plan_markdown = (plan_markdown or "").rstrip()
        if not plan_markdown:
            return None

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        safe_route = re.sub(r"[^A-Za-z0-9._-]+", "_", route or "plan").strip("_") or "plan"
        plans_dir = _project_dir() / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plans_dir / f"{stamp}_{safe_route}_iter{iteration:02d}.md"
        plan_path.write_text(plan_markdown + "\n", encoding="utf-8")

        checklist = ""
        if plan_state and getattr(plan_state, "items", None):
            titles = [item.title.strip() for item in plan_state.items[:3] if item.title.strip()]
            checklist = "; ".join(titles)

        summary_bits = [f"Planner saved Markdown plan for task: {task[:120].strip()}"]
        if checklist:
            summary_bits.append(f"Checklist: {checklist[:160]}")
        summary_bits.append(f"Artifact: {plan_path.name}")
        self._memory.add(
            MemoryEntry(
                category="decision",
                content=" | ".join(summary_bits)[:220],
                tags=["planner", "markdown", "plan"],
            ),
            "project",
        )
        return plan_path

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
