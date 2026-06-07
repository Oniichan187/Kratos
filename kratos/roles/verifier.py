"""Verifier-role message builders."""

from __future__ import annotations

import json

from ..planning import render_checklist
from ..verification import ProvenWork


def _verify_msg(
    task: str,
    plan: str,
    coder_output: str,
    proof: ProvenWork | None = None,
    plan_items: list | None = None,
) -> str:
    from ..prompts import load_prompts
    pm = load_prompts()
    proof_text = json.dumps(proof.to_dict(), ensure_ascii=False, indent=2) if proof else "{}"
    tl = pm.get_snippet("task_label") or "Task: "
    pl = pm.get_snippet("plan_label") or "Plan given to coder:\n"
    cl = pm.get_snippet("proven_work_label") or "PROVEN_WORK evidence from Kratos runtime:\n"
    checklist = render_checklist(plan_items or [], compact=False) if plan_items else ""
    checklist_label = pm.get_snippet("plan_checklist_audit_label") or "Planner checklist audit:\n"
    parts = [
        f"{tl}{task}",
        f"{pl}{plan[:600]}",
    ]
    if checklist:
        parts.append(f"{checklist_label}{checklist}")
    parts.extend([
        f"Coder output:\n{coder_output[:2000]}",
        f"{cl}{proof_text[:4000]}",
    ])
    return "\n\n".join(parts)
