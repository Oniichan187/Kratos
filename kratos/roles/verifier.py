"""Verifier-role message builders."""

from __future__ import annotations

import json

from ..verification import ProvenWork


def _verify_msg(task: str, plan: str, coder_output: str, proof: ProvenWork | None = None) -> str:
    from ..prompts import load_prompts
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
