"""Structured planner-plan parsing and checklist state helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .verification import _clean_command_line, _extract_plan_steps, _extract_step_file_refs


_CHECKLIST_HEADING_RE = re.compile(
    r"^\s*#{1,3}\s*(?:user\s+)?(?:plan\s+)?(?:check\s*list|checklist|todo|to\s*do|tasks?)\b",
    re.I,
)
_EXECUTION_ORDER_HEADING_RE = re.compile(
    r"^\s*#{1,3}\s*(?:user\s+)?(?:plan\s+)?(?:execution\s+order|implementation\s+order|order\s+of\s+work|workflow|steps?)\b",
    re.I,
)
_NEXT_SECTION_RE = re.compile(r"^\s*#{1,3}\s+\S")
_ITEM_START_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|[□☐☑☒])\s*(.+?)\s*$")
_FIELD_RE = re.compile(r"^\s*(?:[-*+]\s*)?(File|Files?|VERIFY|RUN|STEP_VERIFY|Details?)\s*:\s*(.+?)\s*$", re.I)
_CHECKBOX_RE = re.compile(r"^[□☐☑☒]\s*")
_PLAN_VERIFY_RE = re.compile(r"(?im)^\s*(?:STEP_VERIFY|VERIFY|RUN)\s*:\s*(.+?)\s*$")


@dataclass
class PlanItem:
    index: int
    title: str
    details: str = ""
    file_refs: list[str] = field(default_factory=list)
    verify_cmd: str = ""
    status: str = "pending"
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionPlan:
    markdown: str
    items: list[PlanItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "markdown": self.markdown,
            "items": [item.to_dict() for item in self.items],
        }

    def all_done(self) -> bool:
        return bool(self.items) and all(item.status == "done" for item in self.items)

    def checklist_text(self) -> str:
        return render_checklist(self.items, compact=True)

    def status_text(self) -> str:
        return render_plan_status(self.items)


def _strip_checkbox(text: str) -> str:
    return _CHECKBOX_RE.sub("", text).strip()


def _extract_verify_cmd(text: str) -> str:
    m = _PLAN_VERIFY_RE.search(text)
    if not m:
        return ""
    return _clean_command_line(m.group(1))


def _make_item(index: int, title: str, details: str = "") -> PlanItem:
    file_refs = _extract_step_file_refs(details or title)
    verify_cmd = _extract_verify_cmd(details or title)
    return PlanItem(
        index=index,
        title=title.strip(),
        details=details.strip(),
        file_refs=file_refs,
        verify_cmd=verify_cmd,
    )


def _collect_section_lines(markdown: str, heading_re: re.Pattern[str]) -> list[str]:
    lines = markdown.splitlines()
    start = None
    for idx, raw in enumerate(lines):
        if heading_re.match(raw):
            start = idx + 1
            break
    if start is None:
        return []

    collected: list[str] = []
    for raw in lines[start:]:
        if _NEXT_SECTION_RE.match(raw) and not _ITEM_START_RE.match(raw):
            break
        collected.append(raw)
    return collected


def _parse_section_items(markdown: str, heading_re: re.Pattern[str]) -> list[PlanItem]:
    section = _collect_section_lines(markdown, heading_re)
    if not section:
        return []

    items: list[PlanItem] = []
    current_title = ""
    current_details: list[str] = []
    current_index = 0

    def _flush() -> None:
        nonlocal current_title, current_details, current_index
        if not current_title:
            return
        current_index += 1
        items.append(_make_item(current_index, current_title, "\n".join(current_details)))
        current_title = ""
        current_details = []

    for raw in section:
        stripped = raw.strip()
        if not stripped:
            if current_title:
                current_details.append("")
            continue
        item_match = _ITEM_START_RE.match(raw)
        if item_match:
            _flush()
            current_title = _strip_checkbox(item_match.group(1)).strip()
            current_details = [item_match.group(1).strip()]
            continue
        if current_title:
            current_details.append(stripped)
            continue

    _flush()
    return items


def _parse_explicit_checklist(markdown: str) -> list[PlanItem]:
    return _parse_section_items(markdown, _CHECKLIST_HEADING_RE)


def _parse_execution_order(markdown: str) -> list[PlanItem]:
    return _parse_section_items(markdown, _EXECUTION_ORDER_HEADING_RE)


def _clean_item_title(title: str) -> str:
    """Strip markdown bold/backticks the planner wraps around item titles."""
    t = title.strip()
    if t.startswith("**") and t.endswith("**") and len(t) > 4:
        t = t[2:-2]
    return t.strip().strip("`").strip()


def _dedupe_items(items: list[PlanItem]) -> list[PlanItem]:
    """Drop checklist items with identical normalized titles (planners often
    repeat e.g. 'Tests ausführen' twice); reindex the survivors."""
    seen: set[str] = set()
    unique: list[PlanItem] = []
    for item in items:
        key = " ".join(_clean_item_title(item.title).lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        item.title = _clean_item_title(item.title)
        item.index = len(unique) + 1
        unique.append(item)
    return unique


def parse_execution_plan(markdown: str) -> ExecutionPlan:
    """Parse a detailed planner markdown response into structured checklist items."""
    markdown = markdown or ""
    items = _parse_explicit_checklist(markdown)
    if not items:
        items = _parse_execution_order(markdown)
    if not items:
        steps = _extract_plan_steps(markdown)
        items = [_make_item(i + 1, step, step) for i, step in enumerate(steps)]
    if not items and markdown.strip():
        items = [_make_item(1, markdown.strip()[:120], markdown.strip())]
    return ExecutionPlan(markdown=markdown, items=_dedupe_items(items))


def render_checklist(items: list[PlanItem], *, compact: bool = True) -> str:
    if not items:
        return ""
    mark_for = {
        "done": "☑",
        "failed": "☒",
        "in_progress": "☐",
        "pending": "□",
    }
    lines: list[str] = []
    for item in items:
        mark = mark_for.get(item.status, "□")
        title = item.title.strip()
        if compact:
            lines.append(f"{mark} {title}")
            continue
        lines.append(f"{mark} {item.index}. {title}")
        if item.file_refs:
            lines.append(f"   File: {', '.join(item.file_refs)}")
        if item.verify_cmd:
            lines.append(f"   VERIFY: {item.verify_cmd}")
        if item.details:
            lines.append(f"   {item.details}")
    return "\n".join(lines)


def active_checklist_line(compact: str) -> str:
    """From a compact checklist string, return the single *active* item — the
    first item that is not yet done.

    The live bottom bar shows only this one line (the item currently being
    worked on) instead of the whole truncated list, and it advances on its own
    as items complete. Done ``☑`` and failed ``☒`` items are skipped; the first
    pending ``□`` or in-progress ``☐`` line is returned. Returns ``""`` when
    every item is done (caller renders an "all done" hint)."""
    for raw in (compact or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[:1] in ("☑", "☒"):   # done / failed → not the active item
            continue
        return line                  # first □ (pending) or ☐ (in-progress)
    return ""


def render_plan_status(items: list[PlanItem]) -> str:
    if not items:
        return "PLAN STATUS: no checklist items parsed."
    done = sum(1 for item in items if item.status == "done")
    total = len(items)
    lines = [f"PLAN STATUS: {done}/{total} checklist item(s) done"]
    lines.append(render_checklist(items, compact=False))
    return "\n".join(lines)


def refresh_plan_status(plan: ExecutionPlan, proof, touched_paths: list[str] | None = None) -> ExecutionPlan:
    touched = {p.replace("\\", "/") for p in (touched_paths or [])}
    changed = {p.replace("\\", "/") for p in getattr(proof, "files_changed", [])}
    commands = list(getattr(proof, "commands", []) or [])

    for item in plan.items:
        file_refs = [p.replace("\\", "/") for p in item.file_refs]
        file_ok = not file_refs or all(ref in touched or ref in changed for ref in file_refs)
        verify_ok = True
        matched_cmd = ""
        if item.verify_cmd:
            verify_ok = False
            wanted = " ".join(item.verify_cmd.split()).lower()
            for cmd in commands:
                cmd_text = " ".join(str(cmd.get("cmd", "")).split()).lower()
                if (cmd_text == wanted or (wanted and wanted in cmd_text)) and int(cmd.get("exit_code", 1)) == 0:
                    verify_ok = True
                    matched_cmd = str(cmd.get("cmd", ""))
                    break
        else:
            for cmd in commands:
                if cmd.get("is_test") and int(cmd.get("exit_code", 1)) == 0:
                    verify_ok = True
                    matched_cmd = str(cmd.get("cmd", ""))
                    break
        # An item may only be auto-completed if there is at least ONE form of
        # evidence to evaluate (a file to touch or a verify command). Without
        # that, file_ok and verify_ok both default to True and a vague item
        # ("understand the codebase") would be marked done with zero evidence.
        has_evidence = bool(file_refs) or bool(item.verify_cmd)
        if has_evidence and file_ok and verify_ok:
            item.status = "done"
            if matched_cmd and matched_cmd not in item.evidence:
                item.evidence.append(matched_cmd)
            for ref in file_refs:
                if ref not in item.evidence:
                    item.evidence.append(ref)
        elif file_refs and any(ref in touched or ref in changed for ref in file_refs):
            if item.status == "pending":
                item.status = "in_progress"
        elif item.status == "done":
            item.status = "done"

        # Extra: if files for this item were touched and we have *any* passing test command
        # in this proof, consider it done (helps checklist items that don't list an explicit
        # VERIFY: or when the coder used a close variant). This reduces "stuck on same file".
        if item.status != "done" and file_refs and any(ref in touched or ref in changed for ref in file_refs):
            any_passing_test = any(
                int(c.get("exit_code", 1)) == 0 and c.get("is_test")
                for c in commands
            )
            if any_passing_test:
                item.status = "done"
    return plan


def plan_all_done(plan: ExecutionPlan) -> bool:
    return plan.all_done()


def plan_to_dict(plan: ExecutionPlan) -> dict:
    return plan.to_dict()
