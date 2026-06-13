"""Tests for plan parsing + status (incl. the zero-evidence 'done' fix)."""

from kratos.planning import parse_execution_plan, refresh_plan_status, PlanItem, ExecutionPlan
from kratos.verification import ProvenWork


def test_parse_checklist():
    md = (
        "## Plan Checklist\n"
        "- Fix the parser in `pkg/parser.py`\n"
        "- Add tests\n"
    )
    plan = parse_execution_plan(md)
    assert len(plan.items) == 2
    assert plan.items[0].index == 1


def test_zero_evidence_item_not_marked_done():
    """An item with no file refs and no verify command must NOT auto-complete."""
    plan = ExecutionPlan(markdown="", items=[PlanItem(index=1, title="Understand the codebase")])
    proof = ProvenWork(iteration=1)
    proof.commands = [{"cmd": "python -m pytest", "is_test": True, "exit_code": 0}]
    refresh_plan_status(plan, proof, touched_paths=[])
    assert plan.items[0].status != "done"   # no evidence → stays pending


def test_item_with_passing_verify_marked_done():
    plan = ExecutionPlan(markdown="", items=[
        PlanItem(index=1, title="run tests", verify_cmd="python -m pytest")
    ])
    proof = ProvenWork(iteration=1)
    proof.commands = [{"cmd": "python -m pytest", "is_test": True, "exit_code": 0}]
    refresh_plan_status(plan, proof, touched_paths=[])
    assert plan.items[0].status == "done"


def test_item_with_touched_file_and_passing_test_done():
    plan = ExecutionPlan(markdown="", items=[
        PlanItem(index=1, title="fix parser", file_refs=["pkg/parser.py"])
    ])
    proof = ProvenWork(iteration=1)
    proof.commands = [{"cmd": "python -m pytest", "is_test": True, "exit_code": 0}]
    refresh_plan_status(plan, proof, touched_paths=["pkg/parser.py"])
    assert plan.items[0].status == "done"


def test_failing_verify_keeps_item_not_done():
    plan = ExecutionPlan(markdown="", items=[
        PlanItem(index=1, title="run tests", verify_cmd="python -m pytest")
    ])
    proof = ProvenWork(iteration=1)
    proof.commands = [{"cmd": "python -m pytest", "is_test": True, "exit_code": 1}]
    refresh_plan_status(plan, proof, touched_paths=[])
    assert plan.items[0].status != "done"
