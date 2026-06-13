"""Regression tests for the 'godmode' fixes (session 2026-06-13_00-11-02).

Real failures reproduced here:
  - SEARCH always returned 0 hits because the model searched for prose
    descriptions ('Find the exact line range that contains the weather card
    for "Feldkirch"') instead of patterns → smart_search keyword fallback
  - SEARCH had no `a|b` alternation and no regex support → unified pipeline
  - READ_RANGE: fixtures/weather_sample.html failed with 'file does not
    exist' because the file lives in starter_project/fixtures/ → tolerant
    suffix-match path resolution (also for READ and FILE writes)
  - INSPECT with prose ('Get the full text of the fixture file') burned the
    turn → redirected to smart search
  - the TUI plan box stayed frozen at 'PLAN 0/N' → the work-step driver now
    emits change-aware plan_status events
  - identical lookups alternating A,B,A,B were not flagged as a loop
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kratos.execution.search import (  # noqa: E402
    extract_keywords, read_file_range, resolve_project_path, smart_search,
)


@pytest.fixture()
def nested(tmp_path: Path) -> Path:
    sub = tmp_path / "starter_project"
    (sub / "fixtures").mkdir(parents=True)
    (sub / "mini_agent_check").mkdir()
    (sub / "fixtures" / "weather_sample.html").write_text(
        '<div class="weather-card" data-city="Feldkirch">\n'
        "  <span class=\"temp\">21</span>\n"
        "</div>\n"
        '<div class="weather-card" data-city="Bregenz">\n'
        "  <span class=\"temp\">19</span>\n"
        "</div>\n",
        encoding="utf-8",
    )
    (sub / "mini_agent_check" / "cli.py").write_text(
        "import argparse\n\n\ndef main() -> int:\n    return 0\n", encoding="utf-8",
    )
    return tmp_path


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


# ── smart search ──────────────────────────────────────────────────────────────

def test_smart_search_literal_still_first(nested: Path):
    matches, strategy = smart_search(nested, "weather-card")
    assert strategy == "literal" and matches


def test_smart_search_alternation(nested: Path):
    """`a|b` must work — each side literal, OR-combined."""
    matches, strategy = smart_search(nested, "Feldkirch|Bregenz")
    assert matches, "alternation search must hit"
    cities = {m.text for m in matches}
    assert any("Feldkirch" in t for t in cities)
    assert any("Bregenz" in t for t in cities)


def test_smart_search_regex(nested: Path):
    matches, strategy = smart_search(nested, r'data-city="\w+"')
    assert matches and strategy == "regex"


def test_smart_search_prose_fallback_real_query(nested: Path):
    """The EXACT prose query from the failed run must now return hits."""
    q = 'Find the exact line range that contains the weather card for "Feldkirch"'
    matches, strategy = smart_search(nested, q)
    assert matches, f"prose query must hit via keyword fallback (strategy={strategy})"
    assert "keyword fallback" in strategy
    assert any("Feldkirch" in m.text for m in matches)


def test_extract_keywords_prefers_quoted_and_identifiers():
    kws = extract_keywords(
        'Search for the `parse_weather_html` pattern for "Feldkirch" in Python files'
    )
    assert "parse_weather_html" in kws
    assert "Feldkirch" in kws
    assert "the" not in [k.lower() for k in kws]


def test_do_search_prose_returns_hits_now(nested: Path):
    from kratos.execution.tools import do_search
    obs = _drain(do_search(nested, "Find the weather card for “Feldkirch”"))
    assert obs["count"] > 0, obs.get("detail")


def test_do_grep_falls_back_to_smart_search(nested: Path):
    from kratos.execution.tools import do_grep
    obs = _drain(do_grep(nested, "Search for the argparse pattern in Python files"))
    assert obs["count"] > 0, obs.get("detail")
    assert any("argparse" in m["text"] for m in obs["matches"])


# ── tolerant path resolution ──────────────────────────────────────────────────

def test_resolve_project_path_suffix_match(nested: Path):
    resolved, note = resolve_project_path(nested, "fixtures/weather_sample.html")
    assert resolved == "starter_project/fixtures/weather_sample.html"
    assert "suffix match" in note


def test_read_file_range_resolves_nested_path(nested: Path):
    """The EXACT failure from the log: READ_RANGE fixtures/weather_sample.html:1-200."""
    rr = read_file_range(nested, "fixtures/weather_sample.html", 1, 200)
    assert rr.ok, rr.error
    assert "Feldkirch" in rr.content
    assert rr.rel_path == "starter_project/fixtures/weather_sample.html"


def test_do_write_redirects_to_existing_nested_file(nested: Path):
    from kratos.execution.tools import do_write
    from kratos.verification import ProvenWork
    proof = ProvenWork(iteration=1)
    obs = _drain(do_write(nested, "mini_agent_check/cli.py", "# new\n", proof, 0, {}))
    assert obs["ok"]
    # the write must land on the EXISTING nested file, not create a stray copy
    assert (nested / "starter_project" / "mini_agent_check" / "cli.py").read_text("utf-8") == "# new\n"
    assert not (nested / "mini_agent_check").exists()
    assert proof.files_changed == ["starter_project/mini_agent_check/cli.py"]


def test_do_write_new_file_keeps_given_path(nested: Path):
    from kratos.execution.tools import do_write
    from kratos.verification import ProvenWork
    proof = ProvenWork(iteration=1)
    obs = _drain(do_write(nested, "docs/research_notes.md", "# notes\n", proof, 0, {}))
    assert obs["ok"]
    assert (nested / "docs" / "research_notes.md").exists()


# ── INSPECT prose redirect ────────────────────────────────────────────────────

def test_inspect_prose_redirects_to_smart_search(nested: Path):
    """Real output: '### INSPECT: Get the full text of the fixture file'."""
    from kratos.execution.tools import do_inspect
    obs = _drain(do_inspect(None, nested, "Search for the weather card markup for Feldkirch"))
    assert obs["kind"] in ("search", "glob")
    assert obs.get("count", 0) > 0
    assert "smart search" in str(obs.get("detail", "")).lower() or obs["count"] > 0


# ── live plan status from the work-step driver ───────────────────────────────

def test_workstep_driver_emits_plan_status(tmp_path: Path):
    import types
    from unittest.mock import MagicMock
    from kratos.config import KratosConfig
    from kratos.verification import CommandRegistry, ProvenWork
    from kratos.roles.coder import execute_structured_work_steps_for_plan
    from kratos.planning import parse_execution_plan
    from kratos.context import ContextPackage

    root = tmp_path
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", "utf-8")
    (root / "tests" / "test_x.py").write_text("def test_ok():\n    assert True\n", "utf-8")

    outputs = [
        "### FILE: src/fix.py\n```python\nVALUE = 1\n```\n### VERIFY: python -m pytest tests\n",
    ]
    results = [{"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed",
                "stdout": "1 passed", "stderr": ""}]

    class FakeAgent:
        def __init__(self):
            self.config = KratosConfig(always_max_ctx=False, permission="mid", max_work_step_turns=2)
            self.pending_file_changes = []
            self.pending_file_deletions = []
            self._memory = MagicMock()
            self._knowledge = None
            self._indexer = types.SimpleNamespace(root=root)

        def _is_cancelled(self):
            return False

        def _run_coder(self, msg):
            out = outputs.pop(0) if outputs else "### DONE"
            yield ("coder", out, "text")
            return out

        def _run_verification_command(self, vcmd):
            res = dict(results.pop(0)) if results else {"exit_code": 0, "duration_seconds": 0.0,
                                                        "output": "ok", "stdout": "ok", "stderr": ""}
            res.update({"cmd": vcmd.cmd, "purpose": vcmd.purpose,
                        "source": vcmd.source, "is_test": vcmd.is_test})
            return res

    plan = parse_execution_plan(
        "## CHECKLIST\n- Fix the value\n  File: src/fix.py\n  VERIFY: python -m pytest tests\n"
    )
    registry = CommandRegistry(KratosConfig(), root).discover()
    proof = ProvenWork(iteration=1)
    ctx = ContextPackage(user_input="t", intent="coding", route="planner_then_coder")

    gen = execute_structured_work_steps_for_plan(
        FakeAgent(), "fix", plan, ctx, registry, proof, 0, "", root, {},
    )
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration:
        pass

    plan_events = [e for e in events if e[0] == "plan_status"]
    assert plan_events, "work-step driver must emit live plan_status events"
    assert any("PLAN 1/1" in e[1] for e in plan_events), plan_events
