"""Regression tests built from a REAL failed Kratos run.

Source: coding_agent_light_fullcheck, session 2026-06-12_12-03-03.jsonl.
In that run the coder model produced valid work (full ### FILE blocks,
READ_RANGE/SEARCH/VERIFY markers), but the RUNTIME dropped almost all of it:

  - FILE blocks with an annotation line between marker and fence → not parsed
  - ### READ_RANGE swallowed by the ### READ marker (no word boundary)
  - PromptManager.get_marker(key) returns the key itself when unconfigured →
    code fallbacks never used → GREP matched bare 'grep ...' lines in fences
  - '### VERIFY: `python -m pytest`' (backticks) → not recognized as safe
  - pyproject.toml in starter_project/ subdir → no test command discovered
  - model looped on identical READ_RANGE lookups without consequence

Every test input below is verbatim (or minimally trimmed) from that log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kratos.execution.parsing import _parse_file_changes  # noqa: E402
from kratos.execution.tools import (  # noqa: E402
    _normalize_grep_arg, _split_pattern_glob, do_inspect, parse_actions,
)
from kratos.verification import (  # noqa: E402
    CommandRegistry,
    _clean_command_line,
    _detect_project_toolchains,
    _infer_project_verification_commands,
    _is_safe_verification_command,
)


class _KeyEchoPM:
    """Mimics the real PromptManager: get_marker returns the KEY ITSELF when
    the marker is not configured — the exact behavior that broke fallbacks."""

    def get_marker(self, key):
        return key

    def get_snippet(self, key):
        return None


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def main():\n    return 0\n", encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def nested_project(tmp_path: Path) -> Path:
    sub = tmp_path / "starter_project"
    (sub / "tests").mkdir(parents=True)
    (sub / "pyproject.toml").write_text("[project]\nname='x'\n", "utf-8")
    (sub / "tests" / "test_x.py").write_text("def test_ok():\n    assert True\n", "utf-8")
    return tmp_path


def _drain(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_file_block_with_annotation_line_is_parsed():
    """Real failure: 8 complete implementations were dropped because of the
    italic comment line between '### FILE:' and the code fence."""
    text = (
        "### FILE: starter_project/mini_agent_check/scraper.py  \n"
        "*(full file content with fixed `parse_weather_html` implementation)*\n"
        "\n"
        "```python\n"
        "def parse_weather_html(html, city):\n"
        "    return {}\n"
        "```\n"
    )
    changes = _parse_file_changes(text)
    assert changes, "FILE block with annotation line between marker and fence must parse"
    path, content = changes[0]
    assert path == "starter_project/mini_agent_check/scraper.py"
    assert "def parse_weather_html" in content


def test_file_block_plain_still_parses():
    text = "### FILE: a/b.py\n```python\nx = 1\n```\n"
    assert _parse_file_changes(text) == [("a/b.py", "x = 1\n")]


def test_file_block_does_not_steal_next_fence():
    """The junk-line tolerance must not pair a FILE marker with a LATER fence."""
    text = (
        "### FILE: a/b.py\n"
        "no fence here, just prose that goes on\n"
        "more prose\n"
        "even more prose\n"
        "yet more prose lines\n"
        "and the fence below belongs to something else\n"
        "```python\nx = 1\n```\n"
    )
    assert _parse_file_changes(text) == []


def test_read_range_not_swallowed_by_read_marker():
    """Real failure: read_file('_RANGE: docs/research_notes.md:1-30')."""
    text = "### READ_RANGE: docs/research_notes.md:1-30  \n*(show the first 30 lines)*"
    actions = parse_actions(text, _KeyEchoPM())
    assert actions["ranges"] == ["docs/research_notes.md:1-30"]
    assert actions["reads"] == []


def test_marker_fallback_when_get_marker_echoes_key():
    """Real failure: get_marker('grep') returned 'grep', so every line starting
    with the word 'grep' inside a code fence was executed as a GREP action."""
    text = (
        "### INSPECT: **/scraper.py**\n"
        "```powershell\n"
        'grep -n "def parse_weather_html" starter_project/mini_agent_check/scraper.py\n'
        "```\n"
    )
    actions = parse_actions(text, _KeyEchoPM())
    assert actions["greps"] == [], "fence content must not be parsed as GREP actions"
    assert actions["inspects"], "the INSPECT marker itself must be parsed"


def test_grep_rg_style_argument_normalized():
    pat, glob = _normalize_grep_arg('-n "def parse_weather_html" starter_project/mini_agent_check/scraper.py')
    assert pat == "def parse_weather_html"
    assert glob == "starter_project/mini_agent_check/scraper.py"
    pat2, glob2 = _normalize_grep_arg('rg -n "def main"')
    assert pat2 == "def main" and glob2 is None


def test_bracketed_glob_form_from_real_log():
    """Real output: '### SEARCH: **parse_weather_html** [:: starter_project/...py]'."""
    pat, glob = _split_pattern_glob("**parse_weather_html** [:: starter_project/mini_agent_check/scraper.py]")
    assert pat == "parse_weather_html"
    assert glob == "starter_project/mini_agent_check/scraper.py"


def test_inspect_path_arg_redirects_to_read(project: Path):
    """Real output: '### INSPECT: **/scraper.py**' was rejected; now it reads."""
    obs = _drain(do_inspect(None, project, "**/app.py**"))
    assert obs["kind"] == "read" and obs["ok"]
    assert "def main" in obs["content"]


def test_clean_command_line_strips_backticks():
    """Real output: '### VERIFY: `python -m pytest`' was not recognized."""
    cleaned = _clean_command_line("`python -m pytest`")
    assert cleaned == "python -m pytest"
    assert _is_safe_verification_command(cleaned)


def test_nested_project_discovery_with_cwd(nested_project: Path):
    """Real failure: 'PROVEN_WORK missing: no safe build/test command was
    configured, found in README, or inferred from project files.'"""
    assert "python" in _detect_project_toolchains(nested_project)
    cmds = _infer_project_verification_commands(nested_project)
    pytest_cmds = [c for c in cmds if "pytest" in c.cmd]
    assert pytest_cmds, "nested python project must yield a pytest command"
    assert pytest_cmds[0].cwd == "starter_project"
    assert pytest_cmds[0].is_test


def test_pick_auto_verify_cmd_prefers_item_then_registry(nested_project: Path):
    from kratos.config import KratosConfig
    from kratos.planning import PlanItem
    from kratos.roles.coder import _pick_auto_verify_cmd

    registry = CommandRegistry(KratosConfig(), nested_project).discover()

    # item without verify_cmd -> registry test command (with nested cwd)
    item = PlanItem(index=1, title="fix parser")
    cmd = _pick_auto_verify_cmd(item, registry, nested_project)
    assert cmd is not None and cmd.is_test
    assert cmd.cwd == "starter_project"

    # item with a safe verify_cmd -> that command is used
    item2 = PlanItem(index=2, title="fix", verify_cmd="python -m pytest")
    cmd2 = _pick_auto_verify_cmd(item2, registry, nested_project)
    assert cmd2 is not None and "pytest" in cmd2.cmd


def test_lookup_signature_detects_repeats():
    """Real failure: 6 identical READ_RANGE turns in a row, never editing."""
    from kratos.roles.coder import _lookup_signature
    pm = _KeyEchoPM()
    a1 = parse_actions("### READ_RANGE: starter_project/mini_agent_check/cli.py:1-80", pm)
    a2 = parse_actions("### READ_RANGE: starter_project/mini_agent_check/cli.py:1-80", pm)
    a3 = parse_actions("### READ_RANGE: starter_project/mini_agent_check/cli.py:1-150", pm)
    assert _lookup_signature(a1) == _lookup_signature(a2)
    assert _lookup_signature(a1) != _lookup_signature(a3)


def test_search_with_bare_glob_arg_lists_files(project: Path):
    """Real output: '### SEARCH: **test_cli.py**' — a filename, not a text query."""
    from kratos.execution.tools import do_search
    obs = _drain(do_search(project, "**/app.py**"))
    assert obs["kind"] == "glob"
    assert obs["files"] == ["src/app.py"]


# ── session 2026-06-12_18-02 (second failed run): convergence fixes ──────────

def test_search_prose_query_gets_usage_hint(project: Path):
    """Real output: SEARCH 'argparse usage in starter_project/...cli.py' -> 0 hits.
    With smart search the prose query now explains itself (keyword fallback ran,
    usage hint included) instead of a bare 0."""
    from kratos.execution.tools import do_search
    obs = _drain(do_search(project, "argparse usage in starter_project cli"))
    assert obs["count"] == 0
    detail = obs.get("detail", "").lower()
    assert "keyword" in detail or "literal" in detail


def test_plan_items_deduped_and_bold_stripped():
    """Real plan: '\u25a1 **Tests ausf\u00fchren**' appeared twice with bold markers."""
    from kratos.planning import parse_execution_plan
    md = (
        "## CHECKLIST\n"
        "- **Parser reparieren**\n"
        "- **CLI reparieren und erweitern**\n"
        "- **Tests ausf\u00fchren**\n"
        "- **Tests ausf\u00fchren**\n"
    )
    plan = parse_execution_plan(md)
    titles = [i.title for i in plan.items]
    assert titles == ["Parser reparieren", "CLI reparieren und erweitern", "Tests ausf\u00fchren"]
    assert [i.index for i in plan.items] == [1, 2, 3]


def test_failure_readback_injects_broken_file(project: Path):
    """Real failure: after `python -m pytest` exit=2 the model never re-read
    its own broken file. The runtime must inject it into the observation."""
    from kratos.roles.coder import _inject_failure_readback
    (project / "src" / "broken.py").write_text("def kaputt(:\n", "utf-8")
    observations = [
        {"kind": "command", "cmd": "python -m pytest", "ok": False,
         "skipped": False, "exit_code": 2, "output": "SyntaxError: invalid syntax"},
    ]
    _inject_failure_readback(project, ["src/broken.py"], observations)
    notes = [o for o in observations if o.get("kind") == "note"]
    assert notes, "failed verify must inject the touched file's current content"
    assert "def kaputt(:" in notes[0]["detail"]
    assert "src/broken.py" in notes[0]["detail"]


def test_failure_readback_skips_on_success(project: Path):
    from kratos.roles.coder import _inject_failure_readback
    observations = [
        {"kind": "command", "cmd": "python -m pytest", "ok": True,
         "skipped": False, "exit_code": 0, "output": "7 passed"},
    ]
    _inject_failure_readback(project, ["src/app.py"], observations)
    assert not [o for o in observations if o.get("kind") == "note"]


# ── session 2026-06-12_20-27 (third failed run): the cwd root cause ──────────
# Auto-verify used the checklist's `VERIFY: python -m pytest` WITHOUT the
# discovered cwd -> ran in the parent root -> 124x "ModuleNotFoundError:
# No module named 'mini_agent_check'" -> the model could never converge.

def test_registry_cwd_for_matches_toolchain(nested_project: Path):
    from kratos.config import KratosConfig
    registry = CommandRegistry(KratosConfig(), nested_project).discover()
    assert registry.cwd_for("python -m pytest") == "starter_project"
    assert registry.cwd_for("pytest -q") == "starter_project"
    # other toolchains must not inherit the python cwd
    assert registry.cwd_for("npm test") is None


def test_item_verify_cmd_inherits_registry_cwd(nested_project: Path):
    """THE bug of run 3: item verify_cmd won over the registry command and
    dropped the nested cwd."""
    from kratos.config import KratosConfig
    from kratos.planning import PlanItem
    from kratos.roles.coder import _pick_auto_verify_cmd

    registry = CommandRegistry(KratosConfig(), nested_project).discover()
    item = PlanItem(index=1, title="fix parser", verify_cmd="python -m pytest")
    cmd = _pick_auto_verify_cmd(item, registry, nested_project)
    assert cmd is not None
    assert cmd.cmd == "python -m pytest"
    assert cmd.cwd == "starter_project", "item verify_cmd must inherit the discovered cwd"


def test_coder_emitted_verify_inherits_registry_cwd(nested_project: Path):
    """### VERIFY: python -m pytest from the model must also run in the
    nested project directory."""
    from kratos.config import KratosConfig
    from kratos.verification import ProvenWork

    registry = CommandRegistry(KratosConfig(), nested_project).discover()
    captured = {}

    class _Agent:
        def _run_verification_command(self, vcmd):
            captured["cwd"] = vcmd.cwd
            captured["cmd"] = vcmd.cmd
            return {"cmd": vcmd.cmd, "purpose": vcmd.purpose, "source": vcmd.source,
                    "is_test": vcmd.is_test, "exit_code": 0,
                    "duration_seconds": 0.01, "output": "1 passed"}

    from kratos.execution.tools import do_command
    proof = ProvenWork(iteration=1)
    obs = _drain(do_command(_Agent(), nested_project, registry, "python -m pytest", proof))
    assert not obs.get("skipped"), obs
    assert captured["cwd"] == "starter_project"
