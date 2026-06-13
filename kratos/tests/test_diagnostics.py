"""Tests for the failure diagnoser + repair-stall tracker.

These encode the exact failure mode from session 2026-06-13_00-57-15: a
circular import that pytest reports as an exit-code-2 collection error, which
the old loop re-ran 94 times without ever diagnosing.
"""

from kratos.execution.diagnostics import (
    FailureDiagnoser, RepairTracker, diagnose_command,
)

CIRCULAR_OUTPUT = (
    "ImportError while importing test module 'tests/test_scraper.py'.\n"
    "mini_agent_check/scraper.py:13: in <module>\n"
    "    from mini_agent_check.cli import WeatherCard\n"
    "mini_agent_check/cli.py:15: in <module>\n"
    "    from .scraper import load_source, parse_weather_html\n"
    "E   ImportError: cannot import name 'load_source' from partially initialized "
    "module 'mini_agent_check.scraper' (most likely due to a circular import)\n"
)


def test_circular_import_is_diagnosed():
    d = FailureDiagnoser().diagnose(CIRCULAR_OUTPUT, cmd="python -m pytest", exit_code=2)
    assert d is not None
    assert d.category == "circular_import"
    assert "REORDER" in d.fix_instruction.upper() or "reorder" in d.fix_instruction
    assert "broken" in d.fix_instruction.lower() or "lazy import" in d.fix_instruction.lower()
    assert any("scraper.py" in f or "cli.py" in f for f in d.files)


def test_circular_import_signature_is_stable():
    a = FailureDiagnoser().diagnose(CIRCULAR_OUTPUT, exit_code=2)
    b = FailureDiagnoser().diagnose(CIRCULAR_OUTPUT, exit_code=2)
    assert a.signature == b.signature


def test_module_not_found():
    out = "E   ModuleNotFoundError: No module named 'mini_agent_check'"
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "module_not_found"
    assert "working directory" in d.fix_instruction.lower()


def test_syntax_error():
    out = 'File "cli.py", line 9\n    def main(\n           ^\nSyntaxError: invalid syntax'
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "syntax_error"


def test_not_implemented_stub():
    out = "E   NotImplementedError"
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "not_implemented"
    assert "implement" in d.fix_instruction.lower()


def test_assertion_failure():
    out = "E   assert {'city': 'X'} == {'city': 'Feldkirch'}"
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "assertion_failure"


def test_success_returns_none():
    assert diagnose_command({"exit_code": 0, "output": "5 passed"}) is None


def test_blocked_command():
    d = diagnose_command({"exit_code": 126, "blocked": True, "block_reason": "rm -rf"})
    assert d.category == "blocked_command"


def test_diagnose_command_from_result_dict():
    res = {"cmd": "python -m pytest", "exit_code": 2, "output": CIRCULAR_OUTPUT}
    d = diagnose_command(res)
    assert d.category == "circular_import"


# ── stall tracker ───────────────────────────────────────────────────────────

def test_tracker_detects_stall_after_threshold():
    t = RepairTracker(stall_threshold=2)
    sig = "circular_import:mini_agent_check.scraper:load_source"
    assert t.register(sig) == 1
    assert not t.is_stalled(sig)
    assert t.register(sig) == 2
    assert t.is_stalled(sig)
    assert "STALL" in t.escalation_note(sig)


def test_tracker_distinguishes_signatures():
    t = RepairTracker(stall_threshold=2)
    t.register("a")
    t.register("b")
    assert not t.is_stalled("a")
    assert not t.is_stalled("b")


def test_tracker_repeated_immediately():
    t = RepairTracker()
    t.register("a")
    t.register("a")
    assert t.repeated_immediately("a")
    t.register("b")
    assert not t.repeated_immediately("b")


def test_tracker_reset():
    t = RepairTracker()
    t.register("a")
    t.reset()
    assert t.count("a") == 0


# ── signature-mismatch ranking (session 2026-06-13_13-12-22) ─────────────────

SIGNATURE_OUTPUT = (
    "    def test_cli_json_stdout() -> None:\n"
    "E       AssertionError: Traceback (most recent call last):\n"
    "C:\\\\Program Files\\\\Python312\\\\Lib\\\\html\\\\parser.py:338: in goahead\n"
    "    self.handle_starttag(tag, attrs)\n"
    "C:\\\\Users\\\\manue\\\\Downloads\\\\coding_agent_light_fullcheck\\\\starter_project\\\\mini_agent_check\\\\scraper.py:20: in handle_starttag\n"
    "E   TypeError: WeatherHTMLParser.handle_starttag() missing 1 required positional argument: 'attrs_dict'\n"
)


def test_signature_mismatch_beats_downstream_assertion():
    """The TypeError is the root cause; the assertion is a downstream symptom.
    The diagnoser must report the signature mismatch, not assertion_failure."""
    d = FailureDiagnoser().diagnose(SIGNATURE_OUTPUT, cmd="python -m pytest", exit_code=1)
    assert d.category == "signature_mismatch"
    assert "handle_starttag" in d.summary
    assert "override" in d.fix_instruction.lower()


def test_diagnosis_files_drop_stdlib_and_prefer_source():
    d = FailureDiagnoser().diagnose(SIGNATURE_OUTPUT, exit_code=1)
    # stdlib parser.py must not appear; source scraper.py must, shortened
    assert not any("parser.py" in f for f in d.files)
    assert any(f.endswith("mini_agent_check/scraper.py") for f in d.files)
    # absolute path was shortened to a project-relative tail
    assert all(not f.startswith("C:") and not f.startswith("/") for f in d.files)


def test_signature_takes_positional():
    out = "E   TypeError: parse_weather_html() takes 1 positional argument but 2 were given"
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "signature_mismatch"


def test_unexpected_keyword_argument():
    out = "E   TypeError: load_source() got an unexpected keyword argument 'timeout'"
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "signature_mismatch"


# ── parse-returned-None (session 2026-06-13_19-59-46: guessed HTML format) ────

def test_parse_returned_none_steers_to_read_input():
    out = (
        "mini_agent_check/scraper.py:38: in parse_weather_html\n"
        "    temperature_c = int(re.search(r'<span class=\"temperature\">(\\d+)', data).group(1))\n"
        "E   AttributeError: 'NoneType' object has no attribute 'group'\n"
    )
    d = FailureDiagnoser().diagnose(out, cmd="python -m pytest", exit_code=1)
    assert d.category == "parse_returned_none"
    assert "READ" in d.fix_instruction
    assert "input" in d.fix_instruction.lower() or "fixture" in d.fix_instruction.lower()


def test_parse_returned_none_subscript():
    out = "E   TypeError: 'NoneType' object is not subscriptable"
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "parse_returned_none"


def test_real_attribute_error_still_detected():
    out = "E   AttributeError: 'str' object has no attribute 'read_text'"
    d = FailureDiagnoser().diagnose(out, exit_code=1)
    assert d.category == "attribute_error"
