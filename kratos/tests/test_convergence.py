"""Convergence aids: targeted diagnoses for the recurring weak-model failures
seen in real csvstats runs, plus the unbounded-iteration config defaults."""

from kratos.execution.diagnostics import diagnose_command
from kratos.config import KratosConfig


def _r(output, exit_code=1):
    return {"cmd": "python -m pytest", "exit_code": exit_code, "output": output}


def test_numeric_on_string_diagnosis():
    out = ("csvstats/stats.py:14: TypeError\n"
           "E   TypeError: unsupported operand type(s) for +: 'int' and 'str'\n")
    d = diagnose_command(_r(out))
    assert d is not None and d.category == "numeric_on_string"
    assert "float(" in d.fix_instruction  # tells the model to convert


def test_numeric_comparison_on_string_diagnosis():
    out = "E   TypeError: '<' not supported between instances of 'str' and 'int'\nm.py:3: x\n"
    d = diagnose_command(_r(out))
    assert d is not None and d.category == "numeric_on_string"


def test_missing_raise_diagnosis():
    out = ("tests/test_transform.py:32: Failed\n"
           "E   Failed: DID NOT RAISE <class 'ValueError'>\n"
           "csvstats/transform.py:5: x\n")
    d = diagnose_command(_r(out))
    assert d is not None and d.category == "missing_raise"
    assert "raise" in d.fix_instruction.lower()


def test_invalid_int_literal_is_numeric_diagnosis():
    out = "E   ValueError: invalid literal for int() with base 10: 'abc'\nm.py:2: x\n"
    d = diagnose_command(_r(out))
    assert d is not None and d.category == "numeric_on_string"


def test_config_defaults_unbounded():
    c = KratosConfig()
    assert c.max_verify_iterations == 0          # 0 == unbounded
    assert c.no_progress_abort == 40             # generous safety net
    assert c.max_coder_iterations == 0           # already unbounded
