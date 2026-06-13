"""Regression: importing the app package must never abort the interpreter.

Before the fix, ``kratos/app/__init__.py`` eagerly imported ``.cli``, which
calls ``sys.exit(...)`` at import time when ``rich``/``prompt_toolkit``/
``textual`` are missing. That killed *any* process that imported the package
(including the test runner). The package now resolves those shells lazily, so a
plain ``import kratos.app`` succeeds regardless of optional UI deps.
"""

import subprocess
import sys

import pytest


def test_importing_kratos_app_does_not_sys_exit():
    # A subprocess makes a stray sys.exit observable as a non-zero return code.
    proc = subprocess.run(
        [sys.executable, "-c", "import kratos.app; print('ok')"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"import aborted: {proc.stderr}"
    assert "ok" in proc.stdout


def test_app_unknown_attribute_raises_attribute_error():
    import kratos.app
    with pytest.raises(AttributeError):
        kratos.app.totally_unknown_symbol  # noqa: B018


def test_app_declares_lazy_exports():
    import kratos.app
    for name in ("main", "run_tui", "slash_completions"):
        assert name in kratos.app.__all__
