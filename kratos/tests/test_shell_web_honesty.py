"""Focused guarantees for shell execution + web honesty.

  - ShellRunner enforces a real timeout (exit 124, timed_out=True) and reports
    a non-zero exit for a failing command, with separate stdout/stderr.
  - web_search returns an HONEST error (never fabricated results) when the
    provider is unknown or the underlying fetch fails.
"""

from pathlib import Path

from kratos.execution.shell import ShellRunner
from kratos.web import web_search, FetchResult, SearchResult


def test_shell_timeout_reports_124(tmp_path: Path):
    runner = ShellRunner(tmp_path, default_timeout=1)
    # `sleep`/`Start-Sleep` runs longer than the 1s timeout on any interpreter.
    res = runner.run("sleep 5", shell_type="auto", timeout_seconds=1)
    if res["exit_code"] == 127:
        # No interpreter at all in this environment — nothing to assert.
        return
    assert res["timed_out"] is True
    assert res["exit_code"] == 124
    assert res["duration_seconds"] >= 0


def test_shell_nonzero_exit_and_streams(tmp_path: Path):
    runner = ShellRunner(tmp_path, default_timeout=20)
    res = runner.run("python3 -c \"import sys; sys.stderr.write('boom'); sys.exit(3)\"",
                     shell_type="auto")
    if res["exit_code"] == 127:
        return
    assert res["blocked"] is False
    assert res["exit_code"] == 3
    assert "boom" in res["stderr"]
    assert res["stdout"] == "" or "boom" not in res["stdout"]


def test_web_search_unknown_provider_is_honest():
    results, error = web_search("anything", provider="definitely-not-a-provider")
    assert results == []
    assert "not configured" in error.lower()


def test_web_search_failed_fetch_returns_error_not_results():
    def _broken_fetch(url, timeout_seconds=20, project_dir=None, **_kw):
        return FetchResult(url=url, ok=False, error="network error: unreachable")

    results, error = web_search("python dataclass", _fetch=_broken_fetch)
    assert results == []
    assert error and "unavailable" in error.lower()


def test_web_search_parses_when_fetch_succeeds():
    sample = (
        '<a class="result__a" href="/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F">'
        'Python Docs</a>'
        '<a class="result__snippet">official documentation</a>'
    )

    def _ok_fetch(url, timeout_seconds=20, project_dir=None, **_kw):
        return FetchResult(url=url, ok=True, status=200,
                           content_type="text/html", text=sample)

    results, error = web_search("python", _fetch=_ok_fetch)
    assert error == ""
    assert results and isinstance(results[0], SearchResult)
    assert results[0].url == "https://docs.python.org/3/"
