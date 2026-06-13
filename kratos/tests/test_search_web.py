"""Tests for regex/text search, ranged reads, and web fetch/search behavior."""

from pathlib import Path

from kratos.execution.search import (
    search_text, search_regex, read_file_range, glob_files, list_files,
)
from kratos.web import web_fetch, web_search, build_request, scrape_text_from_html


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "import os\n"
        "def load_source(path):\n"
        "    return path.read_text()\n"
        "\n"
        "class WeatherCard:\n"
        "    pass\n"
    )
    (tmp_path / "pkg" / "b.txt").write_text("nothing here")
    return tmp_path


def test_search_text_returns_line_col_context(tmp_path: Path):
    root = _make_project(tmp_path)
    hits = search_text(root, "load_source", glob="*.py")
    assert hits
    h = hits[0]
    assert h.rel_path.endswith("a.py")
    assert h.line == 2
    assert h.column >= 1
    assert any("def load_source" in c for c in h.context)


def test_search_regex_finds_class(tmp_path: Path):
    root = _make_project(tmp_path)
    hits = search_regex(root, r"class\s+\w+:", glob="*.py")
    assert isinstance(hits, list) and hits
    assert hits[0].line == 5


def test_search_regex_invalid_returns_error(tmp_path: Path):
    res = search_regex(tmp_path, r"(unclosed")
    assert isinstance(res, str) and "invalid regex" in res


def test_read_file_range(tmp_path: Path):
    root = _make_project(tmp_path)
    rr = read_file_range(root, "pkg/a.py", 2, 3)
    assert rr.ok
    assert rr.start_line == 2 and rr.end_line == 3
    assert "def load_source" in rr.content
    assert rr.total_lines == 6


def test_glob_and_list(tmp_path: Path):
    root = _make_project(tmp_path)
    assert "pkg/a.py" in list_files(root)
    assert glob_files(root, "*.py") == ["pkg/a.py"]


def test_scrape_text_from_html_drops_script():
    html = "<html><head><style>x</style></head><body><p>Hello</p><script>bad()</script></body></html>"
    text = scrape_text_from_html(html)
    assert "Hello" in text
    assert "bad()" not in text


def test_build_request_sets_user_agent():
    req = build_request("https://example.com")
    assert req.get_header("User-agent", "").startswith("KratosAgent/")


def test_web_search_unconfigured_provider_is_honest():
    results, err = web_search("anything", provider="google")
    assert results == []
    assert "not configured" in err


def test_web_fetch_uses_injected_opener_with_timeout(tmp_path: Path):
    """web_fetch must set a timeout and UA; verify via an injected fake opener."""
    captured = {}

    class _Resp:
        status = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def read(self, n):
            return b"<p>ok</p>"

        def geturl(self):
            return "https://example.com"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, request, timeout=None):
            captured["timeout"] = timeout
            captured["ua"] = request.get_header("User-agent", "")
            return _Resp()

    res = web_fetch("https://example.com", timeout_seconds=7, _opener=_Opener())
    assert res.ok
    assert captured["timeout"] == 7
    assert captured["ua"].startswith("KratosAgent/")
    assert "ok" in res.text


def test_web_fetch_refuses_non_http_scheme():
    res = web_fetch("file:///etc/passwd")
    assert not res.ok
    assert "scheme" in res.error


def test_web_fetch_refuses_private_host():
    res = web_fetch("http://localhost/secret")
    assert not res.ok
    assert "private" in res.error or "loopback" in res.error


# ── web research evidence (sources proven from research.jsonl) ───────────────

def test_collect_research_sources_only_real_and_recent(tmp_path):
    from kratos.web import record_research_note, collect_research_sources
    kdir = tmp_path / ".kratos"
    # an OLD note (before run start) must be ignored
    record_research_note("web_fetch", {"url": "https://old.example/x", "ok": True,
                                        "ts": "2000-01-01T00:00:00+00:00"}, kdir)
    # overwrite ts to be old explicitly (record_research_note stamps its own ts,
    # so we write the old entry directly to be deterministic)
    import json
    (kdir / "research.jsonl").write_text(
        json.dumps({"ts": "2000-01-01T00:00:00+00:00", "kind": "web_fetch",
                    "url": "https://old.example/x", "ok": True}) + "\n", encoding="utf-8")
    since = "2026-01-01T00:00:00+00:00"
    record_research_note("web_fetch", {"url": "https://docs.python.org/3/", "ok": True,
                                       "final_url": "https://docs.python.org/3/"}, kdir)
    record_research_note("web_fetch", {"url": "https://broken.example", "ok": False,
                                       "error": "HTTP 500"}, kdir)
    record_research_note("web_search", {"query": "html.parser", "ok": True,
                                        "results": [{"url": "https://docs.python.org/3/library/html.parser.html"}]}, kdir)
    sources = collect_research_sources(kdir, since_iso=since)
    assert "https://docs.python.org/3/" in sources           # real fetch
    assert "https://docs.python.org/3/library/html.parser.html" in sources  # real search hit
    assert "https://old.example/x" not in sources            # too old → excluded
    assert "https://broken.example" not in sources           # failed fetch → excluded


def test_collect_research_sources_empty_when_nothing_fetched(tmp_path):
    from kratos.web import collect_research_sources
    assert collect_research_sources(tmp_path / ".kratos") == []


def test_web_fetch_then_scrape_end_to_end(tmp_path):
    """Prove the full scrape path: fetch HTML via injected opener → extract text,
    and that the fetch is recorded as a real source."""
    from kratos.web import web_fetch, scrape_text_from_html, collect_research_sources

    class _Resp:
        status = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}
        def read(self, n):
            return b"<html><body><h1>Weather</h1><p>Feldkirch 21 C</p><script>x()</script></body></html>"
        def geturl(self):
            return "https://example.com/weather"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Opener:
        def open(self, request, timeout=None):
            return _Resp()

    kdir = tmp_path / ".kratos"
    res = web_fetch("https://example.com/weather", _opener=_Opener(), project_dir=kdir)
    assert res.ok
    text = scrape_text_from_html(res.text)
    assert "Feldkirch 21 C" in text
    assert "x()" not in text                       # script stripped
    sources = collect_research_sources(kdir)
    assert "https://example.com/weather" in sources


# ── backtick-wrapped paths (session 2026-06-13_20-55-49: every read_range failed) ─

def test_read_file_range_strips_backticks(tmp_path):
    root = _make_project(tmp_path)
    # model emits the path wrapped in backticks, e.g. `pkg/a.py`
    rr = read_file_range(root, "`pkg/a.py`", 1, 3)
    assert rr.ok, rr.error
    assert "import os" in rr.content


def test_resolve_project_path_strips_backticks(tmp_path):
    from kratos.execution.search import resolve_project_path
    root = _make_project(tmp_path)
    resolved, note = resolve_project_path(root, "`pkg/a.py`")
    assert resolved == "pkg/a.py"


def test_do_read_range_handles_backtick_path(tmp_path):
    from kratos.execution.tools import do_read_range
    root = _make_project(tmp_path)

    class PM:
        def get_marker(self, k): return None
        def get_snippet(self, k): return None

    gen = do_read_range(root, "`pkg/a.py`:1-3")
    obs = None
    try:
        while True:
            next(gen)
    except StopIteration as e:
        obs = e.value
    assert obs is not None and obs.get("ok"), obs   # backtick path no longer "file does not exist"
    assert obs["path"].strip("`") == "pkg/a.py"
    assert "import os" in obs.get("content", "")
