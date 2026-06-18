"""Web access for Kratos — HTTP fetching, HTML text extraction, web search.

Pure stdlib (urllib + html.parser); no new dependencies. All network access is
purpose-bound (documentation, API references, error research) and every result
is recorded to ``.kratos/research.jsonl`` so the Reporter can document sources.

  - :func:`web_fetch`             — HTTP(S) GET: timeout, User-Agent, status
                                    check, content-type check, size cap,
                                    bounded redirects.
  - :func:`scrape_text_from_html` — html.parser-based text extraction
                                    (drops script/style/nav noise).
  - :func:`web_search`            — provider-adapter search. Default provider
                                    is DuckDuckGo's HTML endpoint (no API key).
                                    When unreachable/unconfigured it returns a
                                    clean, honest error — never fabricated
                                    results.
  - :func:`collect_research_sources` — the REAL sources fetched this run, read
                                    back from research.jsonl for the Reporter.

Security: only http/https schemes; no shell involvement; responses are data,
never executed; private/link-local literal hosts are refused.
"""

from __future__ import annotations

import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

__all__ = ["FetchResult", "SearchResult", "web_fetch", "scrape_text_from_html", "web_search", "record_research_note", "collect_research_sources"]

_USER_AGENT = "KratosAgent/1.0 (+local coding agent; research fetch)"
# DuckDuckGo's HTML endpoint blocks GET requests with non-browser UAs and returns a 202
# interstitial instead of results.  POST with a desktop browser UA works reliably.
_DDG_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_MAX_BYTES = 2_000_000          # 2 MB response cap
_MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = {"http", "https"}
_TEXTUAL_CONTENT_RE = re.compile(r"(text/|json|xml|javascript|x-www-form)", re.I)


@dataclass
class FetchResult:
    url: str
    ok: bool
    status: int = 0
    content_type: str = ""
    text: str = ""
    error: str = ""
    fetched_at: str = ""
    final_url: str = ""

    def to_dict(self) -> dict:
        d = {"url": self.url, "ok": self.ok, "status": self.status,
             "content_type": self.content_type, "error": self.error,
             "fetched_at": self.fetched_at, "final_url": self.final_url}
        return d


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    retrieved_at: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url,
                "snippet": self.snippet, "retrieved_at": self.retrieved_at}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_private_host(host: str) -> bool:
    """Refuse literal private/loopback/link-local addresses (basic SSRF guard).
    Hostnames are allowed — Kratos is a local tool, not a proxy — but the
    obvious 'fetch my router/localhost' literals are blocked."""
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return host.lower() in {"localhost", "localhost.localdomain"}


def record_research_note(kind: str, payload: dict, project_dir: Path | None = None) -> None:
    """Append a research event to .kratos/research.jsonl (best-effort)."""
    try:
        target_dir = project_dir or (Path.cwd() / ".kratos")
        target_dir.mkdir(parents=True, exist_ok=True)
        entry = {"ts": _now(), "kind": kind, **payload}
        with open(target_dir / "research.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def collect_research_sources(project_dir: Path | None = None,
                             since_iso: str | None = None,
                             max_sources: int = 20) -> list[str]:
    """Return the REAL web sources actually fetched/searched this run.

    Reads ``.kratos/research.jsonl`` (written by web_fetch / web_search) and
    returns only successful, deduplicated sources. ``since_iso`` filters to
    entries at/after a run-start timestamp so old notes don't leak in.

    This is what the Reporter uses for the "Websuche/Webscraping" section —
    sources are PROVEN by the runtime, never taken from model text. That makes
    the fabricated-research-notes failure mode (session 2026-06-13) impossible:
    if the agent did not actually fetch anything, no sources are reported.
    """
    target_dir = project_dir or (Path.cwd() / ".kratos")
    path = Path(target_dir) / "research.jsonl"
    if not path.exists():
        return []
    sources: list[str] = []
    seen: set[str] = set()
    try:
        for line in path.read_text("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_iso and str(entry.get("ts", "")) < since_iso:
                continue
            kind = entry.get("kind")
            if kind == "web_fetch" and entry.get("ok"):
                url = entry.get("final_url") or entry.get("url")
                if url and url not in seen:
                    seen.add(url)
                    sources.append(url)
            elif kind == "web_search" and entry.get("ok"):
                for r in entry.get("results", []) or []:
                    url = r.get("url") if isinstance(r, dict) else None
                    if url and url not in seen:
                        seen.add(url)
                        sources.append(url)
            if len(sources) >= max_sources:
                break
    except OSError:
        return sources
    return sources


def build_request(
    url: str,
    timeout_seconds: int = 20,
    data: bytes | None = None,
    extra_headers: dict | None = None,
) -> urllib.request.Request:
    """Build an HTTP GET (or POST when *data* is provided) request.

    ``extra_headers`` are merged on top of the default headers, allowing
    callers (e.g. the DuckDuckGo search path) to override the User-Agent
    with a browser UA so the endpoint does not serve a 202 interstitial.
    Kept as a separate function so tests can verify headers without network access.
    """
    headers: dict[str, str] = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5",
        "Accept-Language": "en;q=0.9,de;q=0.8",
    }
    if extra_headers:
        headers.update(extra_headers)
    method = "POST" if data is not None else "GET"
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def web_fetch(url: str, timeout_seconds: int = 20,
              max_bytes: int = _MAX_BYTES,
              project_dir: Path | None = None,
              data: bytes | None = None,
              extra_headers: dict | None = None,
              _opener=None) -> FetchResult:
    """HTTP(S) GET (or POST when *data* is provided) with timeout, UA, status/
    content-type checks and size cap.

    ``data``         — raw POST body; when set the request is a POST.
    ``extra_headers``— merged on top of default headers (e.g. to override UA).
    ``_opener``      — injectable for tests (object with ``open(req, timeout=…)``).
    Never raises; always returns a FetchResult.
    """
    result = FetchResult(url=url, ok=False, fetched_at=_now(), final_url=url)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        result.error = f"unsupported scheme {parsed.scheme!r} (only http/https)"
        return result
    if not parsed.hostname:
        result.error = "missing host"
        return result
    if _is_private_host(parsed.hostname):
        result.error = f"refused private/loopback host {parsed.hostname!r}"
        return result

    opener = _opener or urllib.request.build_opener(
        urllib.request.HTTPRedirectHandler  # default handler caps redirects (10); fine
    )
    request = build_request(url, timeout_seconds, data=data, extra_headers=extra_headers)
    try:
        with opener.open(request, timeout=timeout_seconds) as resp:
            status = getattr(resp, "status", None) or resp.getcode() or 0
            result.status = int(status)
            result.final_url = resp.geturl() or url
            headers = getattr(resp, "headers", None)
            ctype = (headers.get("Content-Type", "") if headers else "") or ""
            result.content_type = ctype.split(";")[0].strip().lower()
            if result.status >= 400:
                result.error = f"HTTP {result.status}"
            elif result.content_type and not _TEXTUAL_CONTENT_RE.search(result.content_type):
                result.error = f"non-textual content-type {result.content_type!r} — skipped"
            else:
                raw = resp.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raw = raw[:max_bytes]
                charset = "utf-8"
                if headers:
                    m = re.search(r"charset=([\w-]+)", headers.get("Content-Type", "") or "", re.I)
                    if m:
                        charset = m.group(1)
                try:
                    result.text = raw.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    result.text = raw.decode("utf-8", errors="replace")
                result.ok = True
    except urllib.error.HTTPError as exc:
        result.status = exc.code
        result.error = f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        result.error = f"network error: {exc.reason}"
    except (TimeoutError, OSError, ValueError) as exc:
        result.error = f"fetch failed: {exc}"

    record_research_note("web_fetch", result.to_dict(), project_dir)
    return result


# ── HTML → text extraction (html.parser, no regex-only parsing) ──────────────

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
               "section", "article", "header", "footer", "pre", "blockquote"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        lines = [" ".join(ln.split()) for ln in joined.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def scrape_text_from_html(html: str, max_chars: int = 40_000) -> str:
    """Extract readable text from HTML using html.parser (never regex-only)."""
    if not html:
        return ""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:
        pass  # html.parser is tolerant; keep whatever was extracted
    text = extractor.text()
    return text[:max_chars]


# ── web search (provider adapter) ─────────────────────────────────────────────

_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.S | re.I,
)
_DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
    re.S | re.I,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _ddg_decode_href(href: str) -> str:
    """DuckDuckGo HTML wraps targets as /l/?uddg=<urlencoded>."""
    parsed = urllib.parse.urlparse(href)
    if parsed.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return urllib.parse.unquote(qs["uddg"][0])
    if href.startswith("//"):
        return "https:" + href
    return href


def _strip_tags(fragment: str) -> str:
    return " ".join(_TAG_RE.sub(" ", fragment).split())


def parse_duckduckgo_html(html: str, max_results: int) -> list[SearchResult]:
    """Best-effort parse of the duckduckgo html endpoint result page."""
    now = _now()
    titles = list(_DDG_RESULT_RE.finditer(html))
    snippets = [m.group("snippet") for m in _DDG_SNIPPET_RE.finditer(html)]
    results: list[SearchResult] = []
    for i, m in enumerate(titles[:max_results]):
        results.append(SearchResult(
            title=_strip_tags(m.group("title"))[:200],
            url=_ddg_decode_href(m.group("href")),
            snippet=_strip_tags(snippets[i])[:400] if i < len(snippets) else "",
            retrieved_at=now,
        ))
    return results


def web_search(query: str, max_results: int = 5, timeout_seconds: int = 20,
               provider: str = "duckduckgo",
               project_dir: Path | None = None,
               _fetch=None) -> tuple[list[SearchResult], str]:
    """Search the web. Returns ``(results, error)`` — exactly one is non-empty.

    ``provider="duckduckgo"`` uses the key-less HTML endpoint. Any other
    provider name returns the honest 'not configured' error. ``_fetch`` is
    injectable for tests (callable with the web_fetch signature subset).
    """
    query = (query or "").strip()
    if not query:
        return [], "empty query"
    if provider != "duckduckgo":
        return [], f"Web search provider {provider!r} not configured"

    fetch = _fetch or web_fetch
    # POST to DuckDuckGo's HTML endpoint with a browser User-Agent.
    # GET requests with the KratosAgent UA now receive a 202 anti-bot interstitial
    # (no search results); POST + desktop-browser UA returns a proper 200 result page.
    ddg_url = "https://html.duckduckgo.com/html/"
    post_body = urllib.parse.urlencode({"q": query}).encode("utf-8")
    ddg_headers = {
        "User-Agent": _DDG_BROWSER_UA,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        "Accept-Language": "en-US,en;q=0.9",
    }
    page = fetch(ddg_url, timeout_seconds=timeout_seconds, project_dir=project_dir,
                 data=post_body, extra_headers=ddg_headers)
    if not page.ok:
        err = f"web search unavailable: {page.error or 'fetch failed'}"
        record_research_note("web_search", {"query": query, "ok": False, "error": err}, project_dir)
        return [], err
    results = parse_duckduckgo_html(page.text, max_results)
    record_research_note("web_search", {
        "query": query, "ok": True, "result_count": len(results),
        "results": [r.to_dict() for r in results],
    }, project_dir)
    if not results:
        return [], "web search returned no parseable results"
    return results, ""
