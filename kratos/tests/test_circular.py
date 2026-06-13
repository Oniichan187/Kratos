"""Tests for the deterministic circular-import breaker (last-resort repair)."""

from pathlib import Path

from kratos.execution.circular import find_import_cycles, break_unused_circular_imports


def _make_pkg(tmp_path: Path, scraper_src: str, cli_src: str) -> Path:
    pkg = tmp_path / "mini_agent_check"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .scraper import parse_weather_html\n")
    (pkg / "scraper.py").write_text(scraper_src)
    (pkg / "cli.py").write_text(cli_src)
    return tmp_path


# The exact session pattern: scraper imports an UNUSED name from cli (the cycle),
# cli imports names from scraper that it actually USES.
SCRAPER_BAD = (
    "from __future__ import annotations\n"
    "from .cli import build_parser  # unused — only here it creates the cycle\n"
    "def parse_weather_html(html, city):\n"
    "    return {'city': city}\n"
)
CLI_GOOD = (
    "from __future__ import annotations\n"
    "from .scraper import parse_weather_html\n"
    "def main():\n"
    "    return parse_weather_html('', 'x')\n"
)


def test_detects_two_cycle(tmp_path):
    root = _make_pkg(tmp_path, SCRAPER_BAD, CLI_GOOD)
    cycles = find_import_cycles(root)
    keys = {frozenset(c) for c in cycles}
    assert frozenset(("mini_agent_check/scraper", "mini_agent_check/cli")) in keys


def test_breaks_cycle_by_removing_unused_import(tmp_path):
    root = _make_pkg(tmp_path, SCRAPER_BAD, CLI_GOOD)
    changes = break_unused_circular_imports(root)
    # the unused scraper→cli import must be removed; cli (uses its import) untouched
    assert "mini_agent_check/scraper.py" in changes
    assert "from .cli import build_parser" not in changes["mini_agent_check/scraper.py"]
    assert "mini_agent_check/cli.py" not in changes
    # applying the change actually breaks the cycle
    (root / "mini_agent_check/scraper.py").write_text(changes["mini_agent_check/scraper.py"])
    assert find_import_cycles(root) == []


def test_does_not_touch_when_both_sides_used(tmp_path):
    # both directions use their imports → not safely auto-fixable, leave alone
    scraper = (
        "from .cli import WeatherCard\n"
        "def parse(html):\n"
        "    return WeatherCard()\n"
    )
    cli = (
        "from .scraper import parse\n"
        "class WeatherCard: pass\n"
        "def main():\n    return parse('')\n"
    )
    root = _make_pkg(tmp_path, scraper, cli)
    changes = break_unused_circular_imports(root)
    assert changes == {}   # nothing removed; model must resolve a genuine mutual dependency


def test_no_cycle_no_change(tmp_path):
    root = _make_pkg(tmp_path,
                     "def parse_weather_html(h, c):\n    return {}\n",
                     "from .scraper import parse_weather_html\ndef main():\n    return parse_weather_html('', 'x')\n")
    assert find_import_cycles(root) == []
    assert break_unused_circular_imports(root) == {}
