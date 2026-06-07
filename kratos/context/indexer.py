"""Project file indexer — lazy scanner over the project tree.

Builds a prioritized list of files (paths only, cheap) once per session,
invalidated on demand. Bodies are loaded lazily by ContextBuilder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..llm.tokens import fit_excerpt

_MAX_FILE_BYTES = 80_000    # raw read cap per file (bytes) — ~20K tokens

# (priority, pattern) matched against relative path or filename
_PRIORITY_RULES: list[tuple[int, re.Pattern]] = [
    (10, re.compile(r'(?:^|[\\/])(?:main|index|app|server|__main__|entry)\.(?:py|js|ts|go|rs|cs|java)$', re.I)),
    (9,  re.compile(r'(?:^|[\\/])(?:pyproject\.toml|package\.json|Cargo\.toml|pom\.xml|build\.gradle|CMakeLists\.txt|setup\.py|setup\.cfg)$', re.I)),
    (8,  re.compile(r'(?:^|[\\/])(?:config|settings|configuration).*\.(?:py|js|ts|json|yaml|yml|toml|ini)$', re.I)),
    (7,  re.compile(r'(?:^|[\\/])(?:controller|service|handler|router|middleware).*\.(?:py|js|ts|go|rs|cs|java|kt)$', re.I)),
    (6,  re.compile(r'(?:^|[\\/])(?:model|schema|entity|dto|types?|interface).*\.(?:py|js|ts|go|rs|cs|java|kt)$', re.I)),
    (5,  re.compile(r'(?:^|[\\/])(?:test_|_test\.|\.spec\.|\.test\.).*\.(?:py|js|ts|go|rs|cs|java|kt)$', re.I)),
    (4,  re.compile(r'(?:^|[\\/])(?:README|CHANGELOG|Makefile|Dockerfile|docker-compose)(?:\.\w+)?$', re.I)),
    (4,  re.compile(r'\.github[\\/]workflows[\\/].*\.ya?ml$', re.I)),
    (3,  re.compile(r'\.(?:py|js|ts|go|rs|cs|java|kt|swift|cpp|c|h|hpp)$', re.I)),
    (3,  re.compile(r'\.(?:json|jsonl|jsonc)$', re.I)),
    (2,  re.compile(r'\.(?:md|txt|yaml|yml|toml|ini|cfg|conf|csv|tsv|xml)$', re.I)),
]

_IGNORE = re.compile(
    r'(?:^|[\\/])(?:'
    r'\.git|\.svn|\.hg|\.idea|\.vscode|__pycache__|\.mypy_cache|\.pytest_cache'
    r'|node_modules|\.venv|venv|\.env|env|dist|build|target|out|bin|obj|\.bin'
    r'|\.next|\.nuxt|\.cache|coverage|htmlcov|\.tox|\.eggs|.*\.egg-info'
    r'|models|\.kratos|\.claude'
    r')(?:[\\/]|$)',
    re.I,
)
_SECRET_FILE = re.compile(
    r'(?:^|[\\/])(?:\.env|secrets?\.|credentials?\.|.*\.pem|.*\.key|.*\.pfx|.*\.p12)$',
    re.I,
)


@dataclass
class FileEntry:
    path:     Path
    rel_path: str
    priority: int
    size:     int
    content:  str | None = None

    def excerpt(self, token_budget: int = 2000, max_lines: int | None = None) -> str:
        if not self.content:
            return ""
        return fit_excerpt(self.content, token_budget, max_lines)


class ProjectIndexer:
    """Lazy scanner — builds index once per session, invalidated on demand."""

    def __init__(self, root: Path) -> None:
        self.root   = root.resolve()
        self._index: list[FileEntry] | None = None

    def _ignored(self, path: Path) -> bool:
        rel = path.as_posix()
        return bool(_IGNORE.search(rel) or _SECRET_FILE.search(path.name))

    def _file_priority(self, path: Path) -> int:
        rel  = path.as_posix()
        name = path.name
        for pri, pat in _PRIORITY_RULES:
            if pat.search(rel) or pat.search(name):
                return pri
        return 0

    def build_index(self) -> list[FileEntry]:
        entries: list[FileEntry] = []
        try:
            for path in self.root.rglob("*"):
                if not path.is_file():
                    continue
                if self._ignored(path):
                    continue
                pri = self._file_priority(path)
                if pri == 0:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size > _MAX_FILE_BYTES * 4:
                    continue
                try:
                    rel = str(path.relative_to(self.root)).replace("\\", "/")
                except ValueError:
                    continue
                entries.append(FileEntry(path=path, rel_path=rel, priority=pri, size=size))
        except (PermissionError, OSError):
            pass
        entries.sort(key=lambda e: (-e.priority, e.size))
        self._index = entries
        return entries

    @property
    def index(self) -> list[FileEntry]:
        if self._index is None:
            self.build_index()
        return self._index

    def invalidate(self) -> None:
        self._index = None

    def search_content(
        self, keyword: str, max_results: int = 15
    ) -> list[tuple[FileEntry, int, str]]:
        results: list[tuple[FileEntry, int, str]] = []
        pat = re.compile(re.escape(keyword), re.I)
        for entry in self.index:
            try:
                content = entry.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if pat.search(line):
                    results.append((entry, i, line.strip()[:120]))
                    if len(results) >= max_results:
                        return results
        return results
