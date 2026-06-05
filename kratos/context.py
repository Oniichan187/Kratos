"""Project indexer and token-aware context builder.

Design goals:
  - Works for projects of ANY size (10 files to 10 000 files).
  - File listing (paths only) is always cheap — shown fully up to 200 entries.
  - File bodies are only loaded for the top-ranked candidates and only up to
    the caller's token budget. Excerpts are cut at token level, not line level.
  - The architecture scope always starts by listing ALL files so the planner
    knows the full project structure, then loads as many bodies as fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .tokens import estimate, fit_excerpt

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

ScopeType = Literal[
    "none", "minimal", "targeted", "expanded",
    "diagnostic", "architecture", "full_index_only", "patch_context",
]

# Max number of file BODIES to load per scope (listing is always full)
_SCOPE_MAX_BODIES: dict[str, int] = {
    "none":           0,
    "minimal":        3,
    "targeted":       6,
    "expanded":       12,
    "diagnostic":     8,
    "architecture":   40,    # planner reads all it can fit in budget
    "full_index_only": 0,
    "patch_context":  4,
}


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


@dataclass
class ContextPackage:
    user_input:          str
    intent:              str
    route:               str
    memory_summary:      str = ""
    project_description: str = ""
    files:               list[FileEntry] = field(default_factory=list)
    error_lines:         list[str] = field(default_factory=list)
    token_budget:        int = 12288

    def to_prompt(self) -> str:
        parts: list[str] = []
        if self.project_description:
            parts.append(f"## Project\n{self.project_description}")
        if self.memory_summary:
            parts.append(self.memory_summary)
        if self.files:
            parts.append("## Relevant Files")
            for f in self.files:
                parts.append(f"### {f.rel_path}\n```\n{f.excerpt(1500)}\n```")
        if self.error_lines:
            parts.append("## Errors\n" + "\n".join(self.error_lines[:30]))
        return "\n\n".join(p for p in parts if p)


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


class ContextBuilder:
    def __init__(self, indexer: ProjectIndexer) -> None:
        self.indexer = indexer

    def _load(self, entry: FileEntry) -> None:
        if entry.content is not None:
            return
        try:
            raw = entry.path.read_text(encoding="utf-8", errors="replace")
            if len(raw) > _MAX_FILE_BYTES:
                raw = raw[:_MAX_FILE_BYTES] + "\n... (truncated)"
            entry.content = raw
        except OSError:
            entry.content = ""

    def _score_name(
        self, entry: FileEntry, keywords: list[str], file_paths: list[str]
    ) -> float:
        score = float(entry.priority) * 2
        name_l = entry.rel_path.lower()
        for fp in file_paths:
            fp_l = fp.lower().replace("\\", "/")
            if fp_l in name_l or name_l.endswith(fp_l.lstrip("./\\")):
                score += 25.0
        for kw in keywords:
            if kw.lower() in name_l:
                score += 4.0
        return score

    def _score_content(self, entry: FileEntry, keywords: list[str]) -> float:
        if not entry.content:
            return 0.0
        cl = entry.content.lower()
        return sum(min(cl.count(kw.lower()), 6) * 0.8 for kw in keywords)

    def build(
        self,
        analysis,       # InputAnalysis
        intent: str,
        route: str,
        memory_summary: str = "",
        scope: ScopeType = "targeted",
        token_budget: int = 12288,
    ) -> ContextPackage:
        pkg = ContextPackage(
            user_input=analysis.normalized,
            intent=intent,
            route=route,
            memory_summary=memory_summary,
            error_lines=analysis.error_lines,
            token_budget=token_budget,
        )

        if scope == "none":
            return pkg

        index = self.indexer.index
        if not index:
            return pkg

        # File listing: always include ALL files (paths only — cheap)
        # Cap display at 200 to avoid overflowing context with just the listing
        listing_entries = index[:200]
        listing = "\n".join(f"  {e.rel_path}" for e in listing_entries)
        suffix = f"\n  … and {len(index) - 200} more" if len(index) > 200 else ""
        pkg.project_description = (
            f"Root: {self.indexer.root.name}  ({len(index)} files total)\n"
            f"All files:\n{listing}{suffix}"
        )

        if scope == "full_index_only":
            return pkg

        # Body loading — respect token budget
        max_bodies = _SCOPE_MAX_BODIES.get(scope, 5)
        if max_bodies == 0:
            return pkg

        # Score by filename relevance first (cheap)
        name_scored = sorted(
            ((self._score_name(e, analysis.keywords, analysis.file_paths), e)
             for e in index),
            key=lambda x: -x[0],
        )
        # Take top 3× the limit as candidates, then re-rank by content
        candidates = [e for s, e in name_scored if s > 0][: max_bodies * 3]
        if not candidates and max_bodies > 0:
            # Fallback: take top-priority files even if score == 0
            candidates = index[: max_bodies * 2]

        for e in candidates:
            self._load(e)

        final_scored = sorted(
            ((self._score_name(e, analysis.keywords, analysis.file_paths)
              + self._score_content(e, analysis.keywords), e)
             for e in candidates),
            key=lambda x: -x[0],
        )
        ranked = [e for _, e in final_scored[:max_bodies]]

        # Token-budget: distribute evenly among selected files, respecting budget
        # Reserve ~30 % for listing + memory + framing
        body_budget = int(token_budget * 0.65)
        per_file    = max(200, body_budget // max(1, len(ranked)))

        selected: list[FileEntry] = []
        remaining = body_budget
        for entry in ranked:
            if remaining <= 0:
                break
            alloc = min(per_file, remaining)
            # Re-attach a budget-capped excerpt without mutating original content
            fe = FileEntry(
                path=entry.path,
                rel_path=entry.rel_path,
                priority=entry.priority,
                size=entry.size,
                content=entry.content,
            )
            tok = estimate(entry.content or "")
            if tok > alloc:
                fe.content = fit_excerpt(entry.content or "", alloc)
            remaining -= min(tok, alloc)
            selected.append(fe)

        pkg.files = selected
        return pkg
