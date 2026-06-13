"""Four-tier memory: session (RAM), task (RAM), project (disk), longterm (disk).

Semantic extraction of new entries is done by compress.generate_memory() so
this module stays generic — no project-specific patterns hardcoded here.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


_SECRET_RE = re.compile(
    r'(?:api[-_]?key\s*[:=]\s*\S+|'
    r'password\s*[:=]\s*\S+|'
    r'passwort\s*[:=]\s*\S+|'
    r'secret\s*[:=]\s*\S+|'
    r'token\s*[:=]\s*\S+|'
    r'private[-_]?key\s*[:=]|'
    r'bearer\s+[A-Za-z0-9\-._~+/]+=*|'
    r'sk-[A-Za-z0-9]{20,}|'
    r'ghp_[A-Za-z0-9]{36}|'
    r'-----BEGIN\s+(?:RSA|EC|OPENSSH|PRIVATE))',
    re.I,
)


def _contains_secret(text: str) -> bool:
    return bool(_SECRET_RE.search(text))


@dataclass
class MemoryEntry:
    category: str    # decision | file_role | error_cause | solution | convention | todo
    content:  str
    file_path: str | None = None
    tags:      list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


class MemoryManager:
    def __init__(self, project_dir: Path, global_dir: Path) -> None:
        self._project_file = project_dir / "memory.json"
        self._global_file  = global_dir  / "memory.json"
        self._session:  list[MemoryEntry] = []
        self._task:     list[MemoryEntry] = []
        self._project:  list[MemoryEntry] = self._load(self._project_file)
        self._longterm: list[MemoryEntry] = self._load(self._global_file)

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self, path: Path) -> list[MemoryEntry]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text("utf-8"))
            if not isinstance(data, list):
                return []
            return [MemoryEntry.from_dict(d) for d in data if isinstance(d, dict)]
        except Exception:
            return []

    def _persist(self, entries: list[MemoryEntry], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, entry: MemoryEntry, tier: str = "session") -> None:
        if _contains_secret(entry.content):
            return
        if tier == "session":
            self._session.append(entry)
        elif tier == "task":
            self._task.append(entry)
        elif tier == "project":
            if not any(e.content == entry.content for e in self._project):
                self._project.append(entry)
                self._persist(self._project, self._project_file)
        elif tier == "longterm":
            if not any(e.content == entry.content for e in self._longterm):
                self._longterm.append(entry)
                self._persist(self._longterm, self._global_file)

    def add_from_compress(self, entries: list[dict], tier: str = "project") -> None:
        """Bulk-add entries produced by compress.generate_memory()."""
        for e in entries:
            cat     = str(e.get("category", "decision"))
            content = str(e.get("content", ""))[:200]
            if content and not _contains_secret(content):
                self.add(MemoryEntry(category=cat, content=content), tier)

    def track_files(self, changed_files: list[str]) -> None:
        """Record which files were touched in this task (task tier — session only)."""
        for fp in changed_files:
            self.add(MemoryEntry("file_role", f"Modified: {fp}", file_path=fp), "task")

    # ── read ──────────────────────────────────────────────────────────────────

    def get_relevant(
        self,
        keywords: list[str],
        categories: list[str] | None = None,
        limit: int = 8,
    ) -> list[MemoryEntry]:
        all_entries = self._task + self._session + self._project + self._longterm

        def score(e: MemoryEntry) -> int:
            s = 0
            cl = e.content.lower()
            for kw in keywords:
                if kw.lower() in cl:
                    s += 1
            if categories and e.category in categories:
                s += 2
            return s

        scored = sorted(((score(e), e) for e in all_entries), key=lambda x: -x[0])
        return [e for s, e in scored if s > 0][:limit]

    def format_for_prompt(self, entries: list[MemoryEntry]) -> str:
        if not entries:
            return ""
        lines = ["## Prior Context"]
        for e in entries:
            fp   = f" ({e.file_path})" if e.file_path else ""
            tags = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"- [{e.category}]{fp}{tags}: {e.content[:200]}")
        return "\n".join(lines)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def clear_task(self) -> None:
        self._task.clear()

    def clear_session(self) -> None:
        self._session.clear()

    def clear_project(self) -> None:
        self._project.clear()
        if self._project_file.exists():
            self._project_file.unlink()

    def list_all(self) -> dict[str, list[MemoryEntry]]:
        return {
            "session":  list(self._session),
            "task":     list(self._task),
            "project":  list(self._project),
            "longterm": list(self._longterm),
        }
