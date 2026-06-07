"""Token-aware context builder.

Design goals:
  - Works for projects of ANY size (10 files to 10 000 files).
  - File listing (paths only) is always cheap — shown fully up to 200 entries.
  - File bodies are only loaded for the top-ranked candidates and only up to
    the caller's token budget. Excerpts are cut at token level, not line level.
  - The architecture scope always starts by listing ALL files so the planner
    knows the full project structure, then loads as many bodies as fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..llm.tokens import estimate, fit_excerpt
from .indexer import FileEntry, ProjectIndexer

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
class ContextPackage:
    user_input:          str
    intent:              str
    route:               str
    memory_summary:      str = ""
    project_description: str = ""
    files:               list[FileEntry] = field(default_factory=list)
    error_lines:         list[str] = field(default_factory=list)
    token_budget:        int = 12288
    # NEW (best-possible vector "gets"): high-signal chunks from dynamic retrieval
    # These are the primary source of relevant code for large projects.
    # The old .files (whole-file excerpts) are kept for compatibility/small scopes.
    retrieved_chunks: list = field(default_factory=list)  # list[RetrievedChunk]

    def to_prompt(self) -> str:
        parts: list[str] = []
        if self.project_description:
            parts.append(f"## Project\n{self.project_description}")
        if self.memory_summary:
            parts.append(self.memory_summary)
        if self.retrieved_chunks:
            parts.append("## Relevant Code (retrieved via vector knowledge base)")
            for ch in self.retrieved_chunks[:12]:  # keep prompt reasonable
                try:
                    parts.append(ch.to_prompt_block(900))
                except Exception:
                    parts.append(f"### {getattr(ch, 'rel_path', '?')}\n{getattr(ch, 'text', '')[:800]}")
        if self.files and not self.retrieved_chunks:
            # Fallback / small-project path: only show whole files if we have no rich chunks
            parts.append("## Relevant Files")
            for f in self.files:
                parts.append(f"### {f.rel_path}\n```\n{f.excerpt(1500)}\n```")
        if self.error_lines:
            parts.append("## Errors\n" + "\n".join(self.error_lines[:30]))
        return "\n\n".join(p for p in parts if p)


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
            # Will be populated by caller (agent) with dynamic retrieval results
            retrieved_chunks=[],
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
