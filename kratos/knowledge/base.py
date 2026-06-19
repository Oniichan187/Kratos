"""Vector Knowledge Base for Kratos — best-possible dynamic retrieval ("gets").

This turns the project into a persistent, queryable vector database.
The agent performs continuous, targeted semantic + hybrid "gets" at key moments
(task start, before planner, and especially fresh per-step in stepwise mode)
instead of a static upfront index that dumps bodies into prompts.

Design goals (per approved plan):
- Real vector DB (LanceDB by default — lightweight, serverless, file-based, excellent for local laptop RAG).
- Smart chunking: symbol-centric (functions/classes/tests) + semantic/docs/config chunks.
- Embeddings via Ollama (small embed model, same WSL host as other models).
- Hybrid retrieval: vector similarity + symbol/identifier match + keyword/path/priority + memory facts.
- Continuous "gets": cheap, repeatable retrieval passes using the *current* query (task or exact step text).
- Long-term knowledge: code chunks + durable memory facts are embedded.
- WSL-friendly: DB on shared .kratos/ FS (or WSL-native); embeds go through the bridge to WSL Ollama.
- 6 GB 4050 friendly: embed model is small (low VRAM), vector search is CPU/RAM/disk, main inference stays sequential.
- Graceful degradation: works (with symbols + tree + memory) if no embed model or vector backend unavailable.
- Preserves abliterated nature, max-ctx policy, etc.

The TUI and core stepwise/PROVEN_WORK/lossless compressor contracts are untouched.
Old ProjectIndexer can still provide the cheap global file tree for orientation.

Usage (will be wired in agent.py):
    kb = ProjectKnowledge(config, bridge)
    kb.rebuild()  # or incremental on changes
    chunks = kb.retrieve("implement the FooBar parser handling escaped quotes", top_k=16)
    # chunks -> list of RetrievedChunk with path, symbol, text, summary, scores, etc.

Persistence: .kratos/knowledge/ (lancedb tables + metadata).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Optional heavy deps — graceful fallback
try:
    import lancedb  # type: ignore
    _HAS_LANCEDB = True
except Exception:
    _HAS_LANCEDB = False

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


from ..llm.bridge import OllamaBridge
from ..config import KratosConfig, _project_dir
from ..memory import MemoryManager
from ..llm.tokens import estimate  # rough token helper for budgets

# Files up to this size are read whole and chunked normally. Larger files are
# NOT skipped — they are seek-sampled (a few KB per window) so a file of any size
# is ingested without being loaded into memory. There is deliberately no upper
# size limit on what the knowledge base will ingest.
_FULL_READ_BYTES = 4_000_000   # 4 MB


@dataclass
class RetrievedChunk:
    """A high-signal piece of project knowledge retrieved via hybrid search."""
    rel_path: str
    symbol: str | None = None
    kind: str | None = None  # function, class, test, doc, config, ...
    start_line: int | None = None
    end_line: int | None = None
    text: str = ""
    summary: str = ""  # optional dense compressor-generated role summary
    score: float = 0.0
    vector_score: float = 0.0
    symbol_score: float = 0.0
    keyword_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self, max_chars: int = 1200) -> str:
        """Compact block suitable for LLM prompts."""
        header = f"### {self.rel_path}"
        if self.symbol:
            header += f" :: {self.symbol} ({self.kind or 'symbol'})"
        if self.start_line and self.end_line:
            header += f" (L{self.start_line}-{self.end_line})"
        header += f"  [score={self.score:.2f}]"

        body = self.summary or self.text
        if len(body) > max_chars:
            body = body[:max_chars] + "\n... [truncated]"

        return f"{header}\n```\n{body}\n```"


@dataclass
class _ChunkRecord:
    """Internal record for storage (before/after embedding)."""
    id: str
    rel_path: str
    symbol: str | None
    kind: str | None
    start_line: int | None
    end_line: int | None
    text: str
    summary: str
    mtime: float
    metadata: dict[str, Any]
    vector: list[float] | None = None  # set after embedding


class ProjectKnowledge:
    """The project's living vector knowledge base + retrieval engine.

    Exposes the key "get" operation the agent will call continuously.
    """

    def __init__(self, config: KratosConfig, bridge: OllamaBridge) -> None:
        self.config = config
        self.bridge = bridge

        self._root = Path.cwd().resolve()
        self._kb_dir = (_project_dir() / "knowledge").resolve()
        self._kb_dir.mkdir(parents=True, exist_ok=True)

        self._db = None  # LanceDB connection or None
        self._table = None  # LanceDB table or None
        self._fallback_chunks: list[_ChunkRecord] = []  # when no vector backend

        self._embed_model = getattr(config, "embed_model", "nomic-embed-text")
        self._top_k_default = getattr(config, "retrieval_top_k", 16)

        self._load_or_init_store()

    # ── store lifecycle ─────────────────────────────────────────────────────

    def _load_or_init_store(self) -> None:
        if not _HAS_LANCEDB:
            # Pure fallback: in-memory list + brute-force cosine on next retrieve
            # (still useful; user can `pip install lancedb` later)
            self._fallback_chunks = self._load_fallback_index()
            return

        try:
            self._db = lancedb.connect(str(self._kb_dir))
            table_name = "project_chunks"
            if table_name in self._db.table_names():
                self._table = self._db.open_table(table_name)
            else:
                # Will be created on first add
                self._table = None
        except Exception:
            # Any LanceDB issue → fallback mode
            self._db = None
            self._table = None
            self._fallback_chunks = self._load_fallback_index()

    def _load_fallback_index(self) -> list[_ChunkRecord]:
        idx = self._kb_dir / "fallback_index.json"
        if not idx.exists():
            return []
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            recs: list[_ChunkRecord] = []
            for d in data:
                recs.append(_ChunkRecord(**d))
            return recs
        except Exception:
            return []

    def _save_fallback_index(self) -> None:
        idx = self._kb_dir / "fallback_index.json"
        try:
            data = [r.__dict__ for r in self._fallback_chunks]
            idx.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ── public API the agent will call ──────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """The core "get" — dynamic hybrid retrieval for the current sub-task.

        This is called continuously:
        - Once for the overall task (planner context)
        - Freshly before every coder step (using the exact step text as query)

        Returns the most relevant chunks (vector + symbol + keyword + memory boost).
        """
        k = top_k or self._top_k_default
        if not query or not query.strip():
            return []

        # 1. Embed the query (once, cheap)
        qvec = self._embed([query.strip()])[0] if self._has_embedding() else None

        if self._table is not None and qvec is not None:
            return self._retrieve_lancedb(qvec, k, filters)

        # Fallback or no-embed path: hybrid on whatever we have (symbols + text + memory)
        return self._retrieve_fallback(query, k, filters)

    def rebuild(
        self,
        force: bool = False,
        embed_model: str | None = None,
    ) -> int:
        """(Re)build the vector knowledge base for the current project.

        Returns number of chunks indexed.
        This is what /knowledge rebuild (or enhanced /index) will call.
        """
        if embed_model:
            self._embed_model = embed_model

        chunks = self._discover_and_chunk_project()
        if not chunks:
            return 0

        # Optionally generate dense summaries with the compressor (small + fast)
        self._maybe_add_compressor_summaries(chunks)

        if self._table is not None or _HAS_LANCEDB:
            return self._index_into_lancedb(chunks, force=force)

        # Pure fallback
        self._fallback_chunks = chunks
        self._save_fallback_index()
        return len(chunks)

    def status(self) -> dict[str, Any]:
        """Lightweight status for /knowledge status or UI."""
        n_chunks = 0
        if self._table is not None:
            try:
                n_chunks = self._table.count_rows()
            except Exception:
                n_chunks = -1
        elif self._fallback_chunks:
            n_chunks = len(self._fallback_chunks)

        return {
            "backend": "lancedb" if self._table is not None else "fallback",
            "embed_model": self._embed_model,
            "chunks": n_chunks,
            "kb_dir": str(self._kb_dir),
            "has_lancedb": _HAS_LANCEDB,
        }

    # ── chunking (best-possible, symbol-centric) ─────────────────────────────

    def _chunk_markdown_artifact(
        self,
        rel_path: str,
        text: str,
        *,
        mtime: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[_ChunkRecord]:
        metadata = dict(metadata or {})
        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        heading_re = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
        sections: list[tuple[str, int, int, str]] = []
        current_title = "document"
        current_start = 1
        current_lines: list[str] = []

        def flush(end_line: int) -> None:
            nonlocal current_lines, current_title, current_start
            body = "".join(current_lines).strip()
            if body:
                sections.append((current_title, current_start, end_line, body))
            current_lines = []

        for idx, line in enumerate(lines, 1):
            match = heading_re.match(line)
            if match:
                if current_lines:
                    flush(idx - 1)
                current_title = match.group(1).strip() or "section"
                current_start = idx
                current_lines = [line]
                continue
            if not current_lines:
                current_start = idx
            current_lines.append(line)

        if current_lines:
            flush(len(lines))

        if not sections:
            sections.append(("document", 1, len(lines), text.strip()))

        stamp = float(mtime or time.time())
        records: list[_ChunkRecord] = []
        for idx, (title, start, end, body) in enumerate(sections, 1):
            cid = f"{rel_path}:artifact:{idx}:{hash((rel_path, title, start, body[:160])) & 0xffffffff:x}"
            records.append(
                _ChunkRecord(
                    id=cid,
                    rel_path=rel_path,
                    symbol=title,
                    kind="compression_artifact",
                    start_line=start,
                    end_line=end,
                    text=body,
                    summary="",
                    mtime=stamp,
                    metadata={**metadata, "artifact_kind": "compression", "section": title},
                )
            )
        return records

    def _ingest_chunk_records(self, chunks: list[_ChunkRecord], force: bool = False) -> int:
        if not chunks:
            return 0
        if self._table is not None or _HAS_LANCEDB:
            return self._index_into_lancedb(chunks, force=force)
        self._fallback_chunks.extend(chunks)
        self._save_fallback_index()
        return len(chunks)

    def ingest_markdown_artifact(self, path: Path, metadata: dict[str, Any] | None = None) -> int:
        """Incrementally ingest a Markdown artifact without rebuilding the whole KB."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            mtime = path.stat().st_mtime
        except Exception:
            return 0
        try:
            rel = str(path.resolve().relative_to(self._root)).replace("\\", "/")
        except Exception:
            rel = str(path.name)
        chunks = self._chunk_markdown_artifact(rel, text, mtime=mtime, metadata=metadata)
        return self._ingest_chunk_records(chunks, force=False)

    def _discover_and_chunk_project(self) -> list[_ChunkRecord]:
        """Walk the project and produce high-quality chunks.

        Reuses spirit of the old priority rules + ignore logic (from context.py)
        but is not bound by the old FileEntry design.
        """
        # Lightweight reuse of existing ignore logic (avoid duplicating regexes)
        from ..context.indexer import _IGNORE, _SECRET_FILE, _PRIORITY_RULES  # type: ignore

        records: list[_ChunkRecord] = []
        seen: set[str] = set()

        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if _IGNORE.search(path.as_posix()) or _SECRET_FILE.search(path.name):
                continue

            try:
                rel = str(path.relative_to(self._root)).replace("\\", "/")
                mtime = path.stat().st_mtime
            except Exception:
                continue

            try:
                size = path.stat().st_size
            except Exception:
                continue

            # No size cap — handling projects of ANY size is exactly what the
            # vector DB is for. But stay memory-safe: files up to _FULL_READ_BYTES
            # are read whole and chunked normally; bigger files are represented by
            # evenly-spaced windows read via seek (a few KB each), so a file of any
            # size is ingested across its whole length without loading it into RAM.
            if size > _FULL_READ_BYTES:
                for fc in self._sampled_big_file_chunks(path, max_chunks=80):
                    cid = f"{rel}:bytes:{fc['offset']}"
                    if cid in seen:
                        continue
                    seen.add(cid)
                    records.append(
                        _ChunkRecord(
                            id=cid,
                            rel_path=rel,
                            symbol=None,
                            kind="chunk",
                            start_line=None,
                            end_line=None,
                            text=fc["text"],
                            summary="",
                            mtime=mtime,
                            metadata={"priority": self._priority_for(rel), "sampled": True},
                        )
                    )
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Symbol-centric chunks (the "best" part)
            sym_chunks = self._extract_symbol_chunks(rel, text)
            for sc in sym_chunks:
                cid = f"{rel}:{sc['symbol']}:{sc['start']}"
                if cid in seen:
                    continue
                seen.add(cid)
                records.append(
                    _ChunkRecord(
                        id=cid,
                        rel_path=rel,
                        symbol=sc["symbol"],
                        kind=sc["kind"],
                        start_line=sc["start"],
                        end_line=sc["end"],
                        text=sc["text"],
                        summary="",
                        mtime=mtime,
                        metadata={"priority": self._priority_for(rel)},
                    )
                )

            # Fallback / additional semantic chunks for non-code or large files.
            # Bound per-file windows so a multi-MB data file can't dominate the KB;
            # the sampler spreads them across the whole file.
            if not sym_chunks or len(text) > 4000:
                for fc in self._fallback_semantic_chunks(rel, text, max_chunks=80):
                    cid = f"{rel}:chunk:{fc['start']}"
                    if cid in seen:
                        continue
                    seen.add(cid)
                    records.append(
                        _ChunkRecord(
                            id=cid,
                            rel_path=rel,
                            symbol=None,
                            kind="chunk",
                            start_line=fc["start"],
                            end_line=fc["end"],
                            text=fc["text"],
                            summary="",
                            mtime=mtime,
                            metadata={"priority": self._priority_for(rel)},
                        )
                    )

        # Also turn durable memory facts into embeddable "chunks" (long-term knowledge)
        mem = MemoryManager(_project_dir(), _project_dir().parent / ".kratos")
        for tier, entries in mem.list_all().items():
            for e in entries:
                content = f"[{e.category}] {e.content}"
                if e.file_path:
                    content += f" (file: {e.file_path})"
                cid = f"memory:{tier}:{hash(content) & 0xffffffff:x}"
                if cid not in seen:
                    seen.add(cid)
                    records.append(
                        _ChunkRecord(
                            id=cid,
                            rel_path=e.file_path or f".kratos/memory/{tier}",
                            symbol=None,
                            kind="memory",
                            start_line=None,
                            end_line=None,
                            text=content,
                            summary="",
                            mtime=time.time(),
                            metadata={"tier": tier, "category": e.category},
                        )
                    )

        records.extend(self._discover_compression_artifacts())
        return records

    def _discover_compression_artifacts(self) -> list[_ChunkRecord]:
        """Load Markdown compression artifacts saved under .kratos/knowledge/compressions."""
        artifacts_dir = (_project_dir() / "knowledge" / "compressions").resolve()
        if not artifacts_dir.exists():
            return []

        records: list[_ChunkRecord] = []
        for path in artifacts_dir.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                mtime = path.stat().st_mtime
                rel = str(path.relative_to(self._root)).replace("\\", "/")
            except Exception:
                continue
            records.extend(
                self._chunk_markdown_artifact(
                    rel,
                    text,
                    mtime=mtime,
                    metadata={"tier": "project", "artifact_kind": "compression"},
                )
            )
        return records

    def _extract_symbol_chunks(self, rel_path: str, text: str) -> list[dict]:
        """Simple but effective multi-lang symbol extraction (no tree-sitter for laptop lightness)."""
        chunks: list[dict] = []
        lines = text.splitlines(keepends=True)

        # Common patterns (extend as needed; good enough for "best on this machine")
        patterns = [
            (r"^\s*(def\s+([A-Za-z_]\w*)\s*\()", "function"),
            (r"^\s*(async\s+def\s+([A-Za-z_]\w*)\s*\()", "async_function"),
            (r"^\s*(class\s+([A-Za-z_]\w*)\s*[:\(])", "class"),
            (r"^\s*(interface\s+([A-Za-z_]\w*)\s*[{<])", "interface"),
            (r"^\s*(struct\s+([A-Za-z_]\w*)\s*[{<])", "struct"),
            (r"^\s*(fn\s+([A-Za-z_]\w*)\s*\()", "function"),  # rust
            (r"^\s*(func\s+([A-Za-z_]\w*)\s*\()", "function"),  # go
            (r"^\s*(public|private|protected|internal)?\s*(static)?\s*(class|record|interface)\s+([A-Za-z_]\w*)", "cs_type"),
            (r"^\s*(export\s+)?(async\s+)?(function|const|let|var)\s+([A-Za-z_]\w*)\s*[=:(<]?", "ts_js"),
        ]

        for i, line in enumerate(lines):
            for pat, kind in patterns:
                m = re.match(pat, line, re.I)
                if m:
                    # Capture a reasonable body window (until next symbol or blank block)
                    symbol = m.group(2) or m.group(4) or m.group(1)
                    start = max(0, i - 1)  # include a bit of context
                    end = min(len(lines), i + 40)  # reasonable function size
                    # Stop at next obvious symbol at same or lower indent
                    for j in range(i + 1, min(len(lines), i + 80)):
                        if re.match(r"^\s*(def |class |interface |struct |fn |func |public |private |export |async def )", lines[j], re.I):
                            end = j
                            break
                    chunk_text = "".join(lines[start:end]).rstrip()
                    chunks.append({
                        "symbol": symbol,
                        "kind": kind,
                        "start": start + 1,
                        "end": end,
                        "text": chunk_text,
                    })
                    break  # next line
        return chunks

    def _fallback_semantic_chunks(self, rel_path: str, text: str,
                                  max_chunks: int = 0) -> list[dict]:
        """Overlapping chunks for files without clear symbols or that are very long.

        ``max_chunks`` bounds the number of windows produced for one file. When a
        file would yield more windows than that (e.g. a multi-MB ``.jsonl`` log),
        the windows are sampled evenly across the WHOLE file instead of taking
        only the head, so the data is represented end-to-end without exploding the
        knowledge base. ``0`` means unbounded (original behavior)."""
        chunks = []
        lines = text.splitlines(keepends=True)
        if not lines:
            return chunks
        size = 35
        overlap = 8
        starts = list(range(0, len(lines), size - overlap))
        if max_chunks and len(starts) > max_chunks:
            if max_chunks >= 2:
                sampled = {
                    starts[(i * (len(starts) - 1)) // (max_chunks - 1)]
                    for i in range(max_chunks)
                }
                starts = sorted(sampled)
            else:
                starts = starts[:1]
        for start in starts:
            end = min(len(lines), start + size)
            chunk_text = "".join(lines[start:end]).rstrip()
            if chunk_text:
                chunks.append({
                    "start": start + 1,
                    "end": end,
                    "text": chunk_text,
                })
        return chunks

    def _sampled_big_file_chunks(self, path: Path, max_chunks: int = 80,
                                 window_bytes: int = 4096) -> list[dict]:
        """Memory-safe representative chunks for an arbitrarily large file.

        Only small windows at evenly-spaced byte offsets are read (never the whole
        file), so a file of ANY size — a multi-GB ``.jsonl`` log included — is
        represented across its entire length while peak memory stays at roughly
        ``max_chunks * window_bytes``. Returns ``[{"offset": int, "text": str}]``."""
        out: list[dict] = []
        try:
            size = path.stat().st_size
        except OSError:
            return out
        if size <= 0:
            return out
        n = max(1, max_chunks)
        seen: set[str] = set()
        try:
            with path.open("rb") as fh:
                for i in range(n):
                    offset = (size * i) // n
                    fh.seek(offset)
                    if offset > 0:
                        fh.readline()  # discard partial line → align to a line start
                    raw = fh.read(window_bytes)
                    if not raw:
                        continue
                    block = raw.decode("utf-8", errors="replace")
                    lines = block.splitlines()
                    if len(lines) > 1:
                        lines = lines[:-1]   # drop trailing partial line
                    chunk_text = "\n".join(lines).strip()
                    if not chunk_text:
                        continue
                    key = chunk_text[:160]
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"offset": offset, "text": chunk_text})
        except OSError:
            return out
        return out

    def _priority_for(self, rel_path: str) -> int:
        # Lightweight reuse of the spirit of the old rules (from context/indexer.py)
        from ..context.indexer import _PRIORITY_RULES  # type: ignore
        for pri, pat in _PRIORITY_RULES:
            if pat.search(rel_path):
                return pri
        return 0

    # ── embedding (via bridge, WSL Ollama) ───────────────────────────────────

    def _has_embedding(self) -> bool:
        return bool(self._embed_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama embed via the bridge (same host as the chat models — WSL friendly)."""
        if not texts:
            return []
        try:
            return self.bridge.embed(texts, model=self._embed_model)
        except Exception:
            # Embedding failed (model not pulled, etc.) — return zeros so hybrid still works
            dim = 768
            return [[0.0] * dim for _ in texts]

    # ── indexing into store ──────────────────────────────────────────────────

    def _maybe_add_compressor_summaries(self, chunks: list[_ChunkRecord]) -> None:
        """Optional: use the (tiny, fast) compressor model to create dense summaries."""
        if not getattr(self.config, "enable_dense_chunk_summaries", True):
            return
        # Only for a reasonable number; compressor is cheap but we don't want to burn time on 10k chunks
        for c in chunks[:200]:
            if c.summary:
                continue
            # Very short prompt to the compressor (already ablit + tuned for dense facts)
            try:
                user = f"Create a 1-sentence dense role summary for this code chunk from {c.rel_path} (symbol: {c.symbol or 'n/a'}):\n\n{c.text[:1500]}"
                # We reuse the existing compressor call pattern (non-streaming)
                from .compress import Compressor
                comp = Compressor(self.bridge, self.config)
                # The compressor has a private _call_model; we do a direct small call here
                # to avoid importing internal details. For simplicity we call the bridge chat.
                summary = ""
                for tok, kind in self.bridge.chat(
                    model=self.config.compressor_model,
                    messages=[
                        {"role": "system", "content": "You create extremely dense one-sentence summaries of code chunks for retrieval."},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                    num_predict=64,
                    num_ctx=4096,
                    think=False,
                    keep_alive="0",
                ):
                    if kind == "text":
                        summary += tok
                c.summary = summary.strip()[:300]
            except Exception:
                c.summary = ""

    def _index_into_lancedb(self, chunks: list[_ChunkRecord], force: bool = False) -> int:
        assert self._db is not None
        table_name = "project_chunks"

        # Prepare data for LanceDB (needs vectors)
        texts = [c.text + ("\n" + c.summary if c.summary else "") for c in chunks]
        vectors = self._embed(texts)

        data = []
        for c, vec in zip(chunks, vectors):
            c.vector = vec
            data.append({
                "id": c.id,
                "rel_path": c.rel_path,
                "symbol": c.symbol or "",
                "kind": c.kind or "",
                "start_line": c.start_line or 0,
                "end_line": c.end_line or 0,
                "text": c.text,
                "summary": c.summary,
                "mtime": c.mtime,
                "metadata": json.dumps(c.metadata),
                "vector": vec,
            })

        if self._table is None or force:
            if table_name in self._db.table_names():
                self._db.drop_table(table_name)
            self._table = self._db.create_table(table_name, data=data)
        else:
            # Incremental: delete old versions of these ids, then add
            try:
                ids = [d["id"] for d in data]
                self._table.delete(f"id in {ids}")  # LanceDB supports this
            except Exception:
                pass
            self._table.add(data)

        return len(data)

    # ── retrieval implementations ────────────────────────────────────────────

    def _retrieve_lancedb(self, qvec: list[float], k: int, filters: dict | None) -> list[RetrievedChunk]:
        assert self._table is not None
        try:
            # LanceDB search
            res = self._table.search(qvec).limit(k * 2).to_pandas()  # overfetch for hybrid rerank
        except Exception:
            return []

        # Simple hybrid rerank on the result set (metadata is cheap)
        scored: list[tuple[float, dict]] = []
        for _, row in res.iterrows():
            vec_score = float(row.get("_distance", 1.0))  # lower is better in some versions; we invert
            # LanceDB distance is usually cosine or l2; treat smaller as better for now
            # We will combine with other signals
            text = row.get("text", "") or ""
            sym = row.get("symbol", "") or ""
            kind = row.get("kind", "") or ""
            rel = row.get("rel_path", "") or ""

            # Very lightweight additional scores (no extra model calls)
            sym_score = 2.0 if sym and sym.lower() in (filters or {}).get("query", "").lower() else 0.0
            kw_score = sum(1 for w in (filters or {}).get("query", "").lower().split() if w in text.lower() or w in rel.lower())

            final = (1.0 / (1.0 + vec_score)) + sym_score * 0.3 + kw_score * 0.1  # simple fusion
            scored.append((final, row.to_dict()))

        scored.sort(key=lambda x: -x[0])
        out: list[RetrievedChunk] = []
        for score, r in scored[:k]:
            meta = {}
            try:
                meta = json.loads(r.get("metadata", "{}"))
            except Exception:
                pass
            out.append(RetrievedChunk(
                rel_path=r.get("rel_path", ""),
                symbol=r.get("symbol") or None,
                kind=r.get("kind") or None,
                start_line=int(r.get("start_line") or 0) or None,
                end_line=int(r.get("end_line") or 0) or None,
                text=r.get("text", ""),
                summary=r.get("summary", ""),
                score=float(score),
                vector_score=float(1.0 / (1.0 + r.get("_distance", 1.0))),
                metadata=meta,
            ))
        return out

    def _retrieve_fallback(self, query: str, k: int, filters: dict | None) -> list[RetrievedChunk]:
        q = query.lower()
        scored: list[tuple[float, _ChunkRecord]] = []
        for c in self._fallback_chunks:
            s = 0.0
            if c.symbol and c.symbol.lower() in q:
                s += 3.0
            if any(w in c.text.lower() for w in q.split() if len(w) > 2):
                s += 1.0
            if c.rel_path.lower().split("/")[-1] in q:
                s += 0.5
            if c.metadata.get("tier") == "project":
                s += 0.3
            scored.append((s, c))

        scored.sort(key=lambda x: -x[0])
        out: list[RetrievedChunk] = []
        for score, c in scored[:k]:
            out.append(RetrievedChunk(
                rel_path=c.rel_path,
                symbol=c.symbol,
                kind=c.kind,
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
                summary=c.summary,
                score=float(score),
                metadata=c.metadata,
            ))
        return out
