"""Deterministic circular-import breaker — a last-resort repair for when the
model cannot fix a cycle itself.

Motivation (session 2026-06-13_15-43-37): a 4B abliterated coder re-introduced
the SAME circular import 15 times in a row — even after the stall tracker
escalated with an explicit "break the cycle" instruction — by writing
``from .cli import build_parser`` at module top level (and a misleading comment
claiming it was a lazy import). No amount of prompting fixes a model that weak.

When the diagnoser reports ``circular_import`` and the repair loop has stalled,
Kratos can break the cycle WITHOUT the model: it parses the package, finds a
``from .X import name`` that (a) participates in an import cycle and (b) whose
imported name is never actually used in that file, and removes that dead import
line. Removing a provably-unused import is always safe and breaks the cycle.

Pure stdlib (``ast``), project-root-confined, returns the new file contents so
the caller can apply them through the existing snapshot/verify machinery.
"""

from __future__ import annotations

import ast
from pathlib import Path

__all__ = ["find_import_cycles", "break_unused_circular_imports"]

_IGNORE_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", ".venv", "venv", "env", "node_modules",
    "dist", "build", "bin", "obj", ".idea", ".vs", ".kratos", ".claude",
}


def _iter_package_modules(root: Path) -> dict[str, Path]:
    """Map dotted-ish module key → file path for every .py under *root*.

    Keys are POSIX relative paths without the ``.py`` suffix (e.g.
    ``pkg/scraper``), which is enough to resolve intra-package relative imports.
    """
    out: dict[str, Path] = {}
    for path in root.rglob("*.py"):
        if any(part in _IGNORE_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        out[rel[:-3]] = path  # strip ".py"
    return out


def _resolve_relative(importer_rel: str, level: int, module: str | None) -> str | None:
    """Resolve a relative ``from . import`` target to a module key.

    ``importer_rel`` is the importer's key (e.g. ``pkg/cli``). ``level`` is the
    number of leading dots. Returns the imported module key or None.
    """
    parts = importer_rel.split("/")
    # the package dir of the importer is parts[:-1]; one dot = same package
    base = parts[:-1]
    up = level - 1
    if up > len(base):
        return None
    base = base[: len(base) - up] if up else base
    if module:
        base = base + module.split(".")
    return "/".join(base) if base else None


def _name_used_outside_import(tree: ast.Module, names: set[str], import_node: ast.AST) -> set[str]:
    """Return the subset of *names* that are referenced anywhere except in the
    given import node itself."""
    used: set[str] = set()
    for node in ast.walk(tree):
        if node is import_node:
            continue
        if isinstance(node, ast.Name) and node.id in names:
            used.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in names:
            used.add(node.value.id)
    return used


def _intra_imports(tree: ast.Module, importer_rel: str, modules: dict[str, Path]):
    """Yield (import_node, target_module_key, {bound_names}) for each intra-package
    ``from ... import ...`` in *tree* that resolves to a known module."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level and node.level > 0:
            target = _resolve_relative(importer_rel, node.level, node.module)
        elif node.module:
            target = node.module.replace(".", "/")
        else:
            target = None
        if target is None:
            continue
        if target not in modules:
            # also try matching by suffix (absolute import of a sibling)
            cand = [k for k in modules if k == target or k.endswith("/" + target)]
            if len(cand) != 1:
                continue
            target = cand[0]
        bound = {(a.asname or a.name) for a in node.names if a.name != "*"}
        if bound:
            yield node, target, bound


def find_import_cycles(root: Path) -> list[tuple[str, str]]:
    """Return intra-package 2-cycles as ``(module_a, module_b)`` pairs."""
    modules = _iter_package_modules(root)
    edges: dict[str, set[str]] = {}
    for key, path in modules.items():
        try:
            tree = ast.parse(path.read_text("utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        targets = {t for _, t, _ in _intra_imports(tree, key, modules)}
        edges[key] = targets
    cycles: list[tuple[str, str]] = []
    seen: set[frozenset] = set()
    for a, outs in edges.items():
        for b in outs:
            if a in edges.get(b, set()) and a != b:
                fs = frozenset((a, b))
                if fs not in seen:
                    seen.add(fs)
                    cycles.append((a, b))
    return cycles


def break_unused_circular_imports(root: Path) -> dict[str, str]:
    """Break intra-package import cycles by deleting provably-unused cross-imports.

    For each 2-cycle (A↔B), look at both directions and remove the
    ``from .other import ...`` statement whose every bound name is unused in that
    file. Returns ``{relative_path: new_source}`` for changed files only — never
    touches a file where the imported names are actually used.
    """
    root = Path(root)
    modules = _iter_package_modules(root)
    changes: dict[str, str] = {}

    for a, b in find_import_cycles(root):
        for importer in (a, b):
            other = b if importer == a else a
            path = modules.get(importer)
            if path is None:
                continue
            try:
                src = path.read_text("utf-8", errors="replace")
                tree = ast.parse(src)
            except (OSError, SyntaxError):
                continue
            removed = False
            lines = src.splitlines(keepends=True)
            # collect removals first (line ranges), apply from the bottom up
            to_remove: list[tuple[int, int]] = []
            for node, target, bound in _intra_imports(tree, importer, modules):
                if target != other:
                    continue
                used = _name_used_outside_import(tree, bound, node)
                if used:
                    continue  # names are needed — do not touch
                start = node.lineno - 1
                end = (node.end_lineno or node.lineno) - 1
                to_remove.append((start, end))
            if not to_remove:
                continue
            for start, end in sorted(to_remove, reverse=True):
                del lines[start:end + 1]
                removed = True
            if removed:
                rel = path.relative_to(root).as_posix()
                changes[rel] = "".join(lines)
                # only break one direction of this cycle per pass
                break

    return changes
