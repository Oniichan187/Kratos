"""Map intent → pipeline route."""

from __future__ import annotations

from enum import Enum

from .classifier import Intent


class Route(str, Enum):
    DIRECT_ANSWER      = "direct_answer"       # No LLM — search/list from index
    CONTEXT_ONLY       = "context_only"         # Show context, no LLM
    PLANNER_ONLY       = "planner_only"         # Planner answers, no Coder
    CODER_ONLY         = "coder_only"           # Coder only, no Planner
    PLANNER_THEN_CODER = "planner_then_coder"   # Full pipeline
    DIAGNOSTIC_LOOP    = "diagnostic_loop"      # Coder + build/test retry loop
    ASK_CLARIFICATION  = "ask_clarification"    # Request more info from user


_ROUTE_MAP: dict[Intent, Route] = {
    # ── no LLM needed ────────────────────────────────────────────────────────
    Intent.FILE_SEARCH:   Route.DIRECT_ANSWER,
    Intent.CODE_SEARCH:   Route.DIRECT_ANSWER,

    # ── planner only (no code output) ────────────────────────────────────────
    Intent.QUESTION:      Route.PLANNER_ONLY,
    Intent.EXPLAIN:       Route.PLANNER_ONLY,
    Intent.PLAN_ONLY:     Route.PLANNER_ONLY,
    Intent.LOG_ANALYSIS:  Route.PLANNER_ONLY,

    # ── coder only (continuation or direct command — no planning needed) ─────
    Intent.FOLLOWUP:      Route.CODER_ONLY,   # continue previous context
    Intent.SHELL_GIT:     Route.CODER_ONLY,   # just generate the command

    # ── planner → coder (default for all coding tasks) ───────────────────────
    Intent.CODING:        Route.PLANNER_THEN_CODER,
    Intent.BUGFIX:        Route.PLANNER_THEN_CODER,
    Intent.REFACTOR:      Route.PLANNER_THEN_CODER,
    Intent.DIRECT_IMPL:   Route.PLANNER_THEN_CODER,
    Intent.DOCS:          Route.PLANNER_THEN_CODER,
    Intent.CONFIG_CHANGE: Route.PLANNER_THEN_CODER,
    Intent.DEPENDENCY:    Route.PLANNER_THEN_CODER,
    Intent.PERFORMANCE:   Route.PLANNER_THEN_CODER,
    Intent.SECURITY:      Route.PLANNER_THEN_CODER,
    Intent.UI:            Route.PLANNER_THEN_CODER,
    Intent.DATABASE:      Route.PLANNER_THEN_CODER,

    # ── diagnostic loop (build/test + retry) ─────────────────────────────────
    Intent.BUILD_ERROR:   Route.DIAGNOSTIC_LOOP,
    Intent.TEST_ERROR:    Route.DIAGNOSTIC_LOOP,

    # ── unclear ───────────────────────────────────────────────────────────────
    Intent.UNCLEAR:       Route.ASK_CLARIFICATION,
}


class Router:
    def route(self, intent: Intent) -> Route:
        return _ROUTE_MAP.get(intent, Route.PLANNER_THEN_CODER)
