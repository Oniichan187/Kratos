"""Rule-based intent classifier — no LLM required."""

from __future__ import annotations

import re
from enum import Enum

from .analyzer import InputAnalysis


class Intent(str, Enum):
    QUESTION      = "question"
    CODING        = "coding"
    BUGFIX        = "bugfix"
    REFACTOR      = "refactor"
    BUILD_ERROR   = "build_error"
    TEST_ERROR    = "test_error"
    LOG_ANALYSIS  = "log_analysis"
    FILE_SEARCH   = "file_search"
    CODE_SEARCH   = "code_search"
    EXPLAIN       = "explain"
    PLAN_ONLY     = "plan_only"
    DIRECT_IMPL   = "direct_impl"
    SHELL_GIT     = "shell_git"
    DOCS          = "docs"
    CONFIG_CHANGE = "config_change"
    DEPENDENCY    = "dependency"
    PERFORMANCE   = "performance"
    SECURITY      = "security"
    UI            = "ui"
    DATABASE      = "database"
    FOLLOWUP      = "followup"
    UNCLEAR       = "unclear"


# (intent, list-of-regex-patterns) — first match in priority order wins
_RULES: list[tuple[Intent, list[str]]] = [
    (Intent.BUILD_ERROR, [
        r'\b(?:build|compile|kompilier|linker|cmake|msbuild|dotnet\s+build|mvn\s+build|gradle\s+build)\b.{0,60}(?:error|fail|fehler)',
        r'(?:compilation|build)\s+(?:error|fail)',
        r'\bcannot find symbol\b',
        r'\bSyntaxError\b.*line\s+\d+',
        r'\bModuleNotFoundError\b',
    ]),
    (Intent.TEST_ERROR, [
        r'\b(?:test|spec|unittest|pytest|jest|mocha|vitest)\b.{0,60}(?:fail(?:ed|ing)?|error)',
        r'\bfailing tests?\b',
        r'\bAssertionError\b',
        r'\btest\s+output\b',
    ]),
    (Intent.BUGFIX, [
        r'\b(?:fix|bug|fehler(?:hafte?)?|broken|kaputt|crash|absturz)\b',
        r'\b(?:funktioniert\s+nicht|doesn\'?t?\s+work|not\s+working|incorrect|wrong\s+output)\b',
        r'\b(?:warum\s+(?:schlägt|gibt|ist)|why\s+(?:is|does|do)\b).{0,60}\?',
    ]),
    (Intent.SHELL_GIT, [
        r'\bgit\s+(?:commit|push|pull|merge|rebase|reset|checkout|stash|tag|cherry-pick)\b',
        r'\b(?:run|execute|ausführen|starten|führe\s+aus)\b.{0,30}\b(?:npm|pip|bash|sh|make|docker)\b',
        r'\bnpm\s+(?:run|install|build|test)\b',
    ]),
    (Intent.REFACTOR, [
        r'\b(?:refactor(?:ier)?|umstrukturier|aufräum(?:en)?|clean\s*up|vereinfach|simplify)\b',
        r'\b(?:extract(?:ier)?|auslagern|move|verschiebn?)\b.{0,40}\b(?:class|method|function|module|klasse|funktion)\b',
        r'\b(?:rename|umbenennen|umbenennung)\b',
    ]),
    (Intent.CODING, [
        # Prefix-match German verb stems (covers conjugations: -e, -en, -t, -st)
        r'\b(?:vervollst(?:ä|ae)ndig|fertigstell|implementier|erstell|bau|hinzufüg|programmier|entwickel)\w*'
        r'\W.{0,80}\b(?:function|funktion|class|klasse|method|module|api|endpoint|feature|component|route|service|projekt|project)\b',
        r'\b(?:create?|implement\w*|build|complete\w*|finish\w*)\W.{0,60}\b(?:function|class|method|module|api|endpoint|feature|component|route|service|project)\b',
        r'\b(?:neue?\s+(?:function|funktion|class|klasse|feature|endpoint|route|model|module|component|service))\b',
        r'\b(?:vervollst(?:ä|ae)ndig|complet|finish)\w*\W.{0,40}\b(?:project|projekt|cli|app|script|tool)\b',
    ]),
    (Intent.PLAN_ONLY, [
        r'\b(?:plan(?:e|en)?|analysier|architektur(?:plan)?|konzept|strategi|design\s+(?:doc|plan)|vorschlag)\b',
        r'\b(?:wie\s+(?:soll(?:te)?|würde|kann)\s+ich|how\s+should\s+(?:i|we))\b.{0,60}\?',
        r'\b(?:zeig\s+mir\s+(?:die|den)\s+plan|show\s+me\s+(?:a\s+)?plan)\b',
        r'\b(?:was\s+(?:wäre|ist)\s+der\s+beste\s+(?:weg|ansatz)|what(?:\'s|\s+is)\s+the\s+best\s+(?:way|approach))\b',
    ]),
    (Intent.EXPLAIN, [
        r'\b(?:erkl(?:är|are|äre|aer|ärung)|explain)\b',
        r'\b(?:was\s+macht|what\s+does|wie\s+funktioniert|how\s+(?:does|do)\s+(?:it|this))\b',
        r'\b(?:was\s+ist|what\s+is|was\s+bedeutet|what(?:\'s|\s+is)\s+the\s+(?:purpose|role))\b',
        r'\b(?:zeig\s+mir|show\s+me)\b.{0,30}\b(?:wie|how|was|what)\b',
    ]),
    (Intent.DOCS, [
        r'\b(?:doku(?:mentation)?|docstring|javadoc)\b',
        # Prefix-match verb stems: schreib→schreibe/schreiben, erstell→erstelle
        r'\b(?:schreib|writ|add|erstell|generat|updat)\w*\W.{0,30}\b(?:doku|docs?|comments?|readme)\b',
    ]),
    (Intent.CONFIG_CHANGE, [
        r'\b(?:config(?:uration)?|konfiguration?|settings?|einstellung)\b.{0,40}\b(?:änder|update|set|setze|change|edit)\b',
        r'\b(?:yaml|json|toml|ini|\.env)\b.{0,40}\b(?:änder|update|edit|bearbeit|modify)\b',
    ]),
    (Intent.DEPENDENCY, [
        r'\b(?:dependency|abhängigkeit|paket|package)\b.{0,40}\b(?:hinzufüg|add|install|update|upgrade|remove)\b',
        r'\b(?:pip\s+install|npm\s+install|nuget|cargo\s+add|go\s+get)\b',
        r'\b(?:requirements?\.txt|package\.json|Cargo\.toml|pom\.xml)\b.{0,30}\b(?:add|update)\b',
    ]),
    (Intent.PERFORMANCE, [
        r'\b(?:performance|langsam|slow|optimier|optimize|profil(?:ing)?|bottleneck|memory\s+leak)\b',
        r'\b(?:speed\s+up|faster|schneller|reduce\s+(?:latency|memory)|improve\s+(?:speed|performance))\b',
    ]),
    (Intent.SECURITY, [
        r'\b(?:security|sicherheit|vulnerability|CVE-\d+|injection|XSS|CSRF|authentication|authorization)\b',
        r'\b(?:exploit|attack|secure|absicher(?:n|e|ung)|penetration)\b',
    ]),
    (Intent.DATABASE, [
        r'\b(?:database|datenbank|SQL|query|migration|schema|ORM|JOIN|SELECT|INSERT|UPDATE|DELETE)\b',
        r'\b(?:tabelle|table|index|foreign\s+key|primary\s+key|constraint)\b',
    ]),
    (Intent.UI, [
        r'\b(?:frontend|CSS|HTML|component|layout|button|form|modal|responsive|design)\b',
        r'\b(?:react|vue|angular|svelte|tailwind|bootstrap|style|stylesheet)\b',
    ]),
    (Intent.LOG_ANALYSIS, [
        r'\b(?:logs?|logfile|logging)\b.{0,40}\b(?:analysier|check|prüf|analyz|show|zeig|read|lies)\b',
        r'\b(?:was\s+steht|what(?:\'s|\s+is)\s+in)\b.{0,30}\blog\b',
        r'\b(?:log\s+output|error\s+log|application\s+log)\b',
    ]),
    (Intent.QUESTION, [
        r'\?$',
        r'^(?:wie|was|warum|wann|wo|wer|welche|can|could|would|how|what|why|when|where|who|which)\b',
        r'\b(?:kannst\s+du|can\s+you|could\s+you|würdest\s+du|weißt\s+du)\b',
    ]),
    # FILE_SEARCH and CODE_SEARCH are lowest priority — only matched for short inputs
    # (long prompts that mention files are usually coding tasks, not searches)
    (Intent.FILE_SEARCH, [
        r'\b(?:finde?|suche?|where\s+is|wo\s+ist|locate)\b.{0,40}\b(?:file|datei|ordner|folder)\b',
        r'\b(?:welche?\s+dateien?|which\s+files?|list\s+files?|zeige?\s+dateien?)\b',
        r'\b(?:in\s+welcher\s+datei|in\s+which\s+file)\b',
        r'\b(?:wo\s+ist|where\s+is)\b.{0,60}\.(?:py|js|ts|go|rs|cs|java|kt|json|yaml|yml|toml|md|txt)\b',
    ]),
    (Intent.CODE_SEARCH, [
        r'\b(?:finde?|suche?|find|where|wo)\b.{0,40}\b(?:function|funktion|methode?|method|class|klasse|variable|constant)\b',
        r'\b(?:welche?\s+klasse|which\s+class|where\s+is\s+\w+\s+defined|wo\s+(?:ist|wird|steht)\s+\w+)\b',
        r'\b(?:grep|search\s+(?:for|in)|suche\s+nach|finde\s+alle)\b',
    ]),
]

_COMPILED: list[tuple[Intent, list[re.Pattern]]] = [
    (intent, [re.compile(p, re.I | re.S) for p in patterns])
    for intent, patterns in _RULES
]


class IntentClassifier:
    def classify(self, analysis: InputAnalysis) -> Intent:
        if analysis.is_followup:
            return Intent.FOLLOWUP

        if analysis.has_stacktrace:
            return Intent.BUGFIX

        if analysis.has_log and analysis.error_lines:
            return Intent.LOG_ANALYSIS

        if analysis.has_git_cmd:
            return Intent.SHELL_GIT

        text  = analysis.normalized
        words = len(text.split())

        # For short inputs (≤ 30 words): check FILE_SEARCH/CODE_SEARCH BEFORE
        # QUESTION, because "wo ist die config.py" starts with "wo" which would
        # match the QUESTION pattern first.
        if words <= 30:
            for intent, patterns in _COMPILED:
                if intent not in (Intent.FILE_SEARCH, Intent.CODE_SEARCH):
                    continue
                for pat in patterns:
                    if pat.search(text):
                        return intent

        # Main loop — skip FILE_SEARCH/CODE_SEARCH (handled above or too long)
        for intent, patterns in _COMPILED:
            if intent in (Intent.FILE_SEARCH, Intent.CODE_SEARCH):
                continue
            for pat in patterns:
                if pat.search(text):
                    return intent

        if words <= 3:
            return Intent.QUESTION

        # Long inputs (> 20 words) that matched nothing are almost always
        # coding / multi-step tasks, not ambiguous "unclear" requests.
        if words > 20:
            return Intent.CODING

        return Intent.UNCLEAR
