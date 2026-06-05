"""Input analysis: language detection, artifact extraction, follow-up detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InputAnalysis:
    raw: str
    normalized: str
    language: str           # "de" | "en"
    is_followup: bool
    has_file_paths: bool
    file_paths: list[str]
    has_code_block: bool
    has_stacktrace: bool
    has_log: bool
    has_git_cmd: bool
    has_shell_cmd: bool
    keywords: list[str]
    error_lines: list[str]


_FOLLOWUP_DE = re.compile(
    r'\b(mach\s+weiter|weiter\s+so|weiterführen|fortfahren|fix\s+das|'
    r'wie\s+vorher|nochmal|nochmals|wiederhol|gleiche\s+wie|dasselbe|'
    r'geh\s+weiter|mach\s+das|und\s+weiter|weiter\s+machen)\b',
    re.I,
)
_FOLLOWUP_EN = re.compile(
    r'\b(continue|keep\s+going|fix\s+it|fix\s+that|do\s+it|go\s+on|'
    r'proceed|same\s+as\s+before|as\s+before|do\s+the\s+same|'
    r'keep\s+going|and\s+continue|next\s+step)\b',
    re.I,
)

_PATH_WIN = re.compile(r'[A-Za-z]:[\\/][\w.\\/\-]+')
_PATH_REL = re.compile(r'(?:\.{1,2}[\\/]|(?:[\w][\w\-]*/){1,6}[\w][\w.\-]+\.[\w]{1,6})')
_PATH_POSIX = re.compile(r'/(?:home|usr|var|etc|tmp|mnt|opt)/[\w./\-]+')

_STACKTRACE = re.compile(
    r'(?:Traceback\s*\(most recent call last\)|'
    r'^\s+File ".*",\s+line \d+|'
    r'\b(?:NameError|TypeError|ValueError|AttributeError|ImportError|'
    r'RuntimeError|KeyError|IndexError|OSError|IOError):\s|'
    r'at \w[\w.]+\(\w[\w.]+\.(?:java|kt|cs|py):\d+\))',
    re.M,
)
_CODE_BLOCK = re.compile(r'```[\s\S]{8,}```|`[^`\n]{6,}`')
_GIT_CMD = re.compile(
    r'\bgit\s+(?:commit|push|pull|merge|rebase|reset|checkout|branch|'
    r'status|log|diff|add|stash|tag|clone|fetch|cherry-pick)\b',
    re.I,
)
_SHELL_CMD = re.compile(
    r'\b(?:npm|pip|python3?|bash|sh|cmd|powershell|pwsh|dotnet|'
    r'cargo|go\s+run|mvn|gradle|make|docker|kubectl)\b\s+\S+',
    re.I,
)
_LOG_ERROR = re.compile(
    r'^\s*(?:\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}[:\d.]*\s+)?'
    r'(?:ERROR|WARN(?:ING)?|FATAL|CRITICAL)\b.+$',
    re.M,
)

_DE_MARKERS = re.compile(
    r'\b(?:ich|du|wir|der|die|das|und|ist|nicht|aber|wie|was|mach|erstell|'
    r'bau|füg|zeig|erklär|warum|welche|welchen|nein|ja|bitte|danke|'
    r'hinzufüg|entfern|lösch|änder|schreib|wieso|weshalb|also|wenn)\b',
    re.I,
)

_STOP_WORDS = frozenset(
    'a an the is to in of and for it on at by be do go if me my no or so up '
    'we you all any are but can get has have him his how its let may new nor '
    'not now old one our out own run say she see set sub too two use was way '
    'who why yet ich du wir der die das und ist ein eine als aus bei mit von '
    'zur zum des den dem nach über unter vor'.split()
)


class InputAnalyzer:
    def analyze(self, text: str) -> InputAnalysis:
        norm = text.strip()
        paths = list(dict.fromkeys(
            _PATH_WIN.findall(norm)
            + _PATH_REL.findall(norm)
            + _PATH_POSIX.findall(norm)
        ))
        errors = _LOG_ERROR.findall(norm)

        return InputAnalysis(
            raw=text,
            normalized=norm,
            language="de" if _DE_MARKERS.search(norm) else "en",
            is_followup=bool(_FOLLOWUP_DE.search(norm) or _FOLLOWUP_EN.search(norm)),
            has_file_paths=bool(paths),
            file_paths=paths,
            has_code_block=bool(_CODE_BLOCK.search(norm)),
            has_stacktrace=bool(_STACKTRACE.search(norm)),
            has_log=bool(errors),
            has_git_cmd=bool(_GIT_CMD.search(norm)),
            has_shell_cmd=bool(_SHELL_CMD.search(norm)),
            keywords=self._keywords(norm),
            error_lines=errors,
        )

    @staticmethod
    def _keywords(text: str) -> list[str]:
        words = re.findall(r'\b[A-Za-z_]\w{2,}\b', text)
        seen: set[str] = set()
        out: list[str] = []
        for w in words:
            wl = w.lower()
            if wl not in _STOP_WORDS and wl not in seen:
                seen.add(wl)
                out.append(wl)
                if len(out) >= 30:
                    break
        return out
