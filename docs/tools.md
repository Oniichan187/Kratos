# Kratos Tool-Referenz

Der Coder steuert Tools über Text-Marker (robust für kleine/abliterated Modelle,
kein natives Tool-Calling nötig). Jede Aktion erzeugt eine Observation, die in
den nächsten Modell-Turn zurückgespeist wird.

## Datei-Tools

| Marker | Wirkung |
|---|---|
| `### READ: <pfad>` | Datei lesen (Exzerpt mit Zeilenbereich) |
| `### READ_RANGE: <pfad>:<start>-<ende>` | exakten Zeilenbereich lesen, z. B. `src/app.py:40-90` |
| `### FILE: <pfad>` + ```-Block | Datei vollständig schreiben (Snapshot vor erstem Write) |
| `### DELETE: <pfad>` | Datei löschen (nur mit `permission high`, wird begründet geloggt) |

Pfade werden normalisiert (Backslash und Slash akzeptiert) und sind auf das
Projektverzeichnis beschränkt. Encoding: UTF-8 mit `errors="replace"`.

## Suche

| Marker | Wirkung |
|---|---|
| `### SEARCH: <pattern> [:: <glob>]` | **Smart Search** — eine Pipeline: literal (case-insensitive) → `a\|b`-Alternation → Regex → Keyword-Fallback bei Prosa-Beschreibungen (extrahiert Identifier, Quoted-Strings, Eigennamen). Findet praktisch immer etwas Nützliches; die Strategie wird in der Observation ausgewiesen |
| `### GREP: <regex> [:: <glob>]` | Regex-first; bei 0 Treffern oder ungültiger Regex automatischer Fallback auf Smart Search |
| `### GLOB: <pattern>` | Dateinamen-Suche, z. B. `**/*.test.ts` oder `*.py` |

Pfad-Auflösung überall (READ/READ_RANGE/FILE): existiert `fixtures/x.html` nicht,
aber genau eine Projektdatei endet auf diesen Pfad (z. B.
`starter_project/fixtures/x.html`), wird sie automatisch verwendet (Suffix-Match,
nur bei eindeutigem Treffer). `### INSPECT:` mit Prosa-Beschreibung wird in Smart
Search umgeleitet statt verworfen; schreibfähige Befehle bleiben blockiert.

Ausgeschlossene Ordner: `.git`, `node_modules`, `venv`, `__pycache__`, `bin`,
`obj`, `dist`, `build`, `.kratos` u. a. Binärdateien werden erkannt und übersprungen.

## Shell

| Marker / API | Wirkung |
|---|---|
| `### VERIFY: <cmd>` / `### RUN: <cmd>` | Allowlist-geprüfte Build/Test-Befehle (pytest, dotnet, npm, cargo, go, …) |
| `### INSPECT: <cmd>` | nur lesende Diagnose-Befehle (Get-ChildItem, Select-String, git diff/status, rg, …) |
| `ShellRunner.run_powershell / run_cmd / run_command` | programmatische Ausführung: Timeout, cwd, stdout/stderr getrennt, Exitcode |

Alle Shell-Pfade laufen zusätzlich durch die SafetyGuard-Blocklist (siehe `docs/safety.md`).
Teststatus wird ausschließlich aus echten `CommandResult`-Exitcodes abgeleitet.

## Web (Recherche)

| Marker | Wirkung |
|---|---|
| `### WEB_SEARCH: <query>` | Websuche über Provider-Adapter. Standard: DuckDuckGo-HTML-Endpoint (kein API-Key). Nicht erreichbar/konfiguriert → ehrliche Fehlermeldung, niemals erfundene Treffer |
| `### WEB_FETCH: <url>` | HTTP-GET (nur http/https): User-Agent, Timeout, Statuscode- und Content-Type-Prüfung, 2-MB-Limit; HTML wird per `html.parser` zu Text extrahiert |

**Was Websuche kann:** Doku-/API-Recherche, Fehlermeldungen nachschlagen, direkte URLs laden.
**Was sie nicht kann:** kein JavaScript-Rendering, keine Logins, keine privaten/Loopback-Hosts,
keine Downloads > 2 MB, keine Ausführung geladener Inhalte. Jede Anfrage und Quelle wird in
`.kratos/research.jsonl` dokumentiert (Titel, URL, Snippet, Zeitstempel).

## Verifikation & Logging

- `reporter.verify_files_changed(root, files, snapshots)` — Ground-Truth-Abgleich Disk↔Snapshot
- `reporter.build_final_report(...)` — Endstatus nur aus Evidenz
- `logger.SessionLogger` — JSONL: jede Tool-Aktion, Dateiänderung (`file_write` mit Größe),
  jeder Befehl (`build_test` mit Exitcode + Output-Tail), `file_change_evidence`, `final_report`.
  Secrets werden nicht gesucht und nicht ausgegeben; Secret-Dateien (`.env`, `*.pem`, …)
  sind bereits vom Index ausgeschlossen.
