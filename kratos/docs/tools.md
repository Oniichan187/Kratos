# Kratos-Tools (Datei, Suche, Shell, Web)

## Datei & Suche — `execution/search.py`

Reine Standardbibliothek, auf das Projektwurzelverzeichnis beschränkt.

| Funktion | Zweck |
|---|---|
| `list_files(root)` | rekursive Auflistung (relative POSIX-Pfade) |
| `glob_files(root, pattern)` | Glob (`**/*.py`, `src/**/util*`) |
| `search_text(root, pattern, glob, case_sensitive)` | literale Substring-Suche |
| `search_regex(root, regex, glob, case_sensitive)` | Regex-Suche (Fehlerstring bei ungültigem Pattern) |
| `read_file_range(root, rel, start, end)` | exakter Zeilenbereich (1-basiert) |
| `smart_search(root, pattern)` | mehrstufige Suche (literal → Alternation → Regex → Keyword) |

Jeder Treffer (`SearchMatch`) trägt **Datei, Zeile, Spalte, Match-Text und
Kontextzeilen** — so rät der Coder keine Dateien blind.

Standardmäßig ignorierte Verzeichnisse: `.git .svn .hg .idea .vscode
__pycache__ .pytest_cache node_modules .venv venv env dist build target out bin
obj .next .cache coverage .tox .eggs .kratos .claude models`.

Binärdateien (Null-Byte-Sniff) und Dateien > 2 MB werden bei der Suche
übersprungen.

## Shell — `execution/shell.py` (`ShellRunner`)

Windows-first: PowerShell (`pwsh` > `powershell`) und CMD sind erstklassig,
`bash`/`sh` ist POSIX/WSL-Fallback. Jeder Lauf:

- geht durch `safety.check_command` (blockiert == wird nie ausgeführt),
- setzt Working Directory und hartes Timeout,
- erfasst stdout/stderr **getrennt** plus Exitcode,
- behandelt UTF-8 sauber (`errors="replace"`),
- liefert ein serialisierbares `CommandResult`-Dict:
  `cmd, shell, exit_code, stdout, stderr, duration_seconds, blocked,
  block_reason, timed_out, cwd, timeout_seconds`.

Kein `shell=True` für die Interpreter-Aufrufe (Binär + Argument direkt).

## Web — `web.py`

| Funktion | Verhalten |
|---|---|
| `web_fetch(url, timeout_seconds, max_bytes)` | HTTP(S) GET: User-Agent `KratosAgent/1.0`, Timeout, Statuscheck, Content-Type-Check, 2-MB-Cap, Schema nur http/https, SSRF-Guard (private/loopback-Hosts verweigert) |
| `scrape_text_from_html(html)` | Textextraktion via `html.parser` (kein Regex-only), entfernt script/style/nav |
| `web_search(query, provider)` | Provider-Adapter; ohne konfigurierten Provider ehrliches `Web search provider not configured`, **keine erfundenen Quellen** |
| `record_research_note(...)` | protokolliert jede Web-Aktion nach `.kratos/research.jsonl` für die Reporter-Quellen |

Keine Shell-Kommandos zum Laden von Webseiten. Antworten sind Daten, werden nie
ausgeführt.

## Wird Websuche/Webscraping verlangt?

`core/agent.py` erkennt aus dem Task-Text Schlüsselwörter (Websuche, Recherche,
URL, http(s)://) und setzt `web_requested` im Final-Report. Abschnitt
"Websuche/Webscraping" zeigt dann Verlangt/Durchgeführt/Quellen bzw. den Grund.

## Web-Recherche end-to-end (für schwache/abliterated Modelle)

Webscraping ist nicht nur eine Bibliothek, sondern im Agenten-Loop verdrahtet:

1. **Tool-Marker** — der Coder ruft Web-Tools über Marker auf:
   `### WEB_FETCH: <url>` und `### WEB_SEARCH: <query>` (siehe
   `execution/tools.py: do_web_fetch / do_web_search`). Beide Coder-Loops
   (`run_coder_loop` und der Default `execute_structured_work_steps_for_plan`)
   führen sie über `_run_lookup_actions` aus.
2. **Nudge für schwache Modelle** — verlangt der Task Recherche/URLs, injiziert
   `roles/coder.py: _web_research_hint(task)` eine explizite Anweisung:
   „Du MUSST die Tools wirklich aufrufen … erfinde keine Quellen." Das
   verhindert das Halluzinieren von Links.
3. **Echte Quellen im Report** — jede Fetch/Search-Aktion wird nach
   `.kratos/research.jsonl` geloggt. Am Ende sammelt
   `web.collect_research_sources(project_dir, since_iso=run_start)` nur die
   **tatsächlich** und **erfolgreich** abgerufenen Quellen dieses Laufs.
   `core/agent.py` übergibt sie an den Report. Hat der Agent nichts geholt,
   meldet der Report ehrlich „Durchgeführt: Nein" — fabrizierte Quellen sind
   damit strukturell unmöglich.
4. **Sicherheit** — nur http/https, kein Shell-Download, 2-MB-Cap, Timeout,
   User-Agent, SSRF-Guard (private/loopback-Hosts verweigert).
