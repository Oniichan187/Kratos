# Kratos Agent-Architektur

Pipeline pro Aufgabe:

```
Input → Analyze → Classify → Route → Context (Index + Vector-KB)
  → Planner (Checkliste) → Coder-Loop (OBSERVE→ACT) → ProvenWork-Gate
  → Real-File-Change-Gate → LLM-Verifier → Final-Report (Reporter)
  → bei Fehlern: Retry (max_verify_iterations), sonst ehrlicher FAILED-Report
```

## Module und Verantwortlichkeiten

| Modul | Aufgabe |
|---|---|
| `context/indexer.py` (ProjectIndexer) | rekursive Dateiliste, Prioritäten, ignoriert `.git`, `node_modules`, `bin`, `obj`, `venv`, `__pycache__`, `.kratos`, `dist`, `build`; Größenlimits; Secret-Dateien ausgeschlossen |
| `execution/search.py` (PatternSearch/FileReader) | `list_files`, `glob_files`, `search_text`, `search_regex`, `read_file_range` — Treffer mit Datei/Zeile/Spalte/Kontext |
| `execution/shell.py` (ShellRunner) | PowerShell/CMD/Bash, Timeout, Working Dir, stdout/stderr getrennt, Exitcode, SafetyGuard-Blocklist, Logging |
| `web.py` (WebFetchScraper + WebSearchTool) | `web_fetch` (stdlib, UA, Timeout, Statuscode, Größenlimit), `scrape_text_from_html` (html.parser), `web_search` (Provider-Adapter, Standard: DuckDuckGo-HTML, ohne API-Key); Quellen in `.kratos/research.jsonl` |
| `planning.py` (Planner-Parsing) | Plan → Checkliste (`PlanItem` mit file_refs + verify_cmd), Statusführung |
| `roles/coder.py` (Executor) | OBSERVE→ACT-Loop und strukturierter Work-Step-Driver; mehrere Micro-Turns pro Item; harte Re-Prompts bei marker-losem Output |
| `verification.py` + `core/buildtest.py` (Verifier) | Befehls-Discovery (README/Projektstruktur), Allowlist sicherer Verify-Befehle, ProvenWork-Evidenz |
| `reporter.py` (Reporter) | Final-Report NUR aus Evidenz; `verify_files_changed()` (Hash-Vergleich Snapshot↔Disk); harte Anti-Fake-Erfolg-Regeln |
| `safety.py` (SafetyGuard) | Blocklist gefährlicher Befehle, Pfad-Confinement, Schutz der Git-Interna |
| `logger.py` (SessionLogger) | JSONL-Logs jeder Tool-Aktion, Dateiänderung, jedes Befehls und Testresultats |

## Wie verhindert wird, dass der Agent nur plant

1. **Work-Step-Driver loopt pro Item** (`max_work_step_turns`, Default 4): Prosa ohne Marker
   erzeugt eine "NOTHING HAPPENED"-Observation und einen Re-Prompt; nach 2 marker-losen
   Turns wird das Item abgebrochen und vom äußeren Loop erneut versucht.
2. **Real-File-Change-Gate** in `core/agent.py`: vor dem LLM-Verifier wird per
   Snapshot-Hash-Vergleich geprüft, ob auf der Festplatte wirklich etwas anders ist.
   No-op-Writes zählen nicht. Verlangt die Route Codeänderungen und es gibt keine,
   springt der Agent zurück in die Implementierung (`NEEDS_REVISION`).
3. **Reporter** berechnet den Endstatus ausschließlich aus Evidenz:
   keine echten Änderungen → niemals SUCCESS; Meldung "Keine echten Dateiänderungen erkannt".
4. Nach `max_verify_iterations` Versuchen: ehrlicher FAILED/PARTIAL-Report mit Diagnose.

## Coding-Agent-Test ausführen

```powershell
python -m pytest tests -q            # gesamte Suite
python -m pytest tests/test_agent_tools.py -q   # nur Agent-Härtungs-Tests
```
