# Fehleranalyse — Session 2026-06-13_00-57-15

Grundlage: die echte Kratos-Session-Datei `session_2026-06-13_00-57-15.jsonl`
(172.512 Log-Zeilen). Diese Analyse ist die Begründung für die Verbesserungen in
`execution/diagnostics.py`, `core/agent.py`, `verification.py` und `reporter.py`.

## 1. Welche Aufgabe Kratos lösen sollte

Repariere und erweitere eine kleine Python-CLI (`mini_agent_check`) im Repo
`starter_project`, die Wetterdaten aus lokalem HTML extrahiert: Parser reparieren
(`parse_weather_html`), CLI mit `--format json|csv` und `--out`, optionaler
URL-Support, kurze Web-Recherche, `python -m pytest` muss grün sein.

## 2. Welche Dateien Kratos gelesen hat

Der Projektindex erfasste 11 Dateien, u. a.
`mini_agent_check/{__main__,__init__,scraper,cli}.py`, `tests/test_scraper.py`,
`tests/test_cli.py`, `pyproject.toml`, `expected/expected_feldkirch.json`.

## 3. Welche Dateien Kratos geändert hat

Laut `final_report`: `mini_agent_check/scraper.py`, `mini_agent_check/cli.py`,
`docs/research_notes.md`. Die Änderungen waren real (file_change_evidence), aber
funktional falsch.

## 4. Welche Befehle Kratos ausgeführt hat

Nur ein einziger Verifikationsbefehl, dafür **94-mal**: `python -m pytest`.
Jeder Lauf endete mit **Exitcode 2**.

## 5. Welche Tests fehlschlugen

Alle. pytest brach schon beim *Collecting* ab (Exitcode 2 = Collection-Error,
kein einziger Test lief), über alle 10 Verifizierungs-Iterationen hinweg.

## 6. Welche Fehler in der Implementierung passiert sind

1. **Zirkulärer Import (Kern-Ursache).** `__init__.py` → `scraper.py` →
   `from .cli import WeatherCard`, und `cli.py` → `from .scraper import
   load_source, parse_weather_html`. Beim Import ist `scraper` erst teilweise
   initialisiert →
   `ImportError: cannot import name 'load_source' from partially initialized
   module 'mini_agent_check.scraper' (most likely due to a circular import)`.
2. **`parse_weather_html` war leer** — nur Docstring, kein Funktionskörper.
3. **Typfehler in `load_source`** — Parameter als `Path` deklariert, aber mit
   `str`-Methode `.startswith(...)` benutzt und mit `str`-Argument aufgerufen.
4. **`--out` falsch** — `args.out.mkdir(parents=True)` legte die Ausgabedatei
   selbst als Verzeichnis an, statt nur den Parent-Ordner.
5. **CSV unsauber** — manuell mit `",".join(...)` statt `csv.DictWriter`.

## 7. Falsche Erfolgsmeldung?

**Nein.** Der Reporter meldete ehrlich `PARTIAL` (`tests_passed=false`,
`verifier_accepted=false`). Die Anti-Fake-Success-Gates haben korrekt
gegriffen — das war nicht das Problem.

## 8. Wo der Planner versagt hat

Der Plan adressierte die Symptome ("Parser implementieren", "CLI erweitern"),
erkannte aber nie die strukturelle Ursache: zwei Module, die sich gegenseitig
auf Top-Level importieren. Der Import-Zyklus stand in keinem Plan-Schritt.

## 9. Wo der Coder versagt hat

Der Coder reagierte auf den rohen Traceback nur **kosmetisch**: er verschob die
Import-Zeile (Zeile 13 → 15 → 16 → 17) und fügte `# noqa: F401` hinzu, statt
den Zyklus zu brechen. Ein schwaches lokales Modell kann aus einem rohen
Traceback nicht ableiten "entferne den Cross-Import".

## 10. Wo der Verifier versagt hat

Der Verifier blockierte korrekt jede Iteration (`NEEDS_REVISION`), aber das
Feedback war 10× der identische rohe pytest-Output. Es gab **keine
Stall-Erkennung**: 94 identische Fehlschläge wurden nicht als "wir stecken
fest" erkannt, und das Feedback wurde nie eskaliert oder konkretisiert.

## 11. Harte Regeln, die Kratos daraus lernen muss

1. Rohe Tracebacks in **konkrete, imperative Fix-Anweisungen** übersetzen
   (z. B. Zirkulärimport → "Cross-Import entfernen / lazy import / Symbol in
   drittes Modul"), nicht nur den Output durchreichen.
2. **Identische Fehler zählen.** Wiederholt sich dieselbe Fehler-Signatur,
   eskalieren statt jede Iteration zu verbrennen.
3. Exitcode 2 bei pytest = **Collection-/Import-Fehler**, kein Assertion-Fehler.
   Das ist ein Import-Zeit-Problem und muss zuerst behoben werden.
4. Kosmetisches Umsortieren von Imports behebt keinen Zyklus — der Coder muss
   explizit darauf hingewiesen werden.

## Umsetzung der Lehren

- `execution/diagnostics.py`: `FailureDiagnoser` (Traceback → Diagnose +
  `fix_instruction`) und `RepairTracker` (Stall-Erkennung).
- `core/agent.py`: Diagnose + Stall-Eskalation werden in das
  `verify_feedback` der Repair-Schleife eingespeist und als
  `failure_diagnosis`-Event geloggt.
- `verification.py`: `_format_proven_work_feedback` hängt die Diagnose an.
- `reporter.py`: neuer Abschnitt "Diagnose des letzten Fehlers", Shell-Spalte,
  Websuche-„Verlangt“.

## Nachweis, dass die Diagnose korrekt ist

`starter_project` wurde nach der Diagnose repariert (Zyklus entfernt, Parser via
`html.parser` implementiert, `load_source`-Typ korrigiert, `--out` legt nur den
Parent an, CSV via `csv.DictWriter`). Ergebnis: `python -m pytest` → **7 passed**.
