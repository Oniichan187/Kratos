# Kratos — Full-Check-Analyse & Fixes (2026-06-19)

Analyse des Laufs in `coding_agent_light_fullcheck/fullcheck/.kratos/session_2026-06-19_13-04-14.jsonl`
und Behebung der dabei sichtbar gewordenen Fehler + der vier von dir gemeldeten Punkte.

---

## 1. Hat Kratos richtig gearbeitet? — Nein, knapp gescheitert

Der Lauf endete **nicht** mit SUCCESS. Stand der Tests am Ende des Logs: **5 passed / 4 failed**.
Aktueller Stand der Dateien auf der Platte: **8 passed / 1 failed**.

| Datei | Soll | Was Kratos tat | Ergebnis |
|---|---|---|---|
| `textutils.py` | `slugify` + `is_palindrome` | korrekt implementiert | ✅ |
| `numstats.py` | `mean`-Bug fixen + `median` | korrekt geschrieben — **aber erst NACH dem letzten Testlauf** | ✅ (ungetestet geblieben) |
| `mathutils.py` | `gcd` | **nie angefasst** | ❌ `NotImplementedError` |
| `legacy_helpers.py` | löschen | **nie gelöscht** | ❌ |

Kernproblem: Kratos hat `numstats.py` repariert, **danach aber nie wieder `pytest` ausgeführt** und
`gcd` / die Löschung nicht mehr erreicht. Der einzige Testlauf (13:08:26) lag *vor* dem numstats-Fix,
zeigte also noch `FFFF`. Anschließend lief sich der Coder in einer Suchschleife fest, bis du mit
Strg+C abgebrochen hast (`Coder interrupted` ×3).

---

## 2. Identifizierte Fehler im Agenten (aus dem Log)

**F1 — Endlos-Suchschleife mit 0 Treffern.**
Der Coder hat `### SEARCH: re.sub :: numstats.py` **5+ mal hintereinander** abgesetzt, jedes Mal
0 Treffer (`re.sub` gehört zu `slugify` in textutils, nicht zu numstats). Die vorhandene
Schleifen-Erkennung wird **pro Checklisten-Item zurückgesetzt** und hat die item-übergreifende
Wiederholung deshalb nie erwischt. → verbrannte Turns, Item-Budget aufgebraucht.

**F2 — Doppeltes, identisches Schreiben.**
`numstats.py` wurde **zweimal mit byte-identischem Inhalt** geschrieben (gleicher sha256
`4968b7a0`). Es gab keinen Schutz gegen No-op-Writes.

**F3 — Doppelte `write_file`-Anzeige (dein Punkt 3).**
Pro Schreibvorgang erschien `write_file('…')` zweimal: einmal aus dem Streaming-Filter beim
`### FILE:`-Marker (ohne Bytes) und einmal aus dem echten Tool-Event (mit Bytes/Delta).

**F4 — Live-Todo nicht live + zeigt mehrere Items (dein Punkt 1).**
Die untere Statusleiste rendert `pcompact[:60]` — die ganze, abgeschnittene Checkliste, die über
zwei Zeilen lief und auf `0/12` stehen blieb.

---

## 3. Durchgeführte Fixes

### Punkt 1 — Live-Todo zeigt nur noch das *aktive* Item
- `kratos/planning.py`: neue Funktion `active_checklist_line()` — liefert das erste **nicht
  erledigte** Item (`☑`/`☒` werden übersprungen).
- `kratos/ui/status.py`: die Leiste zeigt jetzt genau **eine** Zeile —
  `PLAN 3/12  ☐ <aktives Item>` — statt der abgeschnittenen Gesamtliste. Sind alle Items fertig:
  `✓ all items done`.
- `kratos/app/cli.py`: `plan_live["active"]` wird an allen drei Stellen (Planner-Ende + zwei
  `plan_status`-Updates) gesetzt. Da das aktive Item mit jedem erledigten Item weiterspringt, wirkt
  die Anzeige jetzt **live**.

### Punkt 3 — `write_file` wird nur noch einmal angezeigt
- `kratos/app/prompt_frame.py` und `kratos/app/tui.py`: der `### FILE:`/`### DELETE:`-Echo aus dem
  Streaming-Filter wurde entfernt. Maßgeblich ist jetzt allein das echte Tool-Event
  (`write_file('numstats.py') -> 404 bytes (-5 +0 lines)`) — „nur das zweite", wie gewünscht.

### Punkt 2 — Log-Fehler behoben
- **F1 / Suchschleife:** neue, **lauf-weite** (item-übergreifende) Dead-Search-Sperre in
  `kratos/roles/coder.py` (`_filter_dead_searches` / `_record_dead_searches`). Eine SEARCH/GREP, die
  schon einmal 0 Treffer lieferte, wird beim erneuten, identischen Aufruf **übersprungen** und durch
  einen Hinweis ersetzt („lies die Datei direkt / nimm einen anderen Begriff"). Aktiv in **beiden**
  Coder-Loops.
- **F2 / Doppel-Write:** No-op-Schutz in `do_write` (`kratos/execution/tools.py`). Ist der Inhalt
  byte-identisch zum aktuellen Datei-Inhalt auf der Platte, wird **nicht** neu geschrieben, sondern
  `write_file('…') -> no change (identical content already on disk)` gemeldet.

### Punkt 4 — Smart-Search arbeitet codex-näher
- `kratos/execution/search.py`: `glob_files()` akzeptiert jetzt **mehrere** Muster, mit Komma oder
  `|` ge-OR-t — wie `rg -g a -g b` bzw. `Get-ChildItem -Include *.py,*.md`.
  Beispiel: `### GLOB: **/*.py, *.md, README*`.
- `kratos/roles/coder.py` (`_EXTRA_MARKERS_DOC`): die Marker-Doku für das Modell beschreibt die
  Suche jetzt explizit codex-artig — Literal / `a|b` / Regex / Mehrfach-Globs / `::`-Scoping /
  `READ`(=Get-Content) / `READ_RANGE`. Regex (`### GREP:`, inkl. rg-Stil `-n "pat" path`) und
  Pipe-Alternation (`a|b`) waren bereits vorhanden und bleiben.

---

## 4. Verifikation

- Logik der neuen/geänderten Funktionen mit eigenständigen Tests geprüft — **alle bestanden**:
  `active_checklist_line` (8 Fälle), Multi-Pattern-`glob_files` (4), Dead-Search-Sperre (5),
  No-op-Write-Entscheidung (3).
- Jede Änderung einzeln auf Syntax durchgesehen; bestehende Tests statisch geprüft (keine ruft
  `do_write` direkt auf; der `glob_files`-Test nutzt ein Einzelmuster → unberührt).
- Hinweis: Der volle `pytest`-Lauf gegen die echten Dateien war in dieser Session nicht möglich,
  weil die Linux-Sandbox einen veralteten (größen-gecachten) Spiegel der bearbeiteten Dateien sieht.
  Die echten Dateien unter `C:\Tools\Kratos` sind korrekt geschrieben. Empfohlen: lokal einmal
  `python -m pytest kratos/tests` laufen lassen.

## 5. Geänderte Dateien
`kratos/planning.py`, `kratos/ui/status.py`, `kratos/app/cli.py`, `kratos/app/prompt_frame.py`,
`kratos/app/tui.py`, `kratos/execution/tools.py`, `kratos/execution/search.py`, `kratos/roles/coder.py`
