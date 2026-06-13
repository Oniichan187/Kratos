# Agent-Verifikation & Verify-Gate

Wie Kratos sicherstellt, dass eine finale Antwort auf echten Fakten beruht und
nicht auf Modellbehauptungen. Alle Regeln sind in `tests/test_reporter_gate.py`
und `tests/test_diagnostics.py` abgesichert.

## Datenquellen (nur diese zählen)

| Fakt | Quelle | Datei |
|---|---|---|
| Geänderte Dateien | Hash/Inhalt vor vs. nach (Snapshots) | `reporter.verify_files_changed` |
| Ausgeführte Befehle | echte `CommandResult`-Dicts | `core/buildtest.py`, `execution/shell.py` |
| Testergebnis | letzter `is_test`-Befehl + Exitcode | `reporter._evaluate_tests` |
| Diff | `git diff --stat`, sonst Snapshot-Vergleich | `reporter._git_diff_stat` |
| Diagnose | geparster Befehls-Output | `execution/diagnostics.py` |

Der Reporter bekommt **niemals** freien Modelltext als Faktenquelle, nur die
strukturierte `ProvenWork`-Evidenz und die Snapshot-Map der Laufzeit.

## files_changed ist technisch bestimmt

Vor jeder Dateiänderung wird der Originalinhalt in `_original_snapshots`
gespeichert. Nach dem Lauf vergleicht `verify_files_changed` Festplatten-Inhalt
gegen Snapshot:

- `before is None, after vorhanden` → `created`
- `before vorhanden, after is None` → `deleted`
- `before == after` → `unchanged` (**No-op-Write zählt NICHT als Änderung**)
- sonst → `modified`

Nur `created|modified|deleted` gelten als echte Änderung.

## Der harte Verify-Gate (Statuslogik)

In `build_final_report`:

1. **Codeänderung verlangt, aber keine echte Änderung** → `FAILED`
   ("Keine echten Dateiänderungen erkannt"). Ein leerer Diff bei verlangter
   Codeänderung kann damit nie SUCCESS werden.
2. **Tests gelaufen UND bestanden UND Verifier akzeptiert** → `SUCCESS`.
3. Sonst (echte Änderung vorhanden) → `PARTIAL` mit Begründung:
   - Tests nicht ausgeführt → "Status kann nicht SUCCESS sein"
   - letzter Testlauf fehlgeschlagen
   - Verifier nicht bestätigt
4. Andernfalls → `FAILED`.

Konsequenzen (genau wie gefordert):
- Tests fehlgeschlagen → niemals SUCCESS.
- Tests nicht ausgeführt → kein SUCCESS (höchstens PARTIAL; SUCCESS nur, wenn
  nachvollziehbar keine Tests existieren und andere Verifikation erfolgreich war).
- Diff leer bei verlangter Codeänderung → kein SUCCESS.

## "Tests bestanden" ist evidenzgebunden

`_evaluate_tests` liefert `(tests_ran, tests_passed, detail)` ausschließlich aus
`proof.commands`. Gibt es keinen tatsächlich ausgeführten `is_test`-Befehl, ist
`tests_ran=False` und der Report schreibt "Ausgeführt: Nein" + Grund — nie
"Tests bestanden".

## Repair-Schleife: Diagnose statt rohem Traceback

Schlägt ein Befehl fehl, wird das Ergebnis durch `diagnose_command()` geschickt
(`execution/diagnostics.py`). Das Resultat ist eine konkrete Anweisung
(z. B. Zirkulärimport → "Cross-Import entfernen / Symbol in drittes Modul /
lazy import"), die in `verify_feedback` an den Coder fließt.

## Stall-Erkennung (Anti-94×-pytest)

`RepairTracker` zählt Fehler-Signaturen über Iterationen. Wiederholt sich
dieselbe Signatur ≥ `repair_stall_threshold` (Default 2) mal, wird eine
Eskalations-Notiz angehängt ("STALL WARNING … ändere den ANSATZ, nicht eine
Zeile") und als `failure_diagnosis`-Event geloggt. So werden nicht 94, sondern
wenige Iterationen für denselben Fehler verbraucht.

## Final-Report-Format

```
## Ergebnis            (SUCCESS/PARTIAL/FAILED)
## Geänderte Dateien   (echte Änderungen mit +/- Zeilen)
## Gefundene Probleme
## Ausgeführte Befehle (| Befehl | Shell | Exitcode | Ergebnis |)
## Tests               (Ausgeführt: Ja/Nein, Ergebnis, Grund)
## Diagnose des letzten Fehlers   (falls Tests fehlschlugen)
## Websuche/Webscraping (Verlangt / Durchgeführt / Quellen)
## Diff-Zusammenfassung
## Offene Einschränkungen
```

## Deterministic last-resort repair (when the model is hopeless)

Some failures recur no matter how good the diagnosis is, because the model is
simply too weak to act on it. Session 2026-06-13_15-43-37: a 4B abliterated
coder re-created the SAME circular import 15 times in a row (the stall tracker
counted all 15 and escalated, but the model still wrote
``from .cli import build_parser`` at top level every time).

For the one catastrophic pattern that is deterministically fixable — an
intra-package import cycle caused by a **provably-unused** cross-import — Kratos
now repairs it WITHOUT the model (``execution/circular.py``):

1. The diagnoser reports ``circular_import`` and the proof gate is unsatisfied.
2. ``break_unused_circular_imports(project_root)`` parses the package (AST),
   finds the 2-cycle, and deletes the ``from .other import X`` line whose every
   bound name is unused in that file.
3. The change flows through the normal snapshot/apply/re-verify machinery.

It never touches an import whose names are actually used (a genuine mutual
dependency the model must resolve) — removing a dead import is always safe.
This is what lets Kratos converge on the most common weak-model failure even
when the model itself cannot.
