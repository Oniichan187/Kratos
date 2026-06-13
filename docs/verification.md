# Kratos Verifikation — wie Erfolg bewiesen wird

## Der Plan-Verify-Loop

```
PLAN → READ → IMPLEMENT → VERIFY → (REPAIR → VERIFY)* → FINAL REPORT
```

Pro Iteration (`max_verify_iterations`, Default 10) gilt:

1. **File-Application-Gate** — fehlgeschlagene Writes ⇒ sofortiger Retry.
2. **Real-File-Change-Gate** — `verify_files_changed()` vergleicht jede behauptete
   Datei per SHA-256 mit ihrem Vor-Lauf-Snapshot:
   - `created` / `modified` / `deleted` = echte Änderung
   - `unchanged` (No-op-Rewrite) und `missing` zählen **nicht**
   - Route verlangt Codeänderungen + keine echte Änderung ⇒ `NEEDS_REVISION`
     mit Feedback „Keine echten Dateiänderungen erkannt“ → zurück in die Implementierung.
     Nach Ausschöpfen der Retries ⇒ FAILED.
3. **ProvenWork-Gate** — Befehls-Evidenz: mindestens ein echter Testbefehl mit
   Exitcode 0 (konfigurierbar über `require_proven_work`, `require_test_for_verified`).
4. **LLM-Verifier** — strenge VERIFIED/NEEDS_REVISION/UNSOLVABLE-Entscheidung,
   gated durch die Gates davor.
5. **Final-Sweep** — nach VERIFIED laufen alle Verify-Befehle noch einmal komplett;
   ein Fehlschlag degradiert zurück zu NEEDS_REVISION.

## Der Final-Report (Reporter)

Der Report wird ausschließlich aus strukturierter Evidenz gebaut
(`ProvenWork` + Snapshot-Abgleich + `git diff --stat`):

- **Status SUCCESS** nur wenn: echte Dateiänderungen ∧ Testbefehl real gelaufen ∧
  Exitcode 0 ∧ Verifier hat akzeptiert.
- **Keine Dateien geändert** ⇒ Status FAILED (bei Code-Aufgaben) und die Meldung
  „Keine echten Dateiänderungen erkannt".
- **Tests nicht ausgeführt** ⇒ niemals „Tests bestanden", sondern
  „Tests nicht ausgeführt" mit Grund.
- **Diff** stammt aus `git diff --stat` oder dem Hash-Vergleich — ohne echte
  Änderungen steht dort „Kein Diff vorhanden." Ein Diff wird nie erfunden.

## Auto-Verify (Runtime beweist Edits selbst)

Wenn das Modell Dateien schreibt, aber kein `### VERIFY` ausgibt (häufigster
Schwach-Modell-Fehler), führt der Work-Step-Driver selbst eine Verifikation aus:
zuerst das `VERIFY:`-Kommando des Checklist-Items (falls sicher), sonst den
ersten Test-Befehl der CommandRegistry. Ergebnis fließt als Evidenz in
ProvenWork und als Observation zurück ans Modell.

Befehls-Discovery unterstützt verschachtelte Layouts: liegt `pyproject.toml`
einen Ordner tiefer (z. B. `starter_project/`), wird `python -m pytest` mit
diesem Unterordner als Working Directory ausgeführt (`VerificationCommand.cwd`).

## Endlosschleifen-Schutz

- `max_verify_iterations` (äußerer Loop), `max_coder_iterations` (ReAct-Loop),
  `max_work_step_turns` (Micro-Turns pro Checklist-Item, Default 4),
  2-Strikes-Regel für marker-lose Turns.
- Danach: ehrlicher Abbruch mit Diagnose im Report — niemals stilles "Erfolg".
