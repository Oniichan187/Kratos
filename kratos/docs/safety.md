# Sicherheit — blockierte Befehle & Pfadschutz

Zentrale Stelle: `safety.py`. Jede Shell-ausführende Oberfläche
(`ShellRunner`, Verifikations-Runner, Inspect-Runner) ruft vor der Ausführung
`check_command` auf; jede dateischreibende Oberfläche ruft `check_path`.
Abgesichert in `tests/test_safety_shell.py`.

## Blockierte Befehle (`check_command` / `is_dangerous_command`)

Windows-first, Patterns case-insensitive. Blockiert == wird **nie** ausgeführt
(Result mit `blocked=True`, Exitcode 126).

**Datenträger/Disk**
- `format <laufwerk>:`, `mkfs`, `dd of=/dev/...`, `diskpart`

**Rekursives/erzwungenes Löschen**
- `del /s /q`, `rd /s`, `rmdir /s`
- `rm -rf`, `rm -rf /` bzw. `~`
- `Remove-Item -Recurse` / `-Force` (ohne `-WhatIf`)

**Systemzustand**
- `shutdown`, `restart-computer`, `stop-computer`, `reboot`
- `reg delete|add`, `Remove-ItemProperty HKLM/HKCU` (Registry)
- `schtasks`, `New-ScheduledTask` (Persistenz)
- `net user`, `net localgroup`
- `bcdedit`, `vssadmin delete`

**Download-and-Execute / Shell-Escape**
- `Invoke-Expression` / `iex`
- `DownloadString(` / `DownloadFile(`
- `curl|wget|iwr|Invoke-WebRequest|Invoke-RestMethod|irm ... | sh|bash|powershell|pwsh|iex|python|cmd`
- `Start-Process ... -WindowStyle Hidden` / `-Verb RunAs`
- `-EncodedCommand`, `FromBase64String`

**Credentials/Exfiltration**
- `mimikatz`, `lsass`, `sekurlsa`
- `Get-Credential`, `cmdkey /list`
- Zugriff auf `*token*/*secret*/*password*/*api_key*`-Env-Variablen

**Rechte/Eigentum & Müll**
- `takeown`, `icacls ... grant`
- `sudo rm`, `chmod 777 /`
- Fork-Bomb `:(){ :|:& };:`

## Pfadschutz (`check_path`)

- Ziel muss **innerhalb** des Projektwurzelverzeichnisses bleiben
  (`../../etc/passwd` und absolute Pfade nach außen → blockiert).
- Git-Interna sind tabu: `.git/objects`, `.git/refs`, `.git/HEAD`.

## Design-Prinzipien

- Unbekanntes Kommando == nicht *explizit* gefährlich; zusätzliche Allowlists
  (Verify-Commands) greifen separat.
- `check_command` wirft nie — robuste Eingabebehandlung.
- Defense-in-depth: auch erlaubte Build/Test-Commands passieren vor der
  Ausführung erneut den Blocklist-Check (`core/buildtest.py`).
