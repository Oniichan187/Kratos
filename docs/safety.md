# Kratos Safety-Modell

## Grundsätze

1. Dateioperationen nur innerhalb des Projektverzeichnisses (Resolve + `relative_to`-Check).
2. Git-Interna (`.git/objects`, `.git/refs`, `.git/HEAD`) sind schreibgeschützt.
3. Shell-Befehle laufen durch zwei Schichten:
   - **Allowlist** (`_is_safe_verification_command` / `_is_safe_inspect_command`):
     nur bekannte Build/Test/Diagnose-Präfixe werden überhaupt ausgeführt.
   - **Blocklist** (`safety.check_command`): zusätzlich werden destruktive Muster
     hart blockiert — auch falls die Allowlist je erweitert wird (Defense in Depth).
4. Löschungen nur mit `permission high`; jede Löschung wird geloggt.
5. Secrets: Kratos sucht nicht nach Tokens/Passwörtern; Secret-Dateien
   (`.env`, `secrets.*`, `*.pem`, `*.key`, `*.pfx`, `*.p12`) sind vom Index ausgeschlossen
   und werden nie in Prompts oder Logs geladen.

## Blockierte Befehlsklassen (Auszug)

- Laufwerk/Datenträger: `format`, `mkfs`, `diskpart`, `dd of=/dev/...`
- Rekursives Löschen: `del /s /q`, `rd /s`, `rm -rf`, `Remove-Item -Recurse/-Force`
- Systemzustand: `shutdown`, `Restart-Computer`, `reg add/delete`, `schtasks`,
  `net user/localgroup`, `bcdedit`, `vssadmin delete`
- Download-and-Execute: `Invoke-Expression`/`iex`, `DownloadString(`,
  `curl/wget/iwr ... | sh/bash/powershell/iex`, `-EncodedCommand`, `FromBase64String`
- Credentials/Exfiltration: `mimikatz`, `lsass`, `Get-Credential`, `cmdkey /list`,
  Zugriff auf `$env:*TOKEN/SECRET/PASSWORD/API_KEY*`
- Privilegien: `takeown`, `icacls ... grant`, `sudo rm`, `chmod 777 /`

Blockierte Befehle werden **nie ausgeführt**; das Ergebnis ist ein `CommandResult`
mit `blocked=True`, Exitcode 126 und dem Grund in `stderr` — sichtbar im Log und
in der Observation des Modells.

## Web

- Nur `http`/`https`; private/Loopback-Literale (localhost, 127.0.0.1, RFC-1918) werden verweigert.
- Antworten sind Daten, werden nie ausgeführt; 2-MB-Cap; Content-Type-Prüfung.
- Kein Shell-Kommando lädt jemals Webseiten (kein curl/wget für Inhalte).
