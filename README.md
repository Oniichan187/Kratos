# Kratos — Local Abliterated Max-Context 4-Role CLI Agent

**Kratos** — fully local, all models **abliterated** (no safety filters).

4 roles that play perfectly together:
- **Planner** — full max ctx (40k) every time
- **Coder** — full 262k ctx, walks the plan **step by step**: think how to implement, show verify command, code it, runtime immediately tests that step, then next step
- **Verifier** — full max ctx, **really executes every test** (per-step + final sweep) and only accepts VERIFIED on solid PROVEN_WORK
- **Auto-Composer (Compressor)** — full ctx, **never destroys information**, feeds durable facts into `.kratos/memory.json` (project + global)

Works for tiny tasks **and** for huge monorepos that massively exceed any ctx window (Large-Input Relay via coder + lossless compress + memory).

GitHub repo: https://github.com/Oniichan187/Kratos

All via Ollama, Windows native, sequential loading (laptop friendly, 4-8 GB VRAM).

---

## Schnellstart

```powershell
pip install -r requirements.txt
python setup_models.py      # einmalig: Modelle einrichten
kratos                      # aus beliebigem Projektverzeichnis starten
```

---

## Modelle (alle abliterated — keine Safety-Filter)

Kratos nutzt bei **jedem** Aufruf das **maximale Kontextfenster** des jeweiligen Modells (innerhalb des VRAM-Ceilings). Keine "kleine Prompts = kleines Fenster".

| Rolle | Modell (abliterated) | max ctx | Aufgabe |
|---|---|---|---|
| **Planner** | `huihui_ai/qwen3-abliterated:8b` | 40 960 | Analyse, detaillierter Plan mit verifizierbaren Schritten |
| **Coder** | `huihui_ai/qwen3.5-abliterated:4b` | 262 144 | Implementiert **einen Plan-Schritt nach dem anderen**: nachdenken, Befehl anzeigen, umsetzen, Test → nächster Schritt |
| **Verifier** | `huihui_ai/qwen3-abliterated:8b` | 40 960 | Führt wirklich alle Tests aus (pro Schritt + finaler Sweep), prüft PROVEN_WORK streng |
| **Auto-Composer** (Compressor) | `kratos-planner` (Phi-4-mini-abliterated GGUF) | ~32k+ | Verlustfreie History-Kompression + Memory-Extraktion in `.kratos/memory.json` (keine Info wird zerstört) |

Alle Modelle laufen **sequenziell** — nie gleichzeitig im VRAM. Optimiert für Laptops (RTX 4050 6 GB Klasse).

---

## Pipeline (max-ctx + stepwise + lossless)

```
User Input
  → Analyzer + Classifier (regel-basiert) → Router
  → Context (mit Memory aus .kratos) + ggf. Large-Input Relay (Coder 262k vor Planner)

  Planner (full 40k ctx) → detaillierter Plan mit NUMMERIERTEN Steps + Verify-Cmds

  Coder (full 262k) — für JEDEN Step:
      - denkt "wie genau umsetzen + Risiken + welcher Befehl zum Testen?"
      - zeigt/implementiert den Code für genau diesen Step
      - Kratos schreibt Datei(en)
      - Kratos führt den Verify-Befehl (Test) für diesen Step aus
      → erst dann nächster Step

  Nach allen Steps:
      - finale Test-Sweep
      - Verifier (full 40k) bekommt ALLE per-step PROVEN_WORK Beweise
      - nur bei echten exit=0 auf allen relevanten Tests + LLM "VERIFIED" → akzeptiert

  Auto-Compress (Compressor full ctx) + Memory-Extraktion
      → .kratos/memory.json (project) + global
      → niemals Infos vernichten (exhaustive + verbatim quotes)
```

### Routen

| Route | Wann |
|---|---|
| `direct_answer` | Datei-/Code-Suche (kein LLM) |
| `planner_only` | Fragen, Erklärungen, Analyse |
| `coder_only` | `mach weiter`, Git-Befehle |
| `planner_then_coder` | Alle Coding-Aufgaben |
| `diagnostic_loop` | Build/Test-Fehler + Retry |
| `ask_clarification` | Unklare Eingabe |

---

## Dynamisches Reasoning

Der Planner aktiviert Chain-of-Thought (`think`) nur wenn der Task es wirklich braucht:
- Retry (vorherige Iteration fehlgeschlagen)
- Architecture/Diagnostic-Scope
- Viele relevante Dateien (>5)
- Langer Task (>40 Wörter)
- Diagnostic-Route

Für einfache Aufgaben: direkte Ausgabe, kein CoT → spart VRAM-Zeit.

---

## Auto-Composer (Compressor) — verlustfrei + .kratos Memory

Immer mit **maximalem Kontextfenster**. Prompts erzwingen Vollständigkeit (exhaustive + wörtliche Zitate kritischer Fakten).

- History-Kompression: alte Turns werden durch dichte, aber **informations-erhaltende** Records ersetzt.
- Nach jedem Task: Memory-Extraktion (decisions, conventions, file_roles, error_cause, solution) → `.kratos/memory.json` (project) + global.
- Wird in jeden relevanten Prompt eingespeist.
- Nie Infos vernichten — das ist eine Kernanforderung.

`/memory list | clear ...` verwaltet es.

---

## Large-Input Relay (für Kontexte die jedes Fenster sprengen)

Wenn der Planner-Input > ~80% von planner_num_ctx:
1. Der **Coder** (mit 262k full ctx) bekommt den riesigen rohen Kontext zuerst.
2. Erzeugt einen verlustarmen, strukturierten Extrakt (auch hier: max-ctx + strenger Prompt).
3. Der Extrakt (viel kleiner) geht an den Planner.

Zusammen mit Memory + Auto-Composer + ContextBuilder (der alle Datei-Pfade immer zeigt) kann Kratos an **riesigen** Repos arbeiten, die das Kontextfenster bei weitem übersteigen.

---

## Token-Budget — Maximum Context Policy

- **Immer Maximum**: `choose_num_ctx(..., force_max_context=True)` → jedes Modell bekommt bei jedem Aufruf sein volles Fenster (capped nur durch `vram_ctx_ceiling`).
- Defaults jetzt direkt auf den Modell-Maxima (Planner 40960, Coder 262144, Verifier 40960, Compressor 32k+, Relay 128k).
- `always_max_ctx: true` in config (auch alte Configs werden beim Laden hochgezogen).
- VRAM-Ceiling weiterhin respektierend (laptop-sicher), aber so hoch wie möglich.
- `/tokens` zeigt realen Verbrauch (von Ollama).

Das ermöglicht kleine schnelle Tasks **und** die Monster-Repos.

---

## Stepwise Coder + Verify-Loop (der Kern)

Coder führt den Plan **Schritt für Schritt** aus (genau wie vom User gewünscht):

1. Planner liefert nummerierte Steps mit Verify-Befehlen.
2. Für Step N:
   - Coder denkt: "Wie setze ich das exakt um? Risiken? Welchen Befehl zeige/empfehle ich zum Test?"
   - Coder gibt die Dateiänderung(en) für **nur diesen Step** aus.
   - Kratos schreibt die Dateien + verifiziert Hash.
   - Kratos führt den (vom Coder oder Projekt empfohlenen) Test/Befehl **sofort** aus.
   - Nur wenn ok → Step N+1.
3. Nach allen Steps: finale volle Test-Suite + Verifier-LLM.
4. Verifier darf **nur** VERIFIED sagen, wenn für (idealerweise jeden) Step echte Tests mit exit=0 nach dem Write gelaufen sind + die finale Evidenz passt.

PROVEN_WORK ist jetzt noch strenger:
- Per-Step Commands werden aufgezeichnet (mit "step": N).
- Finaler Sweep nach "VERIFIED" des LLMs wird zusätzlich ausgeführt.
- Fehlt ein Test oder schlägt einer fehl → NEEDS_REVISION (auch wenn LLM schon Verfied sagte).

UNSOLVABLE → kompletter Rollback der in dieser Runde geschriebenen Dateien.

Wenn keine Tests auto-entdeckt werden:

```powershell
/test python -m pytest tests -q --tb=line
```

---

## Permissions

```
/permission low    → nur lesen
/permission mid    → lesen + schreiben  (default)
/permission high   → lesen + schreiben + löschen
```

---

## Logging

```
/logging on     → startet Session-Log  →  .kratos/session_YYYY-MM-DD_HH-MM-SS.jsonl
/logging off    → beendet Logging
```

---

## Slash-Commands

| Befehl | Beschreibung |
|---|---|
| `/permission [low\|mid\|high]` | Coder-Berechtigungen |
| `/tokens` | Session Token-Verbrauch anzeigen |
| `/logging [on\|off]` | Session-Logging |
| `/index` | Projektdateien anzeigen |
| `/index rebuild` | Index neu aufbauen |
| `/memory list` | Memory-Einträge anzeigen |
| `/memory clear [session\|project\|all]` | Memory löschen |
| `/build [cmd]` | Build-Befehl setzen |
| `/test [cmd]` | Test-Befehl setzen |
| `/models [planner\|coder\|verifier\|compressor <name>]` | Modelle wechseln (bleiben abliterated + max-ctx) |
| `/goal [text]` | Ziel setzen |
| `/scope [global\|project]` | Config-Scope wechseln |
| `/history clear` | Konversation zurücksetzen |
| `/status` | Status-Bar anzeigen |
| `/help` | Alle Befehle |
| `/exit` | Beenden |

---

## Prompt Customization (JSON — der Schlüssel zum "besten Agenten")

Alle System-Prompts + Snippets (Labels, Forced-Instructions, Marker, Predict-Limits) liegen in JSON. 
Der **gesamte Prompt-Flow** (Zusammenbau, bedingte Sections, Memory/Proof/Context-Injection, Stepwise-per-Plan-Item, Relay für Huge-Repos, PROVEN_WORK etc.) bleibt **vollständig ausprogrammiert** (dynamisch in Python).

- Defaults sind im Package (kratos/prompts.py).
- Overrides (Merge, partial OK):
  - `~/.kratos/prompts.json` (global)
  - `.kratos/prompts.json` (project, gewinnt)
- Einfach editierbar für Tuning der "besten" Verhaltensregeln (step-by-step Discipline, lossless Memory etc.), ohne Python zu ändern.

Beispiel `.kratos/prompts.json` (nur was du ändern willst):
```json
{
  "coder_system": "You are Kratos Coder. ... (deine angepasste Version mit extra rules) ...",
  "snippets": {
    "test_files_header": "TEST FILES — EXACT SIGNATURES REQUIRED:",
    "coder_step_forced_prefix": "CRITICAL: ONLY THIS STEP. Begin with ### FILE: ..."
  }
}
```

Befehle:
- `/prompts list` — Übersicht (Rollen + Snippets)
- `/prompts reload` — nach Edit neu laden (nächste Calls nutzen es)
- `/prompts dump .kratos/prompts.json` — Defaults rausschreiben zum Start-Edit

Damit ist Kratos extrem anpassbar, während die starke programmierte Logik (jeden Plan-Schritt denken+Cmd zeigen+umsetzen+sofort testen, Verifier führt Tests wirklich aus, Compressor zerstört keine Info, .kratos Memory, max-ctx, huge+small Projekte) erhalten bleibt.

## Konfiguration

### `.kratos/config.json`

```json
{
  "planner_model":     "huihui_ai/qwen3-abliterated:8b",
  "coder_model":       "huihui_ai/qwen3.5-abliterated:4b",
  "verifier_model":    "huihui_ai/qwen3-abliterated:8b",
  "compressor_model":  "kratos-planner",
  "planner_num_ctx":   40960,
  "coder_num_ctx":     262144,
  "verifier_num_ctx":  40960,
  "compressor_num_ctx": 32768,
  "relay_num_ctx":     131072,
  "vram_ctx_ceiling":  65536,
  "always_max_ctx":    true,
  "compress_threshold": 0.75,
  "relay_threshold":   0.80,
  "max_history_pairs": 8,
  "auto_compress":     true,
  "permission":        "mid",
  "max_verify_iterations": 10,
  "auto_discover_verification": true,
  "require_proven_work": true,
  "require_test_for_verified": true,
  "verification_timeout_seconds": 120
}
```

---

## Architektur

```
C:\Tools\Kratos\
├── kratos.py              ← REPL, _stream_agent, _show_file_ops
├── kratos.bat             ← Globaler Launcher
├── requirements.txt
├── setup_models.py        ← Modell-Setup-Wizard (idempotent)
├── setup_wsl.sh           ← WSL + CUDA Setup (einmalig)
├── tests/
│   └── test_core.py       ← 44 Unit-Tests (kein Ollama nötig)
└── kratos/
    ├── tokens.py          ← choose_num_ctx with force_max_context=True (default)
    ├── compress.py        ← Auto-Composer: lossless history + .kratos memory (max ctx)
    ├── agent.py           ← KratosAgent: Planner→Stepwise-Coder+per-step-tests→Verifier loop
    ├── bridge.py          ← OllamaBridge (full num_ctx always passed)
    ├── config.py          ← always_max_ctx + bumped role num_ctx defaults + auto-upgrade on load
    ├── context.py         ← full file listing + token-aware excerpts (benefits from huge budgets)
    ├── memory.py          ← 4-tier (.kratos/memory.json project + global)
    ... (analyzer, classifier, router, commands, logger, ui)
```

---

## Hardware & Voraussetzungen

| Komponente | Anforderung |
|---|---|
| GPU | NVIDIA CUDA, mind. 4 GB VRAM |
| RAM | mind. 16 GB |
| OS | Windows 10/11 |
| Python | 3.10+ |
| Ollama | nativ auf Windows |

Planner, Coder und Compressor werden **sequenziell** geladen — nie gleichzeitig im VRAM.
