# Kratos — Local Abliterated Dual-Model CLI Agent

Lokaler CLI-Coding-Agent mit drei abliterierten Modellen: **Planner**, **Coder** und **Compressor**.
Läuft vollständig lokal via Ollama. Keine Cloud, keine Safety-Filter.

---

## Schnellstart

```powershell
pip install -r requirements.txt
python setup_models.py      # einmalig: Modelle einrichten
kratos                      # aus beliebigem Projektverzeichnis starten
```

---

## Modelle (alle abliterated)

| Rolle | Modell | max ctx | Aufgabe |
|---|---|---|---|
| **Planner** | `huihui_ai/qwen3-abliterated:8b` | 40 960 | Analyse, Plan, Verifikation |
| **Coder + Relay** | `huihui_ai/qwen3.5-abliterated:4b` | 262 144 | Code, Fixes, Refactoring; Relay für große Inputs |
| **Compressor** | `kratos-planner` (Phi-4-mini-abliterated) | 16 384 | History-Kompression, Memory-Extraktion |

Alle Modelle laufen **sequenziell** — nie gleichzeitig im VRAM. VRAM-sicher auf 4–6 GB.

---

## Pipeline

```
User Input
  → Input Analyzer       (Sprache, Follow-ups, Pfade, Stacktraces)
  → Intent Classifier    (22 Intents, regelbasiert, kein LLM)
  → Router               (8 Routen)
  → Context Builder      (token-bewusst, alle Projektgrößen)

  [Large-Input Relay]    (wenn Input > relay_threshold × planner_ctx)
  → Coder (relay mode)   → kompakter Extrakt → Planner

  Planner → Coder → Verifier
      ↑           ↑
      └───────────┘  (loop bis VERIFIED, UNSOLVABLE oder max_verify_iterations)

  Auto-Compress          (wenn History > compress_threshold × num_ctx)
  → Compressor           → semantische Zusammenfassung → History ersetzt
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

## Auto-Kompression

Wenn die geschätzte Prompt-Größe `compress_threshold × num_ctx` überschreitet:
1. **Compressor** (Phi-4-mini-abliterated, 16K) fasst die ältesten History-Paare zusammen
2. Semantische Zusammenfassung ersetzt die gelöschten Paare → kein Informationsverlust
3. Algo-Fallback wenn Modell nicht verfügbar

Nach jedem erfolgreichen Task extrahiert der Compressor generische Memory-Einträge
(Entscheidungen, Konventionen, Datei-Rollen) → gespeichert in `.kratos/memory.json`.

---

## Large-Input Relay

Wenn ein Projekt-Kontext die Planner-Kapazität (40K) überschreiten würde:
1. **Coder** (262K Kontext) verarbeitet den großen Input zuerst
2. Produziert einen kompakten strukturierten Extrakt
3. Extrakt geht an den **Planner** → kein Overflow

Ermöglicht Arbeit an sehr großen Projekten (1000+ Dateien, riesige Logs).

---

## Token-Budget

- `num_ctx` wird **dynamisch** gewählt: `min(model_max, vram_ceiling, (prompt + output) × 1.3)`
- Planner standard: 12 288 (von max 40 960)
- Coder standard: 24 576 (von max 262 144)
- Relay-Modus: 32 768
- Compressor: 8 192
- Alle Grenzen per Config überschreibbar

Token-Verbrauch wird nach jedem Task angezeigt und ist via `/tokens` abrufbar.

---

## Verify-Loop

```
Planner → Coder → Verifier
              ↑
    NEEDS_REVISION: <Feedback>
              |
         Re-plan + Re-code
              ↓
         Verifier
         VERIFIED → fertig
         UNSOLVABLE → Rollback aller geschriebenen Dateien + Abbruch
         (Safety Cap: max_verify_iterations, default 10)
```

Bei `UNSOLVABLE` werden alle in dieser Runde geschriebenen Dateien auf den Originalzustand zurückgesetzt.

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
| `/models [planner\|coder\|compressor <name>]` | Modelle wechseln |
| `/goal [text]` | Ziel setzen |
| `/scope [global\|project]` | Config-Scope wechseln |
| `/history clear` | Konversation zurücksetzen |
| `/status` | Status-Bar anzeigen |
| `/help` | Alle Befehle |
| `/exit` | Beenden |

---

## Konfiguration

### `.kratos/config.json`

```json
{
  "planner_model":     "huihui_ai/qwen3-abliterated:8b",
  "coder_model":       "huihui_ai/qwen3.5-abliterated:4b",
  "compressor_model":  "kratos-planner",
  "planner_num_ctx":   12288,
  "coder_num_ctx":     24576,
  "compressor_num_ctx": 8192,
  "relay_num_ctx":     32768,
  "vram_ctx_ceiling":  32768,
  "compress_threshold": 0.75,
  "relay_threshold":   0.80,
  "max_history_pairs": 8,
  "auto_compress":     true,
  "permission":        "mid",
  "max_verify_iterations": 10
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
    ├── tokens.py          ← TokenEstimator, choose_num_ctx, relay_needed
    ├── compress.py        ← Compressor (history, memory, relay)
    ├── agent.py           ← KratosAgent: Hauptpipeline
    ├── bridge.py          ← OllamaBridge: HTTP-Streaming + Usage-Tracking
    ├── config.py          ← KratosConfig: global + projektspezifisch
    ├── context.py         ← ProjectIndexer + ContextBuilder (token-aware)
    ├── memory.py          ← MemoryManager: 4 Tier, Secret-Filter
    ├── analyzer.py        ← InputAnalyzer
    ├── classifier.py      ← IntentClassifier: 22 Intents, regelbasiert
    ├── router.py          ← Router: Intent → Route
    ├── commands.py        ← Slash-Command-Handler
    ├── logger.py          ← SessionLogger: JSONL
    └── ui.py              ← Rich UI
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
