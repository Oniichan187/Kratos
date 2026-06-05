# Kratos — Local Coding Agent v2.0

Lokaler CLI-Coding-Agent mit zwei abliterierten Modellen: **Planner** (Analyse + Verifikation) und **Coder** (Implementierung). Läuft vollständig lokal via Ollama.

---

## Schnellstart

```powershell
pip install -r requirements.txt
kratos                      # aus beliebigem Projektverzeichnis
```

Kratos erkennt das aktuelle Verzeichnis automatisch als Projektkontext.

---

## Modelle

| Rolle | Modell | num_ctx | Aufgabe |
|---|---|---|---|
| **Planner** | `huihui_ai/qwen3-abliterated:8b` | 8192 | Analyse, Plan, Verifikation |
| **Coder** | `huihui_ai/qwen3.5-abliterated:4B` | 16384 | Code, Fixes, Refactoring |

---

## Pipeline

```
User Input
  → Input Analyzer       (Sprache, Follow-ups, Pfade, Stacktraces)
  → Intent Classifier    (22 Intents, regelbasiert, kein LLM)
  → Router               (8 Routen)
  → Context Builder      (Projekt scannen, Dateien ranken & laden)
  → Memory Retrieval     (session / task / project / long-term)

Coding-Tasks:
  Planner → Coder → Verifier
      ↑          ↑
      └──────────┘  (loop bis VERIFIED, UNSOLVABLE oder max_verify_iterations)
```

### Routen

| Route | Wann |
|---|---|
| `direct_answer` | Datei-/Code-Suche (kein LLM) |
| `planner_only` | Fragen, Erklärungen, Log-Analyse |
| `coder_only` | `mach weiter`, Git/Shell-Befehle |
| `planner_then_coder` | Alles andere: Coding, Docs, Config, UI, … |
| `diagnostic_loop` | Build/Test-Fehler + Retry |
| `ask_clarification` | Unklare Eingabe |

---

## Permissions

```
/permission low    → nur lesen
/permission mid    → lesen + schreiben  (default)
/permission high   → lesen + schreiben + löschen
```

Dateioperationen innerhalb des Projektordners laufen automatisch ohne Bestätigung.

---

## Logging

```
/logging on     → startet Session-Log  →  .kratos/session_YYYY-MM-DD_HH-MM-SS.jsonl
/logging off    → beendet Logging
/logging        → Status anzeigen
```

### Was geloggt wird

Das Log enthält ALLES — keine Kürzungen:

| Log-Typ | Inhalt |
|---|---|
| `user_input` | Vollständige Eingabe, Wortanzahl |
| `route_decision` | Intent + Route |
| `index_project` | Alle gescannten Dateien mit Größe und Priorität |
| `context_package` | Scope, Memory, geladene Dateien, **vollständiger Context-Prompt** |
| `model_input` | System-Prompt + vollständige Nachricht an Modell |
| `model_thinking` | Alle Chain-of-Thought Tokens |
| `model_output` | Vollständige Modellantwort |
| `verify_decision` | VERIFIED / NEEDS_REVISION / UNSOLVABLE + Feedback |
| `tool_call` | Jeder Tool-Aufruf mit Argumenten |
| `file_write` | Pfad + vollständiger Dateiinhalt (max 100 KB) |
| `file_delete` | Pfad |
| `build_test` | Befehl + vollständiger Output |
| `info/warn/error` | Systemmeldungen |

### Log-Format (JSONL)

```json
{"ts":"2026-06-05T14:23:01.123","type":"user_input","text":"...","word_count":42}
{"ts":"2026-06-05T14:23:01.124","type":"route_decision","intent":"coding","route":"planner_then_coder"}
{"ts":"2026-06-05T14:23:01.125","type":"index_project","project":"myapp","file_count":14,"files":[...]}
{"ts":"2026-06-05T14:23:01.200","type":"context_package","scope":"architecture","files_loaded":[...],"full_context_prompt":"..."}
{"ts":"2026-06-05T14:23:01.201","type":"model_input","role":"planner","system_prompt":"...","message":"..."}
{"ts":"2026-06-05T14:23:08.500","type":"model_thinking","role":"planner","text":"<think>..."}
{"ts":"2026-06-05T14:23:08.501","type":"model_output","role":"planner","text":"Plan: ..."}
{"ts":"2026-06-05T14:23:08.502","type":"model_input","role":"coder","message":"..."}
{"ts":"2026-06-05T14:23:15.000","type":"model_output","role":"coder","text":"### FILE: ..."}
{"ts":"2026-06-05T14:23:15.100","type":"model_input","role":"verifier","message":"..."}
{"ts":"2026-06-05T14:23:17.000","type":"verify_decision","decision":"VERIFIED","feedback":"","iteration":1}
{"ts":"2026-06-05T14:23:17.100","type":"file_write","path":"main.py","content":"..."}
```

---

## Verify-Loop

```
Planner → Coder → Verifier
              ↑
       NEEDS_REVISION: <feedback>
              |
         re-plan + re-code
              ↓
         Verifier
              ↑
          VERIFIED → fertig
          UNSOLVABLE → abbruch mit Erklärung
          (safety cap: max_verify_iterations, default 10)
```

Verifier-Ausgaben:
- `VERIFIED` — Implementierung vollständig und korrekt
- `NEEDS_REVISION:` — konkrete Fehler, wird an Planner zurückgegeben
- `UNSOLVABLE:` — Aufgabe kann nicht gelöst werden (widersprüchliche Anforderungen, fehlende Abhängigkeiten)

---

## Slash-Commands

| Befehl | Beschreibung |
|---|---|
| `/permission [low\|mid\|high]` | Coder-Berechtigungen |
| `/logging [on\|off]` | Session-Logging |
| `/index` | Projektdateien anzeigen |
| `/index rebuild` | Index neu aufbauen |
| `/memory list` | Memory-Einträge anzeigen |
| `/memory clear [session\|project\|all]` | Memory löschen |
| `/build [cmd]` | Build-Befehl setzen |
| `/test [cmd]` | Test-Befehl setzen |
| `/models [planner\|coder <name>]` | Modelle wechseln |
| `/goal [text]` | Ziel setzen |
| `/scope [global\|project]` | Config-Scope wechseln |
| `/history clear` | Konversation zurücksetzen |
| `/status` | Status-Bar anzeigen |
| `/help` | Alle Befehle |
| `/exit` | Beenden |

---

## Architektur

```
C:\Tools\Kratos\
├── kratos.py              ← Einstieg: REPL, _stream_agent, _apply_file_ops
├── kratos.bat             ← Globaler Launcher (C:\Tools\Kratos im PATH)
├── requirements.txt
└── kratos/
    ├── analyzer.py        ← InputAnalyzer: Sprache, Follow-ups, Artefakte
    ├── classifier.py      ← IntentClassifier: 22 Intents, regelbasiert
    ├── router.py          ← Router: Intent → Route
    ├── context.py         ← ProjectIndexer + ContextBuilder
    ├── memory.py          ← MemoryManager: session/task/project/longterm
    ├── agent.py           ← KratosAgent: process(), _run_planner/coder/verifier
    ├── bridge.py          ← OllamaBridge: HTTP-Streaming, Fehlerbehandlung
    ├── config.py          ← KratosConfig: globale + projektspezifische Konfig
    ├── logger.py          ← SessionLogger: vollständiges JSONL-Logging
    ├── commands.py        ← Slash-Command-Handler
    └── ui.py              ← Rich UI: Banner, Headers, Tool-Calls, Permissions
```

### Datenfluss

```
kratos.py  main()
  └── REPL loop
        └── _stream_agent(agent, task, logger)
              ├── logger.log_input(task)
              └── agent.process(task)  →  Generator[(source, content, kind)]
                    ├── ("tool", "index_project(...)") → tool_call() + logger
                    ├── ("tool", "read_file(...)") → tool_call() + logger
                    ├── ("log", json_data) → logger._write()
                    ├── ("header", "planner") → planner_header()
                    ├── ("planner", token, "think"|"text") → console
                    ├── ("end", "planner") → section_end()
                    ├── ("header", "coder") → coder_header()
                    ├── ("coder", token, "think"|"text") → _CoderFilter
                    ├── ("end", "coder") → section_end()
                    ├── ("header", "verify") → verify_header()
                    ├── ("verify", token, "think"|"text") → console
                    └── ("end", "verify") → section_end()
              └── _apply_file_ops(agent, root, logger)
                    ├── write_file → tool_call() + logger.log_file_write()
                    └── delete_file → tool_call() + logger.log_file_delete()
```

---

## Hardware & Voraussetzungen

| Komponente | Anforderung | Getestet |
|---|---|---|
| GPU | NVIDIA CUDA, mind. 4 GB VRAM | RTX 4050 Laptop 6 GB |
| CPU | x86-64, 8+ Kerne | i7-13800H 14 Kerne |
| RAM | mind. 16 GB | 16 GB |
| OS | Windows 10/11 | Windows 11 Pro |
| Python | 3.10+ | 3.12 |
| Ollama | Nativ auf Windows | aktuell |

Planner und Coder werden **sequenziell** geladen — nie gleichzeitig im VRAM.

---

## Konfiguration

### `.kratos/config.json` (pro Projekt, CWD-relativ)

```json
{
  "planner_model": "huihui_ai/qwen3-abliterated:8b",
  "coder_model": "huihui_ai/qwen3.5-abliterated:4B",
  "scope": "project",
  "permission": "mid",
  "planner_num_ctx": 8192,
  "coder_num_ctx": 16384,
  "planner_temp": 0.7,
  "coder_temp": 0.2,
  "max_verify_iterations": 10,
  "build_cmd": null,
  "test_cmd": null,
  "build_test_retries": 3
}
```

### `~/.kratos/config.json` (global, maschinenweite Defaults)

Gleiche Felder. Projektkonfig überschreibt globale Konfig.
