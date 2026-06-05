"""Core unit tests — no Ollama required.

Run:  python -m pytest tests/ -v
or:   python tests/test_core.py
"""

import sys
import json
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kratos.tokens import (
    estimate, estimate_messages, fit_to_budget, fit_excerpt,
    choose_num_ctx, relay_needed, model_max_ctx,
)
from kratos.agent import _parse_file_changes, _parse_file_deletions
from kratos.config import KratosConfig
from kratos.classifier import IntentClassifier, Intent
from kratos.analyzer import InputAnalyzer
from kratos.router import Router, Route
from kratos.compress import Compressor, _algo_compress, _algo_relay, _algo_memory
from kratos.memory import MemoryManager, MemoryEntry


# ── Token estimates ───────────────────────────────────────────────────────────

class TestTokenEstimator(unittest.TestCase):
    def test_empty(self):
        self.assertGreaterEqual(estimate(""), 0)

    def test_short(self):
        tok = estimate("hello world")
        self.assertGreater(tok, 0)

    def test_long_text(self):
        text = "def foo():\n    pass\n" * 200
        tok = estimate(text)
        # rough: ~4000 chars / 3.6 * 1.15 ≈ 1278 tokens + margin
        self.assertGreater(tok, 500)
        self.assertLess(tok, 5000)

    def test_fit_to_budget_short(self):
        text = "short text"
        result = fit_to_budget(text, 100)
        self.assertEqual(result, text)

    def test_fit_to_budget_truncates(self):
        text = "x" * 10000
        result = fit_to_budget(text, 100)
        self.assertIn("[truncated", result)
        self.assertLess(len(result), len(text))

    def test_messages(self):
        msgs = [
            {"role": "system",    "content": "You are an assistant."},
            {"role": "user",      "content": "Hello, what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        tok = estimate_messages(msgs)
        self.assertGreater(tok, 0)


class TestChooseNumCtx(unittest.TestCase):
    def test_basic(self):
        ctx = choose_num_ctx("huihui_ai/qwen3-abliterated:8b", 500, 512, vram_ceiling=32768)
        self.assertGreater(ctx, 512)
        self.assertLessEqual(ctx, 32768)

    def test_capped_by_vram(self):
        ctx = choose_num_ctx("huihui_ai/qwen3.5-abliterated:4b", 100, 100, vram_ceiling=4096)
        self.assertLessEqual(ctx, 4096)

    def test_capped_by_model_max(self):
        ctx = choose_num_ctx("kratos-planner", 1000, 1000, vram_ceiling=100000)
        self.assertLessEqual(ctx, model_max_ctx("kratos-planner"))

    def test_aligned_to_1024(self):
        ctx = choose_num_ctx("huihui_ai/qwen3-abliterated:8b", 500, 500, vram_ceiling=32768)
        self.assertEqual(ctx % 1024, 0)

    def test_model_max_ctx_known(self):
        self.assertEqual(model_max_ctx("huihui_ai/qwen3-abliterated:8b"),    40960)
        self.assertEqual(model_max_ctx("huihui_ai/qwen3.5-abliterated:4b"),  262144)
        self.assertEqual(model_max_ctx("kratos-planner"),                    16384)

    def test_model_max_ctx_unknown(self):
        self.assertEqual(model_max_ctx("some-unknown-model"), 32768)

    def test_relay_needed(self):
        self.assertTrue(relay_needed(10000, 12000, 0.80))
        self.assertFalse(relay_needed(1000,  12000, 0.80))


# ── File operation parsers ────────────────────────────────────────────────────

class TestFileParsers(unittest.TestCase):
    _SAMPLE = """### FILE: src/main.py
```python
def main():
    print("hello")
```

### FILE: src/utils.py
```python
def helper():
    return 42
```

### DELETE: old/legacy.py

### SUMMARY
Changed: src/main.py, src/utils.py — added main and helper"""

    def test_parse_changes(self):
        changes = _parse_file_changes(self._SAMPLE)
        self.assertEqual(len(changes), 2)
        paths = [c[0] for c in changes]
        self.assertIn("src/main.py", paths)
        self.assertIn("src/utils.py", paths)

    def test_parse_content(self):
        changes = _parse_file_changes(self._SAMPLE)
        contents = {p: c for p, c in changes}
        self.assertIn("def main():", contents["src/main.py"])

    def test_parse_deletions(self):
        dels = _parse_file_deletions(self._SAMPLE)
        self.assertIn("old/legacy.py", dels)

    def test_empty_output(self):
        self.assertEqual(_parse_file_changes("no file blocks"), [])
        self.assertEqual(_parse_file_deletions("no delete blocks"), [])


# ── KratosConfig ──────────────────────────────────────────────────────────────

class TestKratosConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = KratosConfig()
        self.assertEqual(cfg.permission, "mid")
        self.assertEqual(cfg.planner_num_ctx,    12288)
        self.assertEqual(cfg.coder_num_ctx,      24576)
        self.assertEqual(cfg.compressor_num_ctx, 8192)
        self.assertIsNotNone(cfg.compressor_model)

    def test_can_write_mid(self):
        cfg = KratosConfig(permission="mid")
        self.assertTrue(cfg.can_write())
        self.assertFalse(cfg.can_delete())

    def test_can_delete_high(self):
        cfg = KratosConfig(permission="high")
        self.assertTrue(cfg.can_write())
        self.assertTrue(cfg.can_delete())

    def test_cannot_write_low(self):
        cfg = KratosConfig(permission="low")
        self.assertFalse(cfg.can_write())

    def test_save_load_roundtrip(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            # Patch _project_dir to temp dir
            from kratos import config as cfg_mod
            orig = cfg_mod._project_dir
            cfg_mod._project_dir = lambda: Path(tmp)
            try:
                c = KratosConfig(permission="high", planner_num_ctx=4096)
                c.save("project")
                loaded = KratosConfig.load()
                self.assertEqual(loaded.permission, "high")
                self.assertEqual(loaded.planner_num_ctx, 4096)
            finally:
                cfg_mod._project_dir = orig


# ── Classifier + Router ───────────────────────────────────────────────────────

class TestClassifierRouter(unittest.TestCase):
    def setUp(self):
        self.analyzer   = InputAnalyzer()
        self.classifier = IntentClassifier()
        self.router     = Router()

    def _classify(self, text: str) -> Intent:
        return self.classifier.classify(self.analyzer.analyze(text))

    def _route(self, text: str) -> Route:
        return self.router.route(self._classify(text))

    def test_coding_intent(self):
        self.assertEqual(self._classify("implement the login feature"), Intent.CODING)

    def test_bugfix_intent(self):
        self.assertEqual(self._classify("fix the broken auth flow"), Intent.BUGFIX)

    def test_question_intent(self):
        self.assertIn(self._classify("what does this function do?"),
                      (Intent.QUESTION, Intent.EXPLAIN))

    def test_file_search_short(self):
        self.assertEqual(self._classify("where is config.py"), Intent.FILE_SEARCH)

    def test_followup(self):
        self.assertEqual(self._classify("continue"), Intent.FOLLOWUP)

    def test_routing_coding(self):
        self.assertEqual(self._route("implement a REST API"), Route.PLANNER_THEN_CODER)

    def test_routing_question(self):
        self.assertEqual(self._route("what is dependency injection?"), Route.PLANNER_ONLY)

    def test_routing_file_search(self):
        self.assertEqual(self._route("where is main.py"), Route.DIRECT_ANSWER)


# ── Algo fallbacks in compress.py ────────────────────────────────────────────

class TestAlgoFallbacks(unittest.TestCase):
    def test_algo_compress_keeps_pairs(self):
        history = [
            {"role": "user",      "content": f"task {i}"}
            if i % 2 == 0 else
            {"role": "assistant", "content": f"response {i}"}
            for i in range(12)
        ]
        orig_len = len(history)
        _algo_compress(history, keep_pairs=2)
        self.assertLess(len(history), orig_len)
        self.assertLessEqual(len(history), 5)   # 2 pairs + potential prefix

    def test_algo_compress_noop_small(self):
        history = [
            {"role": "user",      "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        _algo_compress(history, keep_pairs=4)
        self.assertEqual(len(history), 2)   # unchanged

    def test_algo_relay(self):
        large = "line content\n" * 1000
        result = _algo_relay(large, max_chars=2000)
        self.assertLessEqual(len(result), 2200)   # slight over due to join

    def test_algo_memory_circular(self):
        entries = _algo_memory("detected circular import between modules", ["a.py"])
        cats = [e["category"] for e in entries]
        self.assertIn("convention", cats)

    def test_algo_memory_files(self):
        entries = _algo_memory("", ["a.py", "b.py"])
        self.assertTrue(any("Modified" in e["content"] for e in entries))


class TestCompressorModelFallback(unittest.TestCase):
    """Compressor falls back gracefully when the model returns empty/garbage."""

    def _make_compressor(self, bridge_response: str) -> Compressor:
        cfg = KratosConfig()
        bridge = MagicMock()
        # Simulate model returning empty string
        bridge.chat.return_value = iter([
            (bridge_response, "text"),
            (json.dumps({"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}), "usage"),
        ])
        return Compressor(bridge, cfg)

    def test_compress_history_empty_response_uses_algo(self):
        comp = self._make_compressor("")
        history = [
            {"role": "user",      "content": f"msg {i}"}
            if i % 2 == 0 else
            {"role": "assistant", "content": f"resp {i}"}
            for i in range(10)
        ]
        result = comp.compress_history(history, keep_pairs=2)
        # algo fallback ran (False = model not used)
        self.assertFalse(result)
        # history was still compressed
        self.assertLess(len(history), 10)

    def test_generate_memory_invalid_json_falls_back(self):
        comp = self._make_compressor("not valid json {{{")
        entries = comp.generate_memory("task", "plan", "output", ["a.py"])
        # algo fallback returns a list
        self.assertIsInstance(entries, list)

    def test_relay_empty_response_uses_algo(self):
        comp = self._make_compressor("")
        result = comp.relay_large_input("task", "x" * 5000)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ── MemoryManager ─────────────────────────────────────────────────────────────

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._mem = MemoryManager(Path(self._tmpdir), Path(self._tmpdir))

    def test_add_session(self):
        self._mem.add(MemoryEntry("decision", "use SQLite"), "session")
        all_m = self._mem.list_all()
        self.assertEqual(len(all_m["session"]), 1)

    def test_secret_filtered(self):
        self._mem.add(MemoryEntry("decision", "api_key=sk-abc123456789012345678901234"), "session")
        all_m = self._mem.list_all()
        self.assertEqual(len(all_m["session"]), 0)

    def test_project_no_duplicate(self):
        e = MemoryEntry("convention", "use snake_case")
        self._mem.add(e, "project")
        self._mem.add(e, "project")
        all_m = self._mem.list_all()
        self.assertEqual(len(all_m["project"]), 1)

    def test_get_relevant(self):
        self._mem.add(MemoryEntry("solution", "fixed auth bug in login.py"), "session")
        self._mem.add(MemoryEntry("solution", "unrelated database fix"), "session")
        results = self._mem.get_relevant(["auth", "login"])
        self.assertTrue(any("auth" in e.content for e in results))

    def test_add_from_compress(self):
        entries = [
            {"category": "convention", "content": "avoid global state"},
            {"category": "file_role",  "content": "main.py is the entry point"},
        ]
        self._mem.add_from_compress(entries, "session")
        all_m = self._mem.list_all()
        self.assertEqual(len(all_m["session"]), 2)

    def test_track_files(self):
        self._mem.track_files(["src/a.py", "src/b.py"])
        all_m = self._mem.list_all()
        self.assertEqual(len(all_m["task"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
