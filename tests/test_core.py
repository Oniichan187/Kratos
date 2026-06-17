"""Core unit tests — no Ollama required.

Run:  python -m pytest tests/ -v
or:   python tests/test_core.py
"""

import os
import sys
import json
import tempfile
import threading
import time
import types
import unittest
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kratos.llm.tokens import (
    estimate, estimate_messages, fit_to_budget, fit_excerpt,
    choose_num_ctx, relay_needed, model_max_ctx,
    effective_num_ctx, role_context_windows,
)
from kratos.core.agent import KratosAgent
from kratos.verification import (
    ProvenWork,
    _extract_readme_verification_commands,
    _infer_project_verification_commands,
    _is_safe_inspect_command,
    _is_safe_verification_command,
    _is_test_verification_command,
    _missing_command_paths,
    _proven_work_satisfied,
    _detect_project_toolchains,
    _command_toolchain,
    _compile_check_cmds,
    _extract_step_file_refs,
    _extract_plan_steps,
    _parse_step_tests,
    _patch_dotnet_test_runner,
    CommandRegistry,
)
from kratos.execution.parsing import _parse_file_changes, _parse_file_deletions
from kratos.execution.tools import parse_actions, do_read, do_write, do_delete, do_command, do_inspect
from kratos.roles import _coder_context_block, run_coder_loop
from kratos.config import KratosConfig
from kratos.planning import PlanItem, ExecutionPlan, parse_execution_plan, render_checklist, render_plan_status, refresh_plan_status, plan_all_done
from kratos.llm.bridge import OllamaBridge
from kratos.classifier import IntentClassifier, Intent
from kratos.analyzer import InputAnalyzer
from kratos.router import Router, Route
from kratos.compress import Compressor, _algo_compress, _algo_relay, _algo_memory
from kratos.knowledge import ProjectKnowledge
from kratos.memory import MemoryManager, MemoryEntry
from kratos.prompts import (
    PromptManager, DEFAULT_PROMPTS,
    load_prompts, get_system, get_snippet, get_predict, get_marker,
    reload_prompts,
)
from kratos.app.prompt_frame import _PlannerFilter
from kratos.app.tui import KratosApp
from kratos.roles.verifier import _verify_msg
from kratos.ui.status import status_bar


def drain_generator(gen):
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as exc:
            return events, exc.value


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
        self.assertEqual(model_max_ctx("qwen3:4b"),                          262144)
        self.assertEqual(model_max_ctx("kratos-planner"),                    131072)

    def test_model_max_ctx_unknown(self):
        self.assertEqual(model_max_ctx("some-unknown-model"), 32768)

    def test_relay_needed(self):
        self.assertTrue(relay_needed(10000, 12000, 0.80))
        self.assertFalse(relay_needed(1000,  12000, 0.80))


class TestEffectiveContextWindows(unittest.TestCase):
    def test_effective_coder_window_respects_vram_cap(self):
        ctx = effective_num_ctx(
            "huihui_ai/qwen3.5-abliterated:4b",
            configured_num_ctx=262144,
            vram_ctx_ceiling=65536,
        )
        self.assertEqual(ctx, 65536)

    def test_role_context_windows_show_real_model_call_limits(self):
        cfg = KratosConfig(coder_num_ctx=262144, vram_ctx_ceiling=65536)
        windows = role_context_windows(cfg)
        # coder model (qwen2.5-coder 7b) maxes at 32768 regardless of the 65536 cap;
        # planner/verifier (deepseek-r1 8b, 128k native) are capped at 65536 here.
        self.assertEqual(windows["coder"], 32768)
        self.assertEqual(windows["planner"], 65536)
        self.assertEqual(windows["verifier"], 65536)


class TestPromptContextGuard(unittest.TestCase):
    def _agent(self, **kwargs):
        params = {
            "auto_compress": False,
            "coder_num_ctx": 262144,
            "vram_ctx_ceiling": 65536,
        }
        params.update(kwargs)
        cfg = KratosConfig(**params)
        return KratosAgent(cfg, MagicMock())

    def test_prepare_model_prompt_never_returns_prompt_over_effective_ctx(self):
        agent = self._agent()
        huge_msg = "x" * 320_000
        messages, prompt_tok, num_ctx, stored_msg, events = agent._prepare_model_prompt(
            "coder",
            agent.config.coder_model,
            "system",
            [],
            huge_msg,
            1024,
        )
        self.assertEqual(num_ctx, 32768)
        self.assertLessEqual(prompt_tok, num_ctx)
        self.assertLess(len(stored_msg), len(huge_msg))
        self.assertTrue(any(ev[0] == "warn" for ev in events))
        self.assertEqual(messages[-1]["content"], stored_msg)

    def test_full_prompt_pressure_triggers_history_compression(self):
        agent = self._agent(auto_compress=True)
        history = [
            {"role": "user", "content": "old user " * 2000},
            {"role": "assistant", "content": "old assistant " * 2000},
        ]
        called = {"value": False}

        def fake_compress(hist, keep_pairs=4):
            called["value"] = True
            hist.clear()
            return True

        agent._compressor.compress_history = fake_compress
        agent._prepare_model_prompt(
            "coder",
            agent.config.coder_model,
            "system",
            history,
            "new input " * 40_000,
            1024,
        )
        self.assertTrue(called["value"])

    def test_stepwise_policy_has_no_small_project_full_pass_bypass(self):
        process_src = inspect.getsource(KratosAgent.process)
        self.assertNotIn("small_project_full_pass", process_src)
        self.assertNotIn("Small project detected; using full-pass", process_src)

    def test_knowledge_bootstrap_rebuilds_only_when_empty(self):
        agent = self._agent()

        class FakeKnowledge:
            def __init__(self, chunks):
                self._chunks = chunks
                self.rebuild_calls = []

            def status(self):
                return {"chunks": self._chunks}

            def rebuild(self, force=False):
                self.rebuild_calls.append(force)
                self._chunks = 17
                return 17

        empty = FakeKnowledge(0)
        agent._knowledge = empty
        self.assertEqual(agent._knowledge_chunk_count(), 0)
        self.assertEqual(agent._ensure_knowledge_bootstrap(), 17)
        self.assertEqual(empty.rebuild_calls, [False])

        ready = FakeKnowledge(9)
        agent._knowledge = ready
        self.assertEqual(agent._knowledge_chunk_count(), 9)
        self.assertEqual(agent._ensure_knowledge_bootstrap(), 9)
        self.assertEqual(ready.rebuild_calls, [])

    def test_role_runner_logs_full_model_payloads(self):
        class FakeBridge:
            def chat(self, **kwargs):
                yield ("alpha", "text")
                yield ("beta", "text")
                yield (json.dumps({
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                }), "usage")

        agent = KratosAgent(
            KratosConfig(auto_compress=False, enable_semantic_retrieval=False),
            FakeBridge(),
        )
        events, value = drain_generator(agent._run_coder("unique-user-message"))
        self.assertEqual(value, "alphabeta")

        payloads = [
            json.loads(content)
            for source, content, _kind in events
            if source == "log"
        ]
        model_input = next(item for item in payloads if item["type"] == "model_input")
        model_output = next(item for item in payloads if item["type"] == "model_output")
        model_stream = [item for item in payloads if item["type"] == "model_stream"]

        self.assertEqual(model_input["role"], "coder")
        self.assertIn("messages", model_input)
        self.assertEqual(model_input["messages"][-1]["content"], "unique-user-message")
        self.assertEqual(model_output["text"], "alphabeta")
        self.assertEqual([item["token"] for item in model_stream], ["alpha", "beta"])


class TestOllamaBridgeCancel(unittest.TestCase):
    def test_cancel_active_closes_current_response(self):
        bridge = OllamaBridge()

        class FakeResponse:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        response = FakeResponse()
        bridge._active_response = response
        bridge.cancel_active()
        self.assertTrue(response.closed)

    def test_chat_with_pre_set_cancel_event_does_not_call_network(self):
        bridge = OllamaBridge()
        cancel = threading.Event()
        cancel.set()
        stream = bridge.chat(
            "model",
            [{"role": "user", "content": "hello"}],
            cancel_event=cancel,
        )
        with self.assertRaises(KeyboardInterrupt):
            next(stream)


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

    def test_malformed_file_marker_does_not_swallow_next_file_path(self):
        text = """### FILE: src/TaskRepository.cs (updated)
> Note: no code block for this marker.

### FILE: src/Program.cs
```csharp
public static class Program {}
```
"""
        changes = _parse_file_changes(text)
        self.assertEqual(changes, [("src/Program.cs", "public static class Program {}\n")])

    def test_file_marker_strips_trailing_parenthetical_note(self):
        text = """### FILE: src/TaskParser.cs (updated with overdue logic)
```csharp
public sealed class TaskParser {}
```
"""
        changes = _parse_file_changes(text)
        self.assertEqual(changes, [("src/TaskParser.cs", "public sealed class TaskParser {}\n")])


class TestExecutionPlanHelpers(unittest.TestCase):
    def test_parse_execution_plan_explicit_checklist(self):
        markdown = (
            "## Detailed Plan\n"
            "Work out the fix in depth.\n\n"
            "## CHECKLIST\n"
            "- Coder-Loop and Action-Dispatch fertig implementieren\n"
            "  File: kratos/roles/coder.py\n"
            "  VERIFY: python -m pytest tests/test_core.py\n"
            "- Agent-Wiring auf coder_loop umstellen und Legacy-Fallback erhalten\n"
            "  File: kratos/core/agent.py\n"
            "  VERIFY: python -m pytest tests/test_core.py\n"
        )
        plan = parse_execution_plan(markdown)
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(len(plan.items), 2)
        self.assertEqual(plan.items[0].title, "Coder-Loop and Action-Dispatch fertig implementieren")
        self.assertIn("kratos/roles/coder.py", plan.items[0].file_refs)
        self.assertEqual(plan.items[0].verify_cmd, "python -m pytest tests/test_core.py")

    def test_parse_execution_plan_execution_order_fallback(self):
        markdown = (
            "## Summary\n"
            "- Goal\n\n"
            "## Execution Order\n"
            "1. First fix kratos/app/tui.py\n"
            "2. Then update kratos/prompts_default.json\n"
        )
        plan = parse_execution_plan(markdown)
        self.assertEqual(len(plan.items), 2)
        self.assertEqual(plan.items[0].title, "First fix kratos/app/tui.py")
        self.assertIn("kratos/app/tui.py", plan.items[0].file_refs)

    def test_planner_prompt_requires_order_files_and_verification_sections(self):
        prompts = load_prompts()
        text = prompts.get_system("planner")
        self.assertIn("## Execution Order", text)
        self.assertIn("## Success Criteria", text)
        self.assertIn("## Risks", text)
        self.assertIn("## Files", text)
        self.assertIn("## Verification", text)
        self.assertIn("## CHECKLIST", text)

    def test_render_checklist_and_status(self):
        plan = ExecutionPlan(
            markdown="## CHECKLIST\n- First\n- Second",
            items=[
                PlanItem(index=1, title="First", status="done"),
                PlanItem(index=2, title="Second", status="pending"),
            ],
        )
        compact = render_checklist(plan.items, compact=True)
        verbose = render_checklist(plan.items, compact=False)
        self.assertIn("☑ First", compact)
        self.assertIn("□ Second", compact)
        self.assertIn("1. First", verbose)
        self.assertIn("2. Second", verbose)
        self.assertIn("PLAN STATUS:", render_plan_status(plan.items))

    def test_refresh_plan_status_marks_done(self):
        plan = ExecutionPlan(
            markdown="## CHECKLIST\n- First",
            items=[
                PlanItem(index=1, title="First", file_refs=["src/app.py"], verify_cmd="python -m pytest tests/test_core.py"),
            ],
        )
        proof = ProvenWork(iteration=1)
        proof.files_changed.append("src/app.py")
        proof.commands.append({"cmd": "python -m pytest tests/test_core.py", "exit_code": 0, "is_test": True})
        refreshed = refresh_plan_status(plan, proof, ["src/app.py"])
        self.assertTrue(plan_all_done(refreshed))
        self.assertEqual(refreshed.items[0].status, "done")

    def test_planner_filter_emits_only_checklist(self):
        captured = []
        filt = _PlannerFilter(lambda text, style="": captured.append((text, style)))
        filt.feed(
            "## Detailed Plan\n"
            "Explain everything.\n\n"
            "## CHECKLIST\n"
            "- One\n"
            "  File: a.py\n"
            "  VERIFY: python -m pytest tests/test_core.py\n"
        )
        filt.flush()
        joined = "\n".join(text for text, _style in captured)
        self.assertIn("PLAN CHECKLIST", joined)
        self.assertIn("□ One", joined)
        self.assertNotIn("Explain everything.", joined)

    def test_status_bar_does_not_render_session_total_or_compose_meter(self):
        from io import StringIO
        from rich.console import Console

        panel = status_bar(
            scope="project",
            permission="mid",
            ctx_state={
                "planner": (2048, 40960),
                "coder": (11264, 65536),
                "verifier": (3072, 40960),
            },
            elapsed_s=12.3,
            project_name="demo",
            current_section="coder",
            goal="",
            hint="",
        )
        buf = StringIO()
        cap = Console(file=buf, force_terminal=True, color_system="standard", width=120)
        cap.print(panel)
        rendered = buf.getvalue()
        self.assertNotIn("∑", rendered)
        self.assertNotIn("compose", rendered.lower())
        self.assertIn("C", rendered)
        self.assertIn("11k/64k", rendered)
        self.assertNotIn("P 2k/40k", rendered)
        self.assertNotIn("V 3k/40k", rendered)

    def test_tui_footer_shows_only_active_role_when_busy(self):
        app = object.__new__(KratosApp)
        app._busy = True
        app._task_start = 0.0
        app._last_task_s = None
        app.session_start = 0.0
        app._ctx_live = {
            "planner": (2048, 40960),
            "coder": (11264, 65536),
            "verifier": (3072, 40960),
        }
        app._current_section = "coder"
        app.kratos_config = KratosConfig(
            always_max_ctx=False,
            permission="mid",
            coder_num_ctx=262144,
            vram_ctx_ceiling=65536,
        )
        app.project_root = Path("demo")

        footer = KratosApp._footer_text(app)
        rendered = footer.plain
        self.assertIn("C ", rendered)
        self.assertNotIn("P ", rendered)
        self.assertNotIn("V ", rendered)

    def test_plan_box_limits_to_five_lines_and_shows_overflow(self):
        app = object.__new__(KratosApp)
        app._busy = True
        app._task_start = 0.0
        app._last_task_s = None
        app._ctx_live = {"planner": (0, 40960), "coder": (2048, 65536), "verifier": (0, 40960)}
        app._current_section = "coder"
        app._plan_state = ExecutionPlan(
            markdown="## CHECKLIST\n- One\n- Two\n- Three\n- Four\n- Five\n- Six",
            items=[
                PlanItem(index=i + 1, title=f"Item {i + 1}", status="pending")
                for i in range(6)
            ],
        )
        app._plan_proof = ProvenWork(iteration=0)
        app._plan_done_at = {}
        app._plan_last_status = {}
        app.kratos_config = KratosConfig(always_max_ctx=False, permission="mid")
        app.project_root = Path("demo")

        text = KratosApp._plan_box_text(app)
        lines = text.plain.splitlines()
        self.assertLessEqual(len(lines), 5)
        self.assertIn("PLAN", lines[0])
        self.assertIn("more items", lines[-1])

    def test_plan_box_hides_done_items_after_delay(self):
        app = object.__new__(KratosApp)
        app._busy = True
        app._task_start = 0.0
        app._last_task_s = None
        app._ctx_live = {"planner": (0, 40960), "coder": (2048, 65536), "verifier": (0, 40960)}
        app._current_section = "coder"
        app._plan_state = ExecutionPlan(
            markdown="## CHECKLIST\n- One",
            items=[PlanItem(index=1, title="One", status="done")],
        )
        app._plan_proof = ProvenWork(iteration=0)
        app._plan_done_at = {1: time.monotonic() - 6.0}
        app._plan_last_status = {1: "done"}
        app.kratos_config = KratosConfig(always_max_ctx=False, permission="mid")
        app.project_root = Path("demo")

        visible = KratosApp._plan_visible_items(app)
        self.assertEqual(visible, [])
        text = KratosApp._plan_box_text(app)
        self.assertIn("all checklist items completed", text.plain)

    def test_tui_layout_reserves_fixed_plan_box_band(self):
        css = KratosApp.CSS
        self.assertIn("#status_footer_row", css)
        self.assertIn("#plan_box_row", css)
        self.assertIn("height: 5;", css)
        self.assertIn("min-height: 5;", css)
        self.assertIn("max-height: 16;", css)
        self.assertLess(css.index("#status_footer_row"), css.index("#plan_box_row"))
        self.assertLess(css.index("#plan_box_row"), css.index("#input_row"))

    def test_planner_buffer_syncs_live_plan_box_before_end(self):
        app = object.__new__(KratosApp)
        app._planner_buf = (
            "## Summary\n"
            "- Goal\n\n"
            "## Execution Order\n"
            "1. First fix kratos/app/tui.py\n"
            "2. Then update kratos/prompts_default.json\n"
        )
        app._plan_state = None
        app._plan_source_text = ""
        app._plan_done_at = {}
        app._plan_last_status = {}
        app._busy = False
        app._current_section = "planner"
        app._ctx_live = {"planner": (0, 40960), "coder": (0, 65536), "verifier": (0, 40960)}
        app._task_start = 0.0
        app._last_task_s = None
        app.kratos_config = KratosConfig(always_max_ctx=False, permission="mid")
        app.project_root = Path("demo")
        app._refresh_plan_box = lambda: None

        KratosApp._sync_plan_from_planner_buffer(app)
        self.assertIsNotNone(app._plan_state)
        self.assertGreaterEqual(len(app._plan_state.items), 2)
        rendered = KratosApp._plan_box_text(app).plain
        self.assertIn("First fix kratos/app/tui.py", rendered)


class TestPlannerArtifacts(unittest.TestCase):
    def test_planner_only_writes_markdown_artifact_and_memory_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cwd = os.getcwd()
            os.chdir(root)
            try:
                plan_markdown = (
                    "## Detailed Plan\n"
                    "Work out the fix.\n\n"
                    "## CHECKLIST\n"
                    "- First item\n"
                    "  File: src/app.py\n"
                    "  VERIFY: python -m pytest tests/test_core.py\n"
                )

                def fake_run_planner(self, *args, **kwargs):
                    yield ("planner", plan_markdown, "text")
                    return plan_markdown

                agent = KratosAgent(KratosConfig(always_max_ctx=False), MagicMock())
                agent._knowledge = None
                agent._run_planner = types.MethodType(fake_run_planner, agent)
                agent._compressor.generate_memory = lambda *args, **kwargs: []

                events = list(agent.process("what is dependency injection?"))
                self.assertTrue(any(ev[0] == "header" and ev[1] == "planner" for ev in events))

                plans_dir = root / ".kratos" / "plans"
                md_files = list(plans_dir.glob("*.md"))
                self.assertEqual(len(md_files), 1)
                content = md_files[0].read_text(encoding="utf-8")
                self.assertIn("## Detailed Plan", content)
                self.assertIn("## CHECKLIST", content)
                self.assertFalse(content.lstrip().startswith("{"))

                project_entries = agent.memory.list_all()["project"]
                self.assertTrue(project_entries)
                self.assertTrue(any("Planner saved Markdown plan" in e.content for e in project_entries))
            finally:
                os.chdir(old_cwd)


class TestAutoCompressionArtifacts(unittest.TestCase):
    def test_auto_compress_writes_markdown_and_ingests_knowledge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cwd = os.getcwd()
            os.chdir(root)
            try:
                class FakeKnowledge:
                    def __init__(self):
                        self.calls = []

                    def ingest_markdown_artifact(self, path, metadata=None):
                        self.calls.append((path, metadata))
                        return 1

                cfg = KratosConfig(auto_compress=True, always_max_ctx=False)
                agent = KratosAgent(cfg, MagicMock())
                agent._knowledge = FakeKnowledge()

                def fake_compress(history, keep_pairs=4):
                    history[:]= [{"role": "user", "content": "[Compressed prior context]\n\n## Compressed Summary\nkept"}]
                    return True

                agent._compressor.compress_history = fake_compress
                history = [
                    {"role": "user", "content": "old user " * 200},
                    {"role": "assistant", "content": "old assistant " * 200},
                    {"role": "user", "content": "new user " * 200},
                ]
                events = []
                ok = agent._auto_compress_if_needed(
                    history,
                    agent.config.coder_model,
                    1000,
                    prompt_tokens=900,
                    role="coder",
                    _pending_events=events,
                )

                self.assertTrue(ok)
                artifact_dir = root / ".kratos" / "knowledge" / "compressions"
                files = list(artifact_dir.glob("*.md"))
                self.assertEqual(len(files), 1)
                content = files[0].read_text(encoding="utf-8")
                self.assertIn("# Auto-Compression Artifact", content)
                self.assertIn("## Compressed Summary", content)
                self.assertIn("Role: coder", content)
                self.assertEqual(len(agent._knowledge.calls), 1)
                self.assertTrue(agent._knowledge.calls[0][0].name.endswith(".md"))
                self.assertEqual(agent._knowledge.calls[0][1]["kind"], "compression_artifact")
            finally:
                os.chdir(old_cwd)

    def test_knowledge_retrieves_compression_artifact_dynamically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cwd = os.getcwd()
            os.chdir(root)
            try:
                artifact_dir = root / ".kratos" / "knowledge" / "compressions"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact = artifact_dir / "2026-06-07_18-22-32_coder_demo.md"
                artifact.write_text(
                    "# Auto-Compression Artifact\n\n"
                    "## Compressed Summary\n"
                    "History was compressed because the coder context overflowed.\n\n"
                    "## Removed Context Signals\n"
                    "- `user`: the failing test output mentions AssertionError\n",
                    encoding="utf-8",
                )

                with patch("kratos.knowledge.base._HAS_LANCEDB", False):
                    kb = ProjectKnowledge(KratosConfig(always_max_ctx=False), MagicMock())
                    kb._fallback_chunks = []
                    kb._table = None
                    n = kb.ingest_markdown_artifact(artifact, metadata={"kind": "compression_artifact"})
                    self.assertGreater(n, 0)
                    results = kb.retrieve("AssertionError from compressed coder context", top_k=5)
                    self.assertTrue(any(ch.kind == "compression_artifact" for ch in results))
                    self.assertTrue(any("AssertionError" in (ch.text + ch.summary) for ch in results))
            finally:
                os.chdir(old_cwd)


class TestCoderActionTools(unittest.TestCase):
    def test_parse_actions_accepts_tolerant_markers(self):
        pm = load_prompts()
        text = """### read src/app.py

### inspect: rg -n "TODO" src

### file src/app.py
```python
VALUE = 1
```

### RUN python -m pytest tests
### verify: npm run build
### done:
### delete old.py
"""
        actions = parse_actions(text, pm)
        self.assertEqual(actions["reads"], ["src/app.py"])
        self.assertEqual(actions["inspects"], ['rg -n "TODO" src'])
        self.assertEqual(actions["files"], [("src/app.py", "VALUE = 1\n")])
        self.assertEqual(actions["deletes"], ["old.py"])
        self.assertEqual(actions["commands"], ["python -m pytest tests", "npm run build"])
        self.assertTrue(actions["done"])

    def test_read_write_delete_handlers_update_disk_and_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = ProvenWork(iteration=1)
            snapshots = {}

            _, write_obs = drain_generator(do_write(root, "src/app.py", "VALUE = 1\n", proof, 0, snapshots))
            self.assertTrue(write_obs["ok"])
            self.assertEqual((root / "src" / "app.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertIn("src/app.py", proof.files_changed)
            self.assertTrue(proof.file_checks[-1]["ok"])

            _, read_obs = drain_generator(do_read(root, "src/app.py"))
            self.assertTrue(read_obs["ok"])
            self.assertIn("VALUE = 1", read_obs["content"])

            _, delete_obs = drain_generator(do_delete(root, "src/app.py", proof, 0, snapshots))
            self.assertTrue(delete_obs["ok"])
            self.assertFalse((root / "src" / "app.py").exists())

    def test_handlers_refuse_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = ProvenWork(iteration=1)
            _, write_obs = drain_generator(do_write(root, "../escape.py", "x", proof, 0, {}))
            _, read_obs = drain_generator(do_read(root, "../escape.py"))
            self.assertFalse(write_obs["ok"])
            self.assertFalse(read_obs["ok"])
            self.assertFalse((root.parent / "escape.py").exists())

    def test_command_handler_skips_unsafe_mismatch_and_missing_paths(self):
        class FakeAgent:
            def __init__(self):
                self.calls = 0

            def _run_verification_command(self, command):
                self.calls += 1
                return {
                    "cmd": command.cmd,
                    "purpose": command.purpose,
                    "source": command.source,
                    "is_test": command.is_test,
                    "exit_code": 0,
                    "duration_seconds": 0.01,
                    "output": "ok",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            proof = ProvenWork(iteration=1)
            agent = FakeAgent()

            python_reg = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            _, unsafe = drain_generator(do_command(agent, root, python_reg, "echo hello", proof))
            self.assertTrue(unsafe["skipped"])

            _, missing = drain_generator(do_command(agent, root, python_reg, "python -m pytest tests/test_missing.py", proof))
            self.assertTrue(missing["skipped"])

            dotnet_root = root / "dotnet_only"
            dotnet_root.mkdir()
            (dotnet_root / "App.sln").write_text("", encoding="utf-8")
            dotnet_reg = CommandRegistry(KratosConfig(always_max_ctx=False), dotnet_root).discover()
            _, mismatch = drain_generator(do_command(agent, dotnet_root, dotnet_reg, "python -m pytest tests", proof))
            self.assertTrue(mismatch["skipped"])
            self.assertEqual(agent.calls, 0)

    def test_command_handler_executes_safe_command_through_agent(self):
        class FakeAgent:
            def __init__(self):
                self.calls = []

            def _run_verification_command(self, command):
                self.calls.append(command)
                return {
                    "cmd": command.cmd,
                    "purpose": command.purpose,
                    "source": command.source,
                    "is_test": command.is_test,
                    "exit_code": 0,
                    "duration_seconds": 0.01,
                    "output": "passed",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            reg = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            proof = ProvenWork(iteration=1)
            agent = FakeAgent()
            _, obs = drain_generator(do_command(agent, root, reg, "python -m pytest tests", proof))
            self.assertFalse(obs["skipped"])
            self.assertTrue(obs["ok"])
            self.assertEqual(len(agent.calls), 1)
            self.assertEqual(proof.commands[-1]["exit_code"], 0)

    def test_inspect_handler_executes_read_only_command_through_agent(self):
        class FakeAgent:
            def __init__(self):
                self.calls = []

            def _run_readonly_command(self, cmd, root):
                self.calls.append((cmd, root))
                return {
                    "cmd": cmd,
                    "purpose": "readonly inspection",
                    "source": "coder-inspect",
                    "is_test": False,
                    "exit_code": 0,
                    "duration_seconds": 0.01,
                    "output": "line1\nline2\n",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            agent = FakeAgent()
            events, obs = drain_generator(do_inspect(agent, root, 'Get-Content src/app.py | Select-Object -First 1'))
            self.assertFalse(obs["skipped"])
            self.assertTrue(obs["ok"])
            self.assertEqual(len(agent.calls), 1)
            self.assertTrue(any(ev[0] == "tool" for ev in events))

    def test_inspect_handler_skips_write_or_escape_commands(self):
        class FakeAgent:
            def _run_readonly_command(self, cmd, root):
                raise AssertionError("should not run")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = FakeAgent()
            _, unsafe = drain_generator(do_inspect(agent, root, "Set-Content src/app.py x"))
            _, escape = drain_generator(do_inspect(agent, root, "Get-Content ..\\secret.txt"))
            self.assertTrue(unsafe["skipped"])
            self.assertTrue(escape["skipped"])


class TestCoderReActLoop(unittest.TestCase):
    class FakeAgent:
        def __init__(self, root: Path, outputs: list[str], command_results: list[dict], **cfg):
            self.config = KratosConfig(always_max_ctx=False, **cfg)
            self.pending_file_changes = []
            self.pending_file_deletions = []
            self._memory = MagicMock()
            self._knowledge = None
            self._outputs = list(outputs)
            self.command_results = list(command_results)
            self.prompts = []
            self._indexer = types.SimpleNamespace(root=root)

        def _is_cancelled(self):
            return False

        def _run_coder(self, msg):
            self.prompts.append(msg)
            out = self._outputs.pop(0)
            yield ("coder", out, "text")
            return out

        def _run_verification_command(self, command):
            result = dict(self.command_results.pop(0))
            result.update({
                "cmd": command.cmd,
                "purpose": command.purpose,
                "source": command.source,
                "is_test": command.is_test,
            })
            return result

    def _ctx(self):
        from kratos.context import ContextPackage
        return ContextPackage(user_input="test", intent="coding", route="planner_then_coder")

    def test_loop_converges_after_write_fail_read_fix_pass_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            reg = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            outputs = [
                """### FILE: src/calc.py
```python
def add(a, b):
    return a - b
```
### VERIFY: python -m pytest tests
""",
                "### READ: src/calc.py",
                """### FILE: src/calc.py
```python
def add(a, b):
    return a + b
```
### VERIFY: python -m pytest tests
""",
                "### DONE",
            ]
            results = [
                {"exit_code": 1, "duration_seconds": 0.01, "output": "AssertionError: expected 3"},
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
            ]
            agent = self.FakeAgent(root, outputs, results, permission="mid", max_coder_iterations=6)
            proof = ProvenWork(iteration=1)

            events, result = drain_generator(run_coder_loop(
                agent, "fix add", "plan", None, Intent.CODING, Route.PLANNER_THEN_CODER,
                self._ctx(), reg, proof, 0, "", root, {},
            ))
            transcript, changes, deletions = result

            self.assertEqual(changes["src/calc.py"], "def add(a, b):\n    return a + b\n")
            self.assertEqual(deletions, set())
            self.assertEqual((root / "src" / "calc.py").read_text(encoding="utf-8"), changes["src/calc.py"])
            self.assertEqual([c["exit_code"] for c in proof.commands], [1, 0])
            self.assertTrue(_proven_work_satisfied(proof, require_test=True))
            self.assertEqual(len(agent.prompts), 4)
            self.assertIn("AssertionError", agent.prompts[1])
            self.assertIn("return a - b", agent.prompts[2])
            self.assertIn("CODER LOOP ITERATION 3", transcript)
            self.assertTrue(any("converged" in ev[1] for ev in events if ev[0] == "info"))

    def test_loop_runs_unbounded_when_max_iterations_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            reg = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            outputs = [
                "### VERIFY: python -m pytest tests",
                "### VERIFY: python -m pytest tests",
                "### VERIFY: python -m pytest tests",
                "### VERIFY: python -m pytest tests",
                "### VERIFY: python -m pytest tests",
                "### VERIFY: python -m pytest tests",
                "### DONE",
            ]
            results = [
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
            ]
            agent = self.FakeAgent(root, outputs, results, max_coder_iterations=0)
            proof = ProvenWork(iteration=1)

            events, _ = drain_generator(run_coder_loop(
                agent, "fix", "## CHECKLIST\n- First\n", None, Intent.CODING, Route.PLANNER_THEN_CODER,
                self._ctx(), reg, proof, 0, "", root, {},
            ))
            self.assertEqual(len(agent.prompts), 7)
            self.assertFalse(any("max_coder_iterations=" in ev[1] for ev in events if ev[0] == "warn"))

    def test_loop_honors_max_iterations_without_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            reg = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            outputs = [
                "### VERIFY: python -m pytest tests",
                "### VERIFY: python -m pytest tests",
            ]
            results = [
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
                {"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"},
            ]
            agent = self.FakeAgent(root, outputs, results, max_coder_iterations=2)
            proof = ProvenWork(iteration=1)

            events, _ = drain_generator(run_coder_loop(
                agent, "fix", "plan", None, Intent.CODING, Route.PLANNER_THEN_CODER,
                self._ctx(), reg, proof, 0, "", root, {},
            ))
            self.assertEqual(len(agent.prompts), 2)
            self.assertTrue(any("max_coder_iterations=2" in ev[1] for ev in events if ev[0] == "warn"))

    def test_loop_permission_gate_skips_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            outputs = ["""### FILE: blocked.py
```python
VALUE = 1
```"""]
            agent = self.FakeAgent(root, outputs, [], permission="low", max_coder_iterations=1)
            proof = ProvenWork(iteration=1)

            events, result = drain_generator(run_coder_loop(
                agent, "write", "plan", None, Intent.CODING, Route.PLANNER_THEN_CODER,
                self._ctx(), reg, proof, 0, "", root, {},
            ))
            _, changes, _ = result
            self.assertEqual(changes, {})
            self.assertFalse((root / "blocked.py").exists())
            self.assertTrue(any("write permission disabled" in ev[1] for ev in events if ev[0] == "warn"))

    def test_coder_loop_false_uses_legacy_one_shot_process_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cwd = os.getcwd()
            os.chdir(root)
            try:
                (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
                (root / "tests").mkdir()
                (root / "tests" / "test_dummy.py").write_text("def test_dummy():\n    assert True\n", encoding="utf-8")

                responses = iter([
                    "1. Write legacy file. STEP_VERIFY: `python -m pytest tests`",
                    """### FILE: src/legacy.py
```python
VALUE = 1
```""",
                    "VERIFIED\n(implementation is complete, correct, and final tests passed)",
                ])
                calls = []

                def fake_chat(**kwargs):
                    calls.append(kwargs)
                    return iter([(next(responses), "text")])

                bridge = MagicMock()
                bridge.chat = fake_chat
                agent = KratosAgent(
                    KratosConfig(
                        coder_loop=False,
                        auto_compress=False,
                        always_max_ctx=False,
                        max_verify_iterations=1,
                        verification_timeout_seconds=30,
                    ),
                    bridge,
                )
                agent._knowledge = None
                agent._compressor.generate_memory = lambda *args, **kwargs: []

                events = list(agent.process("implement missing functionality"))
                self.assertTrue((root / "src" / "legacy.py").exists())
                self.assertIn("FULL-PASS MODE", calls[1]["messages"][-1]["content"])
                self.assertNotIn("OBSERVE -> ACT loop", calls[1]["messages"][-1]["content"])
                self.assertTrue(any(ev[0] == "header" and ev[1] == "coder" for ev in events))
            finally:
                os.chdir(old_cwd)


class TestStructuredWorkCompletion(unittest.TestCase):
    """A passing test command must close the current checklist item even when the
    plan was recovered from R1 thinking and parsed to bare titles (no file_refs /
    verify_cmd). Otherwise every item burns all its turns and prints a misleading
    'not verifiably finished' warning despite pytest exit=0 (observed 2026-06-17)."""

    class FakeAgent:
        def __init__(self, root, outputs, results, **cfg):
            self.config = KratosConfig(always_max_ctx=False, **cfg)
            self.pending_file_changes = []
            self.pending_file_deletions = []
            self._memory = MagicMock()
            self._knowledge = None
            self._outputs = list(outputs)
            self.command_results = list(results)
            self.prompts = []
            self._indexer = types.SimpleNamespace(root=root)

        def _is_cancelled(self):
            return False

        def _run_coder(self, msg):
            self.prompts.append(msg)
            out = self._outputs.pop(0) if self._outputs else "### VERIFY: python -m pytest test_mathx.py"
            yield ("coder", out, "text")
            return out

        def _run_verification_command(self, command):
            r = dict(self.command_results.pop(0)) if self.command_results else {
                "exit_code": 0, "duration_seconds": 0.01, "output": "ok"}
            r.update({"cmd": command.cmd, "purpose": command.purpose,
                      "source": command.source, "is_test": command.is_test})
            return r

    def test_passing_test_completes_bare_item_in_one_turn(self):
        from kratos.context import ContextPackage
        from kratos.roles.coder import execute_structured_work_steps_for_plan
        from kratos.planning import parse_execution_plan, plan_all_done

        plan = parse_execution_plan(
            "## Execution Order\n"
            "1. Implement `mathx.add` function in mathx.py.\n"
            "2. Run the test suite with `python -m pytest test_mathx.py`.\n"
            "3. Delete legacy_helper.py.\n"
        )
        self.assertTrue(all(not it.file_refs for it in plan.items))  # the bug precondition

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_mathx.py").write_text("def test_add():\n    assert True\n", encoding="utf-8")
            (root / "mathx.py").write_text("def add(a, b):\n    pass\n", encoding="utf-8")
            reg = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            outs = ["### FILE: mathx.py\n```python\ndef add(a, b):\n    return a + b\n```\n"
                    "### VERIFY: python -m pytest test_mathx.py\n"] * 3
            res = [{"exit_code": 0, "duration_seconds": 0.01, "output": "1 passed"}] * 3
            agent = self.FakeAgent(root, outs, res, permission="high", max_work_step_turns=4)
            proof = ProvenWork(iteration=1)
            events, _ = drain_generator(execute_structured_work_steps_for_plan(
                agent, "impl", plan,
                ContextPackage(user_input="t", intent="refactor", route="planner_then_coder"),
                reg, proof, 0, "", root, {},
            ))

            self.assertTrue(plan_all_done(plan))
            self.assertEqual([it.index for it in plan.items if it.status == "done"], [1, 2, 3])
            self.assertFalse(any("not verifiably finished" in str(c) for _, c, _ in events))
            # one turn per item (3), not 4 turns x 3 items = 12
            self.assertEqual(len(agent.prompts), 3)


class TestProvenWork(unittest.TestCase):
    def test_safe_command_filter_rejects_shell_chains(self):
        self.assertTrue(_is_safe_verification_command("python -m pytest tests"))
        self.assertTrue(_is_safe_verification_command("dotnet run --project tests/TaskBoard.Tests"))
        self.assertFalse(_is_safe_verification_command("python -m pytest tests && del important.txt"))
        self.assertFalse(_is_safe_verification_command("echo hello"))

    def test_safe_inspect_filter_allows_read_only_shell_diagnostics(self):
        self.assertTrue(_is_safe_inspect_command('rg -n "TODO" src'))
        self.assertTrue(_is_safe_inspect_command("Get-Content kratos/knowledge/base.py | Select-Object -First 3"))
        self.assertTrue(_is_safe_inspect_command("git diff -- kratos/core/agent.py"))
        self.assertFalse(_is_safe_inspect_command("Set-Content src/app.py x"))
        self.assertFalse(_is_safe_inspect_command("Get-Content ..\\secret.txt"))

    def test_local_python_smoke_run_is_safe_when_script_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
            self.assertTrue(_is_safe_verification_command("python main.py", root))

    def test_local_python_smoke_run_executes_through_command_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
            proof = ProvenWork(iteration=1)

            class FakeAgent:
                def __init__(self):
                    self.calls = 0

                def _run_verification_command(self, command):
                    self.calls += 1
                    return {
                        "cmd": command.cmd,
                        "purpose": command.purpose,
                        "source": command.source,
                        "is_test": command.is_test,
                        "exit_code": 0,
                        "duration_seconds": 0.01,
                        "output": "ok",
                    }

            agent = FakeAgent()
            registry = CommandRegistry(KratosConfig(always_max_ctx=False), root).discover()
            events, obs = drain_generator(do_command(agent, root, registry, "python main.py", proof))
            self.assertTrue(obs["ok"])
            self.assertEqual(agent.calls, 1)
            self.assertTrue(any(ev[0] == "tool" for ev in events))
            self.assertEqual(proof.commands[-1]["cmd"], "python main.py")

    def test_missing_command_paths_flags_nonexistent_files_only(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_existing.py").write_text("x", encoding="utf-8")

            self.assertEqual(
                _missing_command_paths("python -m pytest tests/test_todo_store.py", root),
                ["tests/test_todo_store.py"],
            )
            self.assertEqual(
                _missing_command_paths("python -m pytest tests/test_existing.py", root), [],
            )
            # bare directory target (no file extension) is left alone
            self.assertEqual(_missing_command_paths("python -m pytest tests", root), [])

    def test_test_command_detection(self):
        self.assertTrue(_is_test_verification_command("python -m pytest tests"))
        self.assertTrue(_is_test_verification_command("dotnet run --project tests/TaskBoard.Tests"))
        self.assertTrue(_is_test_verification_command("npm test"))
        self.assertFalse(_is_test_verification_command("dotnet build src/App.csproj"))

    def test_readme_command_extraction_filters_runtime_examples(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("""```powershell
dotnet build
dotnet run --project tests/TaskBoard.Tests
dotnet run --project src/TaskBoard.Cli -- data/sample.txt list
```
""", encoding="utf-8")

            commands = _extract_readme_verification_commands(root)
            cmd_text = [item.cmd for item in commands]

            # No .sln/.csproj at root → bare "dotnet build" must be filtered out
            # (it would fail with MSB1003 when cwd has no project file).
            self.assertNotIn("dotnet build", cmd_text)
            # Explicit-path "dotnet run --project tests/..." is kept (valid + is_test)
            self.assertIn("dotnet run --project tests/TaskBoard.Tests", cmd_text)
            # CLI smoke commands are safe verification commands when they use
            # dotnet's explicit --project form and contain no shell metacharacters.
            self.assertIn("dotnet run --project src/TaskBoard.Cli -- data/sample.txt list", cmd_text)
            self.assertFalse([
                item for item in commands
                if item.cmd == "dotnet run --project src/TaskBoard.Cli -- data/sample.txt list"
            ][0].is_test)

    def test_readme_command_extraction_keeps_bare_dotnet_build_when_root_has_project(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Place a .sln at root — bare "dotnet build" is now valid from this dir
            (root / "MyApp.sln").write_text("", encoding="utf-8")
            (root / "README.md").write_text("""```\ndotnet build\ndotnet test\n```\n""", encoding="utf-8")

            commands = _extract_readme_verification_commands(root)
            cmd_text = [item.cmd for item in commands]

            self.assertIn("dotnet build", cmd_text)
            self.assertIn("dotnet test", cmd_text)

    def test_dotnet_executable_test_project_inference(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "tests" / "TaskBoard.Tests"
            test_dir.mkdir(parents=True)
            (test_dir / "TaskBoard.Tests.csproj").write_text("""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
</Project>
""", encoding="utf-8")

            commands = _infer_project_verification_commands(root)
            cmd_text = [item.cmd for item in commands]

            self.assertIn("dotnet build tests/TaskBoard.Tests/TaskBoard.Tests.csproj", cmd_text)
            self.assertIn("dotnet run --project tests/TaskBoard.Tests/TaskBoard.Tests.csproj", cmd_text)
            self.assertTrue(any(item.is_test for item in commands))

    def test_proven_work_requires_successful_test(self):
        proof = ProvenWork(iteration=1)
        proof.commands.append({"cmd": "dotnet build", "exit_code": 0, "is_test": False})
        self.assertFalse(_proven_work_satisfied(proof, require_test=True))

        proof.commands.append({"cmd": "python -m pytest tests", "exit_code": 0, "is_test": True})
        self.assertTrue(_proven_work_satisfied(proof, require_test=True))

        proof.commands.append({"cmd": "npm test", "exit_code": 1, "is_test": True})
        self.assertFalse(_proven_work_satisfied(proof, require_test=True))


# ── KratosConfig ──────────────────────────────────────────────────────────────

class TestKratosConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = KratosConfig()
        self.assertEqual(cfg.permission, "mid")
        # Configured ceilings are the model maximums; the EFFECTIVE window per
        # call is capped by vram_ctx_ceiling (see TestEffectiveContextWindows).
        self.assertEqual(cfg.planner_num_ctx,    131072)
        self.assertEqual(cfg.coder_num_ctx,      32768)
        self.assertEqual(cfg.verifier_num_ctx,   131072)
        self.assertEqual(cfg.compressor_num_ctx, 32768)
        # 6 GB-safe KV-cache ceiling — the knob that prevents the planner stall.
        self.assertEqual(cfg.vram_ctx_ceiling,   8192)
        self.assertTrue(getattr(cfg, "always_max_ctx", True))
        self.assertTrue(getattr(cfg, "deterministic_verify", True))
        self.assertEqual(cfg.verifier_model, cfg.planner_model)
        self.assertIsNotNone(cfg.compressor_model)
        self.assertTrue(cfg.require_proven_work)
        self.assertTrue(cfg.require_test_for_verified)

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
                # explicit small + always_max=False → roundtrip should keep the value
                c = KratosConfig(permission="high", planner_num_ctx=4096, always_max_ctx=False)
                c.save("project")
                loaded = KratosConfig.load()
                self.assertEqual(loaded.permission, "high")
                self.assertEqual(loaded.planner_num_ctx, 4096)
                self.assertFalse(loaded.always_max_ctx)
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

    def test_implement_missing_functionality_is_coding(self):
        text = "implement all missing CLI functionality and make the tests pass"
        self.assertEqual(self._classify(text), Intent.CODING)

    def test_bugfix_intent(self):
        self.assertEqual(self._classify("fix the broken auth flow"), Intent.BUGFIX)

    def test_question_intent(self):
        self.assertIn(self._classify("what does this function do?"),
                      (Intent.QUESTION, Intent.EXPLAIN))

    def test_file_search_short(self):
        self.assertEqual(self._classify("where is config.py"), Intent.FILE_SEARCH)

    def test_followup(self):
        self.assertEqual(self._classify("continue"), Intent.FOLLOWUP)

    def test_long_task_with_reset_word_still_uses_planner(self):
        text = """
TASK: Complete and clean up the toolkit in the fullcheck directory so the
whole test suite passes.

```
Implement slugify, is_palindrome, median, gcd and fix the mean bug.
Do not modify tests.
```

Reset: to repeat this full check, restore the starter files and run it again.
German note: um den Full-Check zu wiederholen, stelle die Dateien wieder her,
for example with git checkout.
"""
        self.assertNotEqual(self._classify(text), Intent.FOLLOWUP)
        self.assertNotEqual(self._classify(text), Intent.SHELL_GIT)
        self.assertEqual(self._route(text), Route.PLANNER_THEN_CODER)

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


# ── PromptManager ────────────────────────────────────────────────────────────

class TestPromptManager(unittest.TestCase):
    def setUp(self):
        # Fresh manager using only package defaults (no on-disk overrides in test)
        self._pm = PromptManager.__new__(PromptManager)
        from kratos.prompts import _DEFAULT_JSON
        self._pm._default_path  = _DEFAULT_JSON
        self._pm._global_path   = Path("/nonexistent/global/prompts.json")
        self._pm._project_path  = Path("/nonexistent/project/prompts.json")
        self._pm._effective     = dict(DEFAULT_PROMPTS)

    def test_get_system_planner(self):
        s = self._pm.get_system("planner")
        self.assertIn("Kratos Planner", s)
        self.assertIn("CHECKLIST", s)
        self.assertIn("VERIFY", s)

    def test_get_system_coder(self):
        s = self._pm.get_system("coder")
        self.assertIn("Kratos Coder", s)
        self.assertIn("### FILE:", s)
        self.assertIn("### STEP_VERIFY:", s)

    def test_get_system_verifier(self):
        s = self._pm.get_system("verifier")
        self.assertIn("Kratos Verifier", s)
        self.assertIn("PROVEN_WORK", s)
        self.assertIn("VERIFIED", s)
        self.assertIn("NEEDS_REVISION", s)
        self.assertIn("UNSOLVABLE", s)

    def test_verify_msg_includes_planner_checklist_audit(self):
        proof = ProvenWork(iteration=1)
        msg = _verify_msg(
            "fix the bug",
            "## CHECKLIST\n- One\n- Two",
            "coder output",
            proof,
            [
                PlanItem(index=1, title="One", status="pending"),
                PlanItem(index=2, title="Two", status="done"),
            ],
        )
        self.assertIn("Planner checklist audit", msg)
        self.assertIn("One", msg)
        self.assertIn("Two", msg)

    def test_get_system_compress(self):
        s = self._pm.get_system("compress")
        self.assertIn("LOSSLESS", s)

    def test_get_system_memory(self):
        s = self._pm.get_system("memory")
        self.assertIn("JSON", s)
        self.assertIn("category", s)

    def test_get_system_relay(self):
        s = self._pm.get_system("relay")
        self.assertTrue(len(s) > 10)

    def test_get_system_unknown_returns_empty(self):
        s = self._pm.get_system("doesnotexist_role_xyz")
        self.assertEqual(s, "")

    def test_get_snippet_file_marker(self):
        s = self._pm.get_snippet("file_marker")
        self.assertEqual(s, "### FILE:")

    def test_get_snippet_step_verify_marker(self):
        s = self._pm.get_snippet("step_verify_marker")
        self.assertEqual(s, "### STEP_VERIFY:")

    def test_get_snippet_missing_returns_empty(self):
        s = self._pm.get_snippet("no_such_snippet_xyz")
        self.assertEqual(s, "")

    def test_get_predict_defaults(self):
        # predict.plan capped at 2048 to bound DeepSeek-R1 thinking; the planner's
        # _plan_from_thinking fallback salvages a plan if the budget is hit.
        self.assertEqual(self._pm.get_predict("plan"), 2048)
        self.assertEqual(self._pm.get_predict("code"), 16384)
        self.assertEqual(self._pm.get_predict("verify"), 512)
        self.assertEqual(self._pm.get_predict("relay"), 1200)

    def test_get_predict_missing_returns_1024(self):
        self.assertEqual(self._pm.get_predict("no_such_key"), 1024)

    def test_get_marker_file(self):
        self.assertEqual(self._pm.get_marker("file"), "### FILE:")

    def test_get_marker_step_verify(self):
        self.assertEqual(self._pm.get_marker("step_verify"), "### STEP_VERIFY:")

    def test_get_all_returns_copy(self):
        all_p = self._pm.get_all()
        self.assertIsInstance(all_p, dict)
        all_p["planner_system"] = "mutated"
        self.assertNotEqual(self._pm.get_system("planner"), "mutated")

    def test_dump_defaults(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = Path(f.name)
        try:
            self._pm.dump_defaults(tmp)
            data = json.loads(tmp.read_text(encoding="utf-8"))
            self.assertIn("planner_system", data)
            self.assertIn("coder_system", data)
            self.assertIn("verifier_system", data)
            self.assertIn("snippets", data)
            self.assertIn("predict", data)
            self.assertIn("markers", data)
        finally:
            os.unlink(tmp)

    def test_json_override_merges_snippets(self):
        import tempfile, os
        from kratos.prompts import _DEFAULT_JSON
        override = {"snippets": {"task_label": "TASK: "}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(override, f)
            tmp = Path(f.name)
        try:
            pm = PromptManager.__new__(PromptManager)
            pm._default_path  = _DEFAULT_JSON
            pm._global_path   = tmp
            pm._project_path  = Path("/nonexistent/project/prompts.json")
            pm._effective     = {}
            pm._load_all()
            self.assertEqual(pm.get_snippet("task_label"), "TASK: ")
            # Other snippets must be untouched
            self.assertEqual(pm.get_snippet("file_marker"), "### FILE:")
        finally:
            os.unlink(tmp)

    def test_json_override_replaces_system(self):
        import tempfile, os
        from kratos.prompts import _DEFAULT_JSON
        override = {"planner_system": "Custom planner system"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(override, f)
            tmp = Path(f.name)
        try:
            pm = PromptManager.__new__(PromptManager)
            pm._default_path  = _DEFAULT_JSON
            pm._global_path   = tmp
            pm._project_path  = Path("/nonexistent/project/prompts.json")
            pm._effective     = {}
            pm._load_all()
            self.assertEqual(pm.get_system("planner"), "Custom planner system")
            # Other roles must be untouched
            self.assertIn("Kratos Coder", pm.get_system("coder"))
        finally:
            os.unlink(tmp)

    def test_reload_restores_defaults(self):
        from kratos.prompts import _DEFAULT_JSON
        pm = PromptManager.__new__(PromptManager)
        pm._default_path  = _DEFAULT_JSON
        pm._global_path   = Path("/nonexistent/global/prompts.json")
        pm._project_path  = Path("/nonexistent/project/prompts.json")
        pm._effective     = {}
        pm._load_all()
        pm._effective["planner_system"] = "temporary mutation"
        pm.reload()
        self.assertIn("Kratos Planner", pm.get_system("planner"))

    def test_module_level_get_system(self):
        s = get_system("planner")
        self.assertIn("Kratos Planner", s)

    def test_module_level_get_snippet(self):
        s = get_snippet("file_marker")
        self.assertEqual(s, "### FILE:")

    def test_module_level_get_predict(self):
        self.assertEqual(get_predict("code"), 16384)

    def test_module_level_get_marker(self):
        self.assertEqual(get_marker("file"), "### FILE:")

    def test_default_prompts_all_required_keys_present(self):
        required_systems = [
            "planner_system", "coder_system", "verifier_system",
            "compress_system", "memory_system", "relay_system",
        ]
        for key in required_systems:
            self.assertIn(key, DEFAULT_PROMPTS, f"Missing required key: {key}")
            self.assertIsInstance(DEFAULT_PROMPTS[key], str)
            self.assertGreater(len(DEFAULT_PROMPTS[key]), 20)

    def test_default_prompts_markers_consistent(self):
        markers = DEFAULT_PROMPTS.get("markers", {})
        snippets = DEFAULT_PROMPTS.get("snippets", {})
        self.assertEqual(markers.get("file"), snippets.get("file_marker"))
        self.assertEqual(markers.get("step_verify"), snippets.get("step_verify_marker"))

    def test_planner_prompt_no_pytest_bias(self):
        """planner_system must not encourage pytest for non-Python projects."""
        # Check DEFAULT_PROMPTS (source of truth) not the merged effective prompts
        p = DEFAULT_PROMPTS.get("planner_system", "")
        # Old biased phrase must be gone from the defaults
        self.assertNotIn("prefer pytest, cargo test", p)
        # New toolchain-neutral instruction must be present
        self.assertIn("project's real", p.lower())

    def test_planner_prompt_uses_markdown_template_with_checklist(self):
        p = DEFAULT_PROMPTS.get("planner_system", "")
        self.assertIn("## Summary", p)
        self.assertIn("## Key Changes", p)
        self.assertIn("## Test Plan", p)
        self.assertIn("## Assumptions", p)
        self.assertIn("## CHECKLIST", p)
        self.assertIn("File:", p)
        self.assertIn("VERIFY:", p)

    def test_coder_prompt_no_pytest_example(self):
        """coder_system must not use pytest as the canonical STEP_VERIFY example."""
        # Check DEFAULT_PROMPTS (source of truth)
        p = DEFAULT_PROMPTS.get("coder_system", "")
        self.assertNotIn("### STEP_VERIFY: pytest -q", p)
        # Must mention the toolchain-aware rule
        self.assertIn("toolchain", p.lower())


# ── Toolchain detection ───────────────────────────────────────────────────────

class TestToolchainDetection(unittest.TestCase):

    def _tmp(self, files: list[str]) -> Path:
        import tempfile, os
        d = Path(tempfile.mkdtemp())
        for f in files:
            p = d / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        return d

    def test_detects_dotnet_via_csproj(self):
        d = self._tmp(["src/MyApp/MyApp.csproj"])
        result = _detect_project_toolchains(d)
        self.assertIn("dotnet", result)
        self.assertNotIn("python", result)

    def test_detects_dotnet_via_sln(self):
        d = self._tmp(["MySolution.sln"])
        result = _detect_project_toolchains(d)
        self.assertIn("dotnet", result)

    def test_detects_python_via_pyproject(self):
        d = self._tmp(["pyproject.toml", "tests/test_x.py"])
        result = _detect_project_toolchains(d)
        self.assertIn("python", result)
        self.assertNotIn("dotnet", result)

    def test_detects_python_via_tests_dir(self):
        d = self._tmp(["tests/__init__.py"])
        result = _detect_project_toolchains(d)
        self.assertIn("python", result)

    def test_detects_node(self):
        d = self._tmp(["package.json"])
        result = _detect_project_toolchains(d)
        self.assertIn("node", result)

    def test_detects_cargo(self):
        d = self._tmp(["Cargo.toml"])
        result = _detect_project_toolchains(d)
        self.assertIn("cargo", result)

    def test_detects_go(self):
        d = self._tmp(["go.mod"])
        result = _detect_project_toolchains(d)
        self.assertIn("go", result)

    def test_empty_dir_returns_empty_set(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        result = _detect_project_toolchains(d)
        self.assertEqual(result, set())

    def test_mixed_project_detects_both(self):
        d = self._tmp(["package.json", "src/App.csproj"])
        result = _detect_project_toolchains(d)
        self.assertIn("dotnet", result)
        self.assertIn("node", result)


class TestCommandToolchain(unittest.TestCase):

    def test_pytest(self):
        self.assertEqual(_command_toolchain("pytest -q -k SomeTest"), "python")

    def test_python_m_pytest(self):
        self.assertEqual(_command_toolchain("python -m pytest tests/"), "python")

    def test_python_unittest(self):
        self.assertEqual(_command_toolchain("python -m unittest discover"), "python")

    def test_dotnet_test(self):
        self.assertEqual(_command_toolchain("dotnet test"), "dotnet")

    def test_dotnet_run(self):
        self.assertEqual(_command_toolchain("dotnet run --project tests/MyApp.Tests"), "dotnet")

    def test_dotnet_build(self):
        self.assertEqual(_command_toolchain("dotnet build src/App.csproj"), "dotnet")

    def test_npm_test(self):
        self.assertEqual(_command_toolchain("npm test"), "node")

    def test_yarn_test(self):
        self.assertEqual(_command_toolchain("yarn test"), "node")

    def test_cargo_test(self):
        self.assertEqual(_command_toolchain("cargo test"), "cargo")

    def test_go_test(self):
        self.assertEqual(_command_toolchain("go test ./..."), "go")

    def test_unknown_returns_none(self):
        self.assertIsNone(_command_toolchain("make all"))

    def test_empty_returns_none(self):
        self.assertIsNone(_command_toolchain(""))


class TestToolchainMismatchGuard(unittest.TestCase):
    """Verify that a pytest command is rejected when only dotnet is detected."""

    def test_pytest_rejected_for_dotnet_project(self):
        """The guard logic: cmd_tc not in _project_toolchains → reject."""
        project_toolchains = {"dotnet"}
        raw_cmd = "pytest -q -k TaskFormatterTests"
        cmd_tc = _command_toolchain(raw_cmd)
        tc_mismatch = bool(
            project_toolchains and cmd_tc and cmd_tc not in project_toolchains
        )
        self.assertTrue(tc_mismatch,
                        "pytest should be flagged as a toolchain mismatch for a dotnet project")

    def test_dotnet_accepted_for_dotnet_project(self):
        project_toolchains = {"dotnet"}
        raw_cmd = "dotnet run --project tests/TaskBoard.Tests"
        cmd_tc = _command_toolchain(raw_cmd)
        tc_mismatch = bool(
            project_toolchains and cmd_tc and cmd_tc not in project_toolchains
        )
        self.assertFalse(tc_mismatch)

    def test_pytest_accepted_for_python_project(self):
        project_toolchains = {"python"}
        raw_cmd = "pytest -q tests/"
        cmd_tc = _command_toolchain(raw_cmd)
        tc_mismatch = bool(
            project_toolchains and cmd_tc and cmd_tc not in project_toolchains
        )
        self.assertFalse(tc_mismatch)

    def test_empty_toolchain_set_never_rejects(self):
        """If toolchain detection found nothing, we must not block any command."""
        project_toolchains: set[str] = set()
        raw_cmd = "pytest -q"
        cmd_tc = _command_toolchain(raw_cmd)
        tc_mismatch = bool(
            project_toolchains and cmd_tc and cmd_tc not in project_toolchains
        )
        self.assertFalse(tc_mismatch,
                         "Empty toolchain set should never cause a mismatch reject")


# ── Coder context block ───────────────────────────────────────────────────────

class TestCoderContextBlock(unittest.TestCase):

    def _make_ctx(self, files: list[dict]) -> object:
        """Build a minimal ContextPackage directly (no InputAnalysis needed)."""
        from kratos.context import ContextPackage, FileEntry
        pkg = ContextPackage(
            user_input="test", intent="coding", route="planner_then_coder",
        )
        for f in files:
            content = f.get("content", "")
            fe = FileEntry(
                path=Path(f["rel_path"]),
                rel_path=f["rel_path"],
                priority=f.get("priority", 5),
                size=len(content),
                content=content,
            )
            pkg.files.append(fe)
        return pkg

    def _make_pm(self):
        pm = load_prompts()
        return pm

    def test_empty_ctx_returns_empty(self):
        ctx = self._make_ctx([])
        result = _coder_context_block(ctx, self._make_pm())
        self.assertEqual(result, "")

    def test_test_file_detected(self):
        ctx = self._make_ctx([
            {"rel_path": "tests/TaskFormatterTests.cs", "content": "public class TaskFormatterTests {}"},
        ])
        result = _coder_context_block(ctx, self._make_pm(), step_mode=True)
        self.assertIn("TaskFormatterTests.cs", result)
        self.assertIn("TEST FILES", result)

    def test_stub_file_step_mode_detects_todo(self):
        ctx = self._make_ctx([
            {"rel_path": "src/TaskParser.cs", "content": "// TODO: implement\npublic class TaskParser {}"},
        ])
        result = _coder_context_block(ctx, self._make_pm(), step_mode=True)
        self.assertIn("TaskParser.cs", result)
        self.assertIn("STUB FILES", result)

    def test_stub_file_no_step_mode_uses_notimplementederror(self):
        """Without step_mode, // TODO does NOT count as a stub."""
        ctx = self._make_ctx([
            {"rel_path": "src/TaskParser.cs", "content": "// TODO: implement\npublic class TaskParser {}"},
        ])
        result = _coder_context_block(ctx, self._make_pm(), step_mode=False)
        # File should appear as done_file, not stub
        self.assertNotIn("STUB FILES", result)

    def test_noise_files_excluded(self):
        ctx = self._make_ctx([
            {"rel_path": "README.md", "content": "# readme"},
            {"rel_path": "src/App.cs", "content": "public class App {}"},
        ])
        result = _coder_context_block(ctx, self._make_pm())
        self.assertNotIn("README.md", result)
        self.assertIn("App.cs", result)


class TestExtractPlanSteps(unittest.TestCase):
    """_extract_plan_steps only picks up numbered items, never bullets."""

    _REALISTIC_PLAN = (
        "What exactly needs to be done:\n"
        "  Implement missing stubs.\n\n"
        "Which files are relevant:\n"
        "  - src/TaskBoard.Cli/Services/TaskFormatter.cs\n"
        "  - src/TaskBoard.Cli/Services/TaskParser.cs\n\n"
        "STEP-BY-STEP CHECKLIST:\n"
        "  1. Implement FormatList  File: src/TaskFormatter.cs  STEP_VERIFY: `dotnet run --project tests/TaskBoard.Tests`\n"
        "  2. Implement FormatStats  File: src/TaskFormatter.cs  STEP_VERIFY: `dotnet run --project tests/TaskBoard.Tests`\n"
        "  3. Implement ParseLine  File: src/TaskParser.cs  STEP_VERIFY: `dotnet run --project tests/TaskBoard.Tests`\n\n"
        "Final verification: dotnet run --project tests/TaskBoard.Tests\n\n"
        "Potential risks:\n"
        "  - Circular imports could occur\n"
        "  - API mismatches possible\n"
    )

    def test_picks_only_numbered(self):
        steps = _extract_plan_steps(self._REALISTIC_PLAN)
        self.assertEqual(len(steps), 3, f"Expected 3 numbered steps, got {len(steps)}: {steps}")

    def test_excludes_file_path_bullets(self):
        steps = _extract_plan_steps(self._REALISTIC_PLAN)
        for s in steps:
            self.assertNotIn("TaskFormatter.cs\n", s)
            self.assertNotIn("TaskParser.cs\n", s)

    def test_excludes_risk_bullets(self):
        steps = _extract_plan_steps(self._REALISTIC_PLAN)
        for s in steps:
            self.assertNotIn("Circular imports", s)
            self.assertNotIn("API mismatches", s)

    def test_step_content_is_action(self):
        steps = _extract_plan_steps(self._REALISTIC_PLAN)
        self.assertIn("Implement FormatList", steps[0])
        self.assertIn("Implement FormatStats", steps[1])
        self.assertIn("Implement ParseLine", steps[2])

    def test_fallback_on_no_numbered(self):
        plan = "Edit the formatter to add overdue support.\n\nChange the parser to validate columns."
        steps = _extract_plan_steps(plan)
        self.assertGreater(len(steps), 0)

    def test_empty_plan(self):
        self.assertEqual(_extract_plan_steps(""), [])

    def test_step_n_colon_format(self):
        plan = "Step 1: Implement formatter\nStep 2: Implement parser"
        steps = _extract_plan_steps(plan)
        self.assertEqual(len(steps), 2)


class TestCompileCheckCmds(unittest.TestCase):
    """_compile_check_cmds returns language-appropriate compile commands."""

    def test_dotnet_returns_build(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            cmds = _compile_check_cmds({"dotnet"}, Path(d))
        self.assertEqual(len(cmds), 1)
        self.assertIn("dotnet build", cmds[0].cmd)
        self.assertFalse(cmds[0].is_test)
        self.assertEqual(cmds[0].source, "auto-compile")

    def test_dotnet_nested_project_returns_explicit_build_target(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            test_dir = root / "tests" / "TaskBoard.Tests"
            test_dir.mkdir(parents=True)
            (test_dir / "TaskBoard.Tests.csproj").write_text("<Project/>", encoding="utf-8")

            cmds = _compile_check_cmds({"dotnet"}, root)

        self.assertEqual(len(cmds), 1)
        self.assertEqual(
            cmds[0].cmd,
            "dotnet build tests/TaskBoard.Tests/TaskBoard.Tests.csproj --nologo -v:minimal",
        )
        self.assertFalse(cmds[0].is_test)

    def test_python_returns_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cmds = _compile_check_cmds({"python"}, Path(d))
        self.assertEqual(cmds, [])

    def test_node_no_tsconfig_returns_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cmds = _compile_check_cmds({"node"}, Path(d))
        self.assertEqual(cmds, [])

    def test_node_with_tsconfig_returns_tsc(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "tsconfig.json").write_text("{}")
            cmds = _compile_check_cmds({"node"}, Path(d))
        self.assertEqual(len(cmds), 1)
        self.assertIn("tsc", cmds[0].cmd)
        self.assertFalse(cmds[0].is_test)

    def test_empty_toolchains_returns_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cmds = _compile_check_cmds(set(), Path(d))
        self.assertEqual(cmds, [])

    def test_mixed_dotnet_node_tsconfig(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            Path(d, "tsconfig.json").write_text("{}")
            cmds = _compile_check_cmds({"dotnet", "node"}, Path(d))
        self.assertEqual(len(cmds), 2)
        cmd_strs = {c.cmd for c in cmds}
        self.assertTrue(any("dotnet" in c for c in cmd_strs))
        self.assertTrue(any("tsc" in c for c in cmd_strs))


class TestExtractStepFileRefs(unittest.TestCase):
    """_extract_step_file_refs pulls file paths out of a step description."""

    def test_file_colon_prefix(self):
        refs = _extract_step_file_refs("Implement formatter  File: src/Foo.cs  STEP_VERIFY: dotnet test")
        self.assertIn("src/Foo.cs", refs)

    def test_backtick_path(self):
        refs = _extract_step_file_refs("Edit `src/Services/TaskParser.cs` to add validation")
        self.assertIn("src/Services/TaskParser.cs", refs)

    def test_quoted_path(self):
        refs = _extract_step_file_refs('Modify "tests/TaskBoard.Tests/Foo.cs" to check X')
        self.assertIn("tests/TaskBoard.Tests/Foo.cs", refs)

    def test_no_refs(self):
        refs = _extract_step_file_refs("No file paths here at all, just words.")
        self.assertEqual(refs, [])

    def test_capped_at_six(self):
        text = " ".join(f"`src/File{i}.cs`" for i in range(10))
        refs = _extract_step_file_refs(text)
        self.assertLessEqual(len(refs), 6)

    def test_duplicate_deduplicated(self):
        refs = _extract_step_file_refs("`src/Foo.cs` and `src/Foo.cs` again")
        self.assertEqual(refs.count("src/Foo.cs"), 1)


class TestParseStepTests(unittest.TestCase):
    """_parse_step_tests extracts ### STEP_TEST: blocks from coder output."""

    _SAMPLE = (
        "### FILE: src/Foo.cs\n```csharp\npublic class Foo {}\n```\n"
        "### SUMMARY\nStep 1 done\n"
        "### STEP_VERIFY: dotnet run --project tests/T.csproj\n"
        "### STEP_TEST: tests/_KratosStep1Test.cs\n"
        "```csharp\ninternal static class _KratosStep1Test {\n"
        "    public static void RunAll() { Console.WriteLine(\"PASS\"); }\n"
        "}\n```\n"
    )

    def test_extracts_path(self):
        tests = _parse_step_tests(self._SAMPLE)
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0][0], "tests/_KratosStep1Test.cs")

    def test_extracts_content(self):
        tests = _parse_step_tests(self._SAMPLE)
        self.assertIn("_KratosStep1Test", tests[0][1])
        self.assertIn("RunAll", tests[0][1])

    def test_no_step_test_returns_empty(self):
        text = "### FILE: src/Foo.cs\n```csharp\n// code\n```\n### SUMMARY\nDone\n### STEP_VERIFY: dotnet run\n"
        self.assertEqual(_parse_step_tests(text), [])

    def test_multiple_blocks(self):
        text = (
            "### STEP_TEST: tests/_KratosStep1Test.cs\n```csharp\nclass T1 {}\n```\n"
            "### STEP_TEST: tests/_KratosStep2Test.cs\n```csharp\nclass T2 {}\n```\n"
        )
        tests = _parse_step_tests(text)
        self.assertEqual(len(tests), 2)
        self.assertEqual(tests[0][0], "tests/_KratosStep1Test.cs")
        self.assertEqual(tests[1][0], "tests/_KratosStep2Test.cs")


class TestPatchDotnetTestRunner(unittest.TestCase):
    """_patch_dotnet_test_runner inserts a RunAll() call into Program.cs."""

    def setUp(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_inserts_after_last_run_all(self):
        prog = self._tmp / "Program.cs"
        prog.write_text("FooTests.RunAll();\nBarTests.RunAll();\n")
        prog_path, original = _patch_dotnet_test_runner(self._tmp, "_KratosStep1Test")
        content = prog_path.read_text()
        self.assertIn("_KratosStep1Test.RunAll();", content)
        # Must appear AFTER the last existing RunAll
        last_bar = content.rfind("BarTests.RunAll()")
        kratos_pos = content.find("_KratosStep1Test.RunAll()")
        self.assertGreater(kratos_pos, last_bar)

    def test_original_content_preserved(self):
        prog = self._tmp / "Program.cs"
        prog.write_text("FooTests.RunAll();\n")
        _, original = _patch_dotnet_test_runner(self._tmp, "_KratosStep1Test")
        self.assertIn("FooTests.RunAll();", original)

    def test_restores_correctly(self):
        prog = self._tmp / "Program.cs"
        orig_text = "FooTests.RunAll();\n"
        prog.write_text(orig_text)
        prog_path, original = _patch_dotnet_test_runner(self._tmp, "_KratosStep1Test")
        prog_path.write_text(original, "utf-8")
        self.assertEqual(prog_path.read_text(), orig_text)

    def test_returns_none_if_no_program_cs(self):
        prog_path, original = _patch_dotnet_test_runner(self._tmp, "_KratosStep1Test")
        self.assertIsNone(prog_path)
        self.assertIsNone(original)

    def test_appends_when_no_existing_run_all(self):
        prog = self._tmp / "Program.cs"
        prog.write_text("// No RunAll calls here\n")
        prog_path, _ = _patch_dotnet_test_runner(self._tmp, "_KratosStep1Test")
        self.assertIn("_KratosStep1Test.RunAll();", prog_path.read_text())


class TestCommandRegistry(unittest.TestCase):
    """CommandRegistry discovers project commands and generates formatted output."""

    def setUp(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _fake_config(self, build_cmd=None, test_cmd=None):
        class FakeConfig:
            pass
        c = FakeConfig()
        c.build_cmd = build_cmd
        c.test_cmd = test_cmd
        return c

    def test_detects_dotnet_toolchain(self):
        (self._tmp / "MyApp.sln").write_text("")
        reg = CommandRegistry(self._fake_config(), self._tmp).discover()
        self.assertIn("dotnet", reg.toolchains)

    def test_compile_commands_dotnet(self):
        (self._tmp / "MyApp.sln").write_text("")
        reg = CommandRegistry(self._fake_config(), self._tmp).discover()
        self.assertTrue(any("dotnet build" in c.cmd for c in reg.compile_commands))

    def test_verify_hint_contains_toolchain(self):
        (self._tmp / "MyApp.sln").write_text("")
        reg = CommandRegistry(self._fake_config(), self._tmp).discover()
        hint = reg.verify_hint()
        self.assertIn("dotnet", hint.lower())

    def test_format_for_prompt_has_command_registry_header(self):
        (self._tmp / "MyApp.sln").write_text("")
        reg = CommandRegistry(self._fake_config(), self._tmp).discover()
        fmt = reg.format_for_prompt()
        self.assertIn("COMMAND REGISTRY", fmt)

    def test_is_toolchain_mismatch_pytest_on_dotnet(self):
        (self._tmp / "MyApp.sln").write_text("")
        reg = CommandRegistry(self._fake_config(), self._tmp).discover()
        self.assertTrue(reg.is_toolchain_mismatch("pytest tests/"))

    def test_is_toolchain_mismatch_dotnet_on_dotnet(self):
        (self._tmp / "MyApp.sln").write_text("")
        reg = CommandRegistry(self._fake_config(), self._tmp).discover()
        self.assertFalse(reg.is_toolchain_mismatch("dotnet run --project tests/T.csproj"))

    def test_configured_commands_take_priority(self):
        reg = CommandRegistry(
            self._fake_config(test_cmd="python -m pytest tests/"), self._tmp
        ).discover()
        test_cmds = reg.test_commands()
        self.assertTrue(any("pytest" in c.cmd for c in test_cmds))

    def test_empty_dir_gives_empty_commands(self):
        reg = CommandRegistry(self._fake_config(), self._tmp).discover()
        self.assertEqual(reg.commands, [])
        self.assertEqual(reg.compile_commands, [])
        hint = reg.verify_hint()
        self.assertEqual(hint, "")


class TestPromptManagerNewMethods(unittest.TestCase):
    """Test get_toolchain and get_plan_config on PromptManager."""

    def setUp(self):
        from kratos.prompts import _DEFAULT_JSON
        self._pm = PromptManager.__new__(PromptManager)
        self._pm._default_path = _DEFAULT_JSON
        self._pm._global_path = Path("/nonexistent/global.json")
        self._pm._project_path = Path("/nonexistent/project.json")
        self._pm._effective = {}
        self._pm._load_all()

    def test_get_toolchain_safe_verify_prefixes(self):
        prefixes = self._pm.get_toolchain("safe_verify_prefixes")
        self.assertIsInstance(prefixes, list)
        self.assertIn("pytest", prefixes)
        self.assertIn("dotnet run --project", prefixes)

    def test_get_toolchain_blocked_chars(self):
        chars = self._pm.get_toolchain("blocked_verify_chars")
        self.assertIn("&&", chars)
        self.assertIn(";", chars)

    def test_get_toolchain_compile_commands(self):
        cmds = self._pm.get_toolchain("compile_commands")
        self.assertIsInstance(cmds, dict)
        self.assertIn("dotnet", cmds)
        self.assertIn("dotnet build", cmds["dotnet"])

    def test_get_toolchain_step_test_marker(self):
        marker = self._pm.get_toolchain("step_test_marker")
        self.assertEqual(marker, "### STEP_TEST:")

    def test_get_plan_config_step_regex(self):
        regex = self._pm.get_plan_config("step_regex")
        self.assertIsNotNone(regex)
        import re
        self.assertIsNotNone(re.match(regex, "1. Implement formatter", re.I))

    def test_get_plan_config_step_skip_prefixes(self):
        skips = self._pm.get_plan_config("step_skip_prefixes")
        self.assertIn("what ", skips)
        self.assertIn("potential", skips)

    def test_get_toolchain_missing_key_returns_default(self):
        self.assertIsNone(self._pm.get_toolchain("nonexistent_key"))
        self.assertEqual(self._pm.get_toolchain("nonexistent_key", "fallback"), "fallback")

    def test_markers_includes_step_test(self):
        marker = self._pm.get_marker("step_test")
        self.assertEqual(marker, "### STEP_TEST:")


class TestKratosCliPromptUi(unittest.TestCase):
    """Regression tests for the interactive prompt wrapper (now kratos/app/*)."""

    @classmethod
    def setUpClass(cls):
        from kratos.app import cli, prompt_frame
        cls.cli = cli
        cls.prompt_frame = prompt_frame

    def setUp(self):
        if not getattr(self.prompt_frame, "_HAS_PT", False):
            self.skipTest("prompt_toolkit is not installed")

    def test_slash_completer_lists_root_commands(self):
        from prompt_toolkit.document import Document

        completions = list(self.cli._COMPLETER.get_completions(Document("/"), None))
        texts = {item.text for item in completions}

        self.assertIn("/help", texts)
        self.assertIn("/models", texts)
        self.assertIn("/permission", texts)

    def test_slash_completer_lists_subcommands(self):
        from prompt_toolkit.document import Document

        completions = list(self.cli._COMPLETER.get_completions(Document("/models c"), None))
        texts = {item.text for item in completions}

        self.assertEqual(texts, {"coder", "compressor"})

    def test_tui_slash_completions_root(self):
        """slash_completions() in tui.py covers the same roots as the legacy completer."""
        from kratos.app.tui import slash_completions

        values = {v for v, _, _ in slash_completions("/")}
        self.assertIn("/help", values)
        self.assertIn("/models", values)
        self.assertIn("/permission", values)

    def test_tui_slash_completions_subcommands(self):
        """slash_completions() returns correct subcommand set for /models c."""
        from kratos.app.tui import slash_completions

        values = {v for v, _, _ in slash_completions("/models c")}
        self.assertEqual(values, {"coder", "compressor"})

    def test_tui_slash_completions_empty_for_non_slash(self):
        """slash_completions() returns nothing for ordinary text."""
        from kratos.app.tui import slash_completions

        self.assertEqual(slash_completions("hello"), [])
        self.assertEqual(slash_completions(""), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
