"""The diagnosis attached to a failed command observation must actually be
shown to the coder by format_observation (otherwise the per-item auto-verify
feedback is useless — the gap behind session 2026-06-13_13-12-22)."""

from kratos.execution.tools import format_observation


class _PM:
    """Minimal PromptManager stub → forces fallback templates."""
    def get_snippet(self, key):
        return None


def test_failed_command_observation_includes_diagnosis():
    obs = [{
        "kind": "command", "cmd": "python -m pytest", "skipped": False,
        "exit_code": 1, "is_test": True,
        "output": "E   TypeError: handle_starttag() missing 1 required positional argument",
        "detail": ("DIAGNOSIS [signature_mismatch]: handle_starttag() ...\n"
                   "REQUIRED FIX: match the base method signature exactly"),
    }]
    rendered = format_observation(obs, _PM())
    assert "DIAGNOSIS [signature_mismatch]" in rendered
    assert "REQUIRED FIX" in rendered
    assert "raw output (tail)" in rendered          # diagnosis shown ABOVE raw output
    assert "exit=1" in rendered


def test_successful_command_observation_has_no_diagnosis():
    obs = [{"kind": "command", "cmd": "python -m pytest", "skipped": False,
            "exit_code": 0, "is_test": True, "output": "3 passed"}]
    rendered = format_observation(obs, _PM())
    assert "DIAGNOSIS" not in rendered
    assert "3 passed" in rendered
