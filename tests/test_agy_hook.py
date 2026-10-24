from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.agy_hook import handle_hook_event
from skill_inject_mcp.codex_integration import GUIDANCE


def test_agent_autonomous_hook_injects_on_demand_guidance() -> None:
    res = handle_hook_event(
        {
            "hook_event_name": "PreInvocation",
            "prompt": "Optimize Apache AGE Cypher variable-length path query",
        }
    )
    assert res["continue"] is True
    assert res["mode"] == "agent_autonomous"
    assert res["additionalContext"] == GUIDANCE
    assert "resolve_skills" in res["additionalContext"]


def test_unsupported_or_missing_hook_event_passes_through_cleanly() -> None:
    assert handle_hook_event({"hook_event_name": "PostToolUse"}) == {"continue": True}
    assert handle_hook_event({}) == {"continue": True}
    assert handle_hook_event("not-a-dict") == {"continue": True}


def test_agy_hook_cli_round_trip_and_fail_open() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "agy_hook.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps({"hook_event_name": "SessionStart"}).encode("utf-8"),
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout.decode("utf-8"))
    assert payload["continue"] is True
    assert payload["mode"] == "agent_autonomous"
    assert payload["additionalContext"] == GUIDANCE

    bad_proc = subprocess.run(
        [sys.executable, str(script)],
        input=b"\xff\xfe{not valid json",
        capture_output=True,
        check=True,
    )
    bad_payload = json.loads(bad_proc.stdout.decode("utf-8"))
    assert bad_payload == {"continue": True}
