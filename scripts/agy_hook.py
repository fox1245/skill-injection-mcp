#!/usr/bin/env python3
"""Antigravity (AGY) hook integration for skill-inject-mcp.

Follows the agent-autonomous on-demand skill discovery architecture:
- No regex heuristics or hardcoded keyword/suffix lists.
- No forced per-turn retrieval overhead on every user prompt.
- Injects lightweight skill-discovery guidance (`GUIDANCE`) on supported
  session/invocation events so the agent autonomously decides when unfamiliar
  specialist work or a concrete capability gap warrants calling `resolve_skills`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from skill_inject_mcp.codex_integration import GUIDANCE
except Exception:
    GUIDANCE = ""

SUPPORTED_EVENTS = frozenset({"SessionStart", "PreInvocation", "UserPromptSubmit"})


def handle_hook_event(payload: Any) -> dict[str, Any]:
    """Return agent-autonomous skill discovery guidance for supported hook events."""
    if not isinstance(payload, dict):
        return {"continue": True}

    event_name = payload.get("hook_event_name")
    if not isinstance(event_name, str) or event_name not in SUPPORTED_EVENTS or not GUIDANCE:
        return {"continue": True}

    return {
        "continue": True,
        "mode": "agent_autonomous",
        "additionalContext": GUIDANCE,
    }


def main() -> int:
    try:
        raw_bytes = sys.stdin.buffer.read()
        raw = raw_bytes.decode("utf-8", errors="replace").strip()
        payload = json.loads(raw) if raw else {}
        response = handle_hook_event(payload)
    except Exception:
        response = {"continue": True}
    out_bytes = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
    sys.stdout.buffer.write(out_bytes)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
