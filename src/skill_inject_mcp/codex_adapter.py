"""Codex catalog synchronization and advisory UserPromptSubmit context."""
from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path

from skill_inject_mcp.engine import SkillInjectEngine


def sync_catalog(executable: str, cwd: Path, destination: Path) -> dict:
    """Read Codex's enabled global/system/plugin catalog without starting a task."""
    process = subprocess.Popen(
        [executable, "app-server", "--stdio"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    inbox = queue.Queue()
    def read():
        for line in process.stdout:
            try:
                inbox.put(json.loads(line))
            except ValueError:
                pass
    threading.Thread(target=read, daemon=True).start()
    def call(identifier, method, params):
        process.stdin.write(json.dumps({"id": identifier, "method": method, "params": params}) + "\n")
        process.stdin.flush()
        while True:
            response = inbox.get(timeout=30)
            if response.get("id") == identifier:
                if "error" in response:
                    raise RuntimeError("Codex catalog request failed")
                return response["result"]
    try:
        call(1, "initialize", {
            "clientInfo": {"name": "skill_injection_catalog", "title": "Skill catalog", "version": "1.0"},
            "capabilities": {"experimentalApi": True, "requestAttestation": False},
        })
        process.stdin.write('{"method":"initialized"}\n')
        process.stdin.flush()
        result = call(2, "skills/list", {"cwds": [str(cwd)], "forceReload": True})
        entries = result["data"][0]
        if entries.get("errors"):
            raise RuntimeError("Codex reported skill catalog errors")
        skills = [
            {"name": s["name"], "path": s["path"], "scope": s["scope"], "enabled": True}
            for s in entries["skills"] if s["enabled"] and s["scope"] in ("user", "system")
        ]
        if not skills:
            raise RuntimeError("Codex returned an empty global skill catalog")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Serialize catalog updates across this launcher; original skill files stay untouched.
        import os, tempfile
        fd, temporary = tempfile.mkstemp(prefix="catalog-", suffix=".json", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({"skills": skills}, output, ensure_ascii=False, indent=2)
            Path(temporary).replace(destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {"skills": len(skills), "manifest": str(destination)}
    finally:
        process.terminate()
        process.wait(timeout=10)


def prompt_context(engine: SkillInjectEngine, prompt: str) -> dict:
    """Discovery is advisory; binding still requires an explicit structured resolve."""
    stripped = prompt.strip()
    acknowledgements = {"", "네", "응", "좋아", "고마워", "감사합니다", "ok", "okay", "thanks", "yes", "no"}
    if stripped.casefold().rstrip(".! ") in acknowledgements:
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ""}}
    with engine._lock:
        try:
            engine.ensure_index()
            hits = engine.retriever.retrieve(stripped, top_k=3, queries=[stripped])
            candidates = []
            for hit in hits:
                skill = engine.registry.get(hit.skill_id)
                if skill is not None:
                    candidates.append({"skill_id": skill.skill_id, "description": skill.description[:400]})
            context = (
                "Skill Injection MCP discovery (unverified candidates, not bindings):\n"
                + json.dumps(candidates, ensure_ascii=False)
                + "\nFor substantive work, formulate atomic requirements from the full conversation "
                  "and call skill-injection.resolve_skills before committing to a workflow. "
                  "Preserve all user constraints. Read selected skills with get_skill_body and the "
                  "returned registry_snapshot. Never interpret discovery or complete as execution success. "
                  "Do not force irrelevant skills or stop solely because no matching skill exists."
            )
        except Exception as exc:
            context = (
                f"Skill discovery unavailable ({type(exc).__name__}). "
                "Use skill-injection.resolve_skills when available; do not invent a binding."
            )
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}
