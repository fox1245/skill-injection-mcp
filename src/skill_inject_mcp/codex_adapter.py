"""Codex catalog synchronization and advisory UserPromptSubmit context."""
from __future__ import annotations

import json
import asyncio
import queue
import subprocess
import threading
from pathlib import Path

from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.retrieve.hybrid import reciprocal_rank_fusion
from skill_inject_mcp.retrieve.checks import lexical_gate
from skill_inject_mcp.registry.layer import classify_requirement_layer
from skill_inject_mcp.neograph_runtime import run_async_stages


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


def _hook_result(candidates: list, status: str, catalog: str = "unavailable", snapshot: str | None = None) -> dict:
    context = (
        "Skill Injection MCP discovery (unverified candidates, not bindings):\n"
        + json.dumps(candidates, ensure_ascii=False)
        + f"\nRetrieval: {status}; catalog: {catalog}; registry_snapshot: {snapshot or 'unavailable'}."
        + "\nFor substantive work, formulate atomic requirements from the full conversation "
          "and call skill-injection.resolve_skills before committing to a workflow. "
          "Preserve all user constraints. Read selected skills with get_skill_body and the "
          "returned registry_snapshot. Never interpret discovery or complete as execution success. "
          "Do not force irrelevant skills or stop solely because no matching skill exists."
    )
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}


async def async_prompt_context(engine: SkillInjectEngine, prompt: str) -> dict:
    result = None
    async def discover():
        nonlocal result
        result = await _discover_prompt_context(engine, prompt)
    async def validate_output():
        if not isinstance(result, dict) or "hookSpecificOutput" not in result:
            raise RuntimeError("Invalid skill discovery hook output")
    execution = await run_async_stages("skill_prompt_hook", [
        ("discover", discover), ("validate_output", validate_output),
    ])
    if result["hookSpecificOutput"].get("additionalContext"):
        result["hookSpecificOutput"]["additionalContext"] += (
            "\nExecutor: neograph-engine " + execution["version"] +
            "; nodes: " + ", ".join(execution["nodes"]) + "."
        )
    return result


async def _discover_prompt_context(engine: SkillInjectEngine, prompt: str) -> dict:
    """Bound advisory discovery; cancellation reaches the actual async HTTP request."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + engine.settings.hook_timeout_s
    stripped = prompt.strip()
    acknowledgements = {"", "네", "응", "좋아", "고마워", "감사합니다", "ok", "okay", "thanks", "yes", "no"}
    if stripped.casefold().rstrip(".! ") in acknowledgements:
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ""}}
    snapshot = None
    try:
        snapshot = engine.acquire_hook_snapshot()
        if snapshot is None:
            engine.request_background_refresh()
            if engine._last_refresh_error:
                return _hook_result([], f"unavailable; catalog refresh failed ({engine._last_refresh_error})")
            return _hook_result([], "unavailable; index warmup requested")
        catalog = engine.hook_catalog_state(snapshot)
        width = engine.settings.retrieve_top_k
        sparse = snapshot.sparse.search_readonly(stripped, top_k=width)
        selected = []
        status = "BM25-only fallback (embedding deadline exceeded)"
        # Reserve a little time for local ranking, rendering, and cancellation cleanup.
        remaining = deadline - loop.time() - min(.05, engine.settings.hook_timeout_s * .1)
        if remaining > 0:
            try:
                vector = (await asyncio.wait_for(snapshot.embedder.aembed_queries([stripped]), remaining))[0]
                dense = snapshot.dense.search_readonly(vector, top_k=width)
                hits = reciprocal_rank_fusion(dense, sparse, k=engine.settings.rrf_k)
                selected = [hit.skill_id for hit in hits[:3]]
                status = "hybrid (query cache or live embedding)"
            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                status = f"BM25-only fallback (embedding unavailable: {type(exc).__name__})"
        if status.startswith("BM25-only"):
            # BM25 ranks every token overlap; it is not confidence. In degraded
            # discovery, omit weak hits unless the description positively covers
            # the query terms. Do not infer cross-language support from this filter.
            for skill_id, score, _ in sparse[:3]:
                skill = snapshot.registry.get(skill_id)
                if skill is not None and score > 0 and lexical_gate(
                    stripped, skill.model_copy(update={"body": ""}),
                )[0]:
                    selected.append(skill_id)
        candidates = []
        for skill_id in selected:
            skill = snapshot.registry.get(skill_id)
            if skill is not None:
                candidates.append({"skill_id": skill.skill_id, "description": skill.description[:400]})
        wanted = classify_requirement_layer(stripped)
        layered = [
            item for item in candidates
            if (skill := snapshot.registry.get(item["skill_id"])) is not None and skill.layer == wanted
        ]
        if layered:
            candidates = layered
        # Source files may have changed while the embedding request was in flight.
        catalog = engine.hook_catalog_state(snapshot)
        return _hook_result(candidates, status, catalog, snapshot.snapshot_id)
    except Exception as exc:
        return _hook_result([], f"unavailable ({type(exc).__name__})")
    finally:
        if snapshot is not None:
            cleanup = snapshot.release(defer_disposal=True)
            if cleanup is not None:
                engine.defer_snapshot_cleanup(cleanup)


def prompt_context(engine: SkillInjectEngine, prompt: str) -> dict:
    """Synchronous convenience helper. MCP uses async_prompt_context on its shared loop."""
    async def run():
        try:
            return await async_prompt_context(engine, prompt)
        finally:
            await engine.aclose_async_clients()
    return asyncio.run(run())
