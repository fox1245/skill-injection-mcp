from __future__ import annotations

from typing import Any, Literal
import asyncio
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import SkillInjectRequest, SkillInjectResponse
from skill_inject_mcp.presentation import ResolutionDetails


@asynccontextmanager
async def lifespan(server):
    _engine.request_background_refresh()
    try:
        yield {}
    finally:
        await _engine.aclose()


mcp = FastMCP(
    "skill-inject-mcp",
    instructions=(
        "Find skills on demand for unfamiliar work or a concrete capability gap. "
        "Read an explicitly named skill directly; reuse an existing selection while requirements remain unchanged. "
        "No search is required for ordinary questions or every workflow. No-match is not a reason to stop "
        "independent work or retry without new information. Inspect verifier mode, gaps and degradation; "
        "complete verifies skill support, not execution. Request full details when evidence review is needed."
    ),
    lifespan=lifespan,
)
_engine = SkillInjectEngine()
_details = ResolutionDetails()


@mcp.tool()
async def resolve_skills(request: SkillInjectRequest, detail: Literal["summary", "full"] = "summary") -> SkillInjectResponse:
    """Find skills when needed; search_query is a retrieval hint, not a requirement.

    constraints.top_k limits returned evidence (default 5), not acceptance checks.
    Unknown fields and schema versions are rejected. Complete means required
    requirements passed the configured verifier and dependencies, not successful execution.
    Default summary retains decisions, gaps and diagnostics. Use detail="full" or
    get_resolution_details(resolution_id) for unabridged citations and execution traces.
    """
    result = await asyncio.to_thread(_engine.resolve, request)
    return _details.present(result, detail)


@mcp.tool()
async def get_resolution_details(resolution_id: str) -> SkillInjectResponse:
    """Read unabridged evidence for a resolution, without another search or model call.

    Details are local to this process, retained for 15 minutes within 32 entries/4 MiB.
    Expired/evicted IDs report an error. Historical results retain their original snapshot.
    """
    try:
        return _details.get(resolution_id)
    except KeyError as exc:
        raise ToolError(str(exc)) from None


@mcp.tool()
async def reindex_skills(skills_dir: str | None = None) -> dict[str, Any]:
    """Rebuild indexes; reuse unchanged document vectors and preserve the old snapshot on failure."""
    from pathlib import Path
    return await asyncio.to_thread(_engine.reindex, Path(skills_dir) if skills_dir else None)


@mcp.tool()
async def get_skill_body(skill_id: str, registry_snapshot: str | None = None) -> dict[str, Any]:
    """Read the indexed SKILL.md; optionally require the snapshot returned by resolve_skills."""
    return await asyncio.to_thread(_engine.get_skill_body, skill_id, registry_snapshot)



@mcp.tool()
async def codex_prompt_hook(prompt: str) -> dict[str, Any]:
    """Optional discovery hook, disabled by default; never required before a workflow.

    Enable explicitly with SKILL_INJECT_PROMPT_HOOK_ENABLED=true and register a hook
    only when automatic suggestions are wanted. Direct resolve_skills remains available.
    """
    if not _engine.settings.prompt_hook_enabled:
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": ""}}
    from skill_inject_mcp.codex_adapter import async_prompt_context
    return await async_prompt_context(_engine, prompt)


def main() -> None:
    try:
        mcp.run(transport="stdio")
    finally:
        _engine.close()


if __name__ == "__main__":
    main()
