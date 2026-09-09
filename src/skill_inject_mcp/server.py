from __future__ import annotations

from typing import Any
import asyncio
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import SkillInjectRequest, SkillInjectResponse


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
        "Resolve required capabilities before inventing workflows. Inspect verification_mode: "
        "semantic judges original requirements across languages with source citations; lexical is "
        "an offline fallback. complete does not certify execution success. Review partial/unknown checks."
    ),
    lifespan=lifespan,
)
_engine = SkillInjectEngine()


@mcp.tool()
async def resolve_skills(request: SkillInjectRequest) -> SkillInjectResponse:
    """Resolve requirements to Agent Skills; search_query is only a retrieval hint.

    constraints.top_k limits returned evidence (default 5), not acceptance checks.
    Unknown fields and schema versions are rejected. Complete means required
    requirements passed the configured verifier and dependencies, not successful execution.
    """
    return await asyncio.to_thread(_engine.resolve, request)


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
    """Codex UserPromptSubmit hook: return advisory skill candidates, never block."""
    from skill_inject_mcp.codex_adapter import async_prompt_context
    return await async_prompt_context(_engine, prompt)


def main() -> None:
    try:
        mcp.run(transport="stdio")
    finally:
        _engine.close()


if __name__ == "__main__":
    main()
