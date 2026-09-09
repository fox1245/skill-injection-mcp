from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import SkillInjectRequest, SkillInjectResponse

mcp = FastMCP(
    "skill-inject-mcp",
    instructions=(
        "Resolve required capabilities before inventing workflows. complete means all required "
        "requirements have conservative textual skill support and valid dependencies, not that "
        "execution succeeded. Inspect checks, missing_terms and evidence. Unknown checks need review."
    ),
)
_engine = SkillInjectEngine()


@mcp.tool()
def resolve_skills(request: SkillInjectRequest) -> SkillInjectResponse:
    """Resolve requirements to Agent Skills; search_query is only a retrieval hint.

    constraints.top_k limits returned evidence (default 5), not acceptance checks.
    Unknown fields and schema versions are rejected. Complete means required
    textual support and dependencies were checked, not successful execution.
    """
    return _engine.resolve(request)


@mcp.tool()
def reindex_skills(skills_dir: str | None = None) -> dict[str, Any]:
    """Rebuild indexes; reuse unchanged document vectors and preserve the old snapshot on failure."""
    from pathlib import Path
    return _engine.reindex(skills_dir=Path(skills_dir) if skills_dir else None)


@mcp.tool()
def get_skill_body(skill_id: str, registry_snapshot: str | None = None) -> dict[str, Any]:
    """Read the indexed SKILL.md; optionally require the snapshot returned by resolve_skills."""
    return _engine.get_skill_body(skill_id, registry_snapshot)


def main() -> None:
    try:
        mcp.run(transport="stdio")
    finally:
        _engine.close()


if __name__ == "__main__":
    main()
