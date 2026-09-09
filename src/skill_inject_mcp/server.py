from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import SkillInjectRequest

mcp = FastMCP(
    "skill-inject-mcp",
    instructions=(
        "Skill Injection MCP: call resolve_skills before inventing workflows. "
        "Only match_status=complete means all required skills are bound. "
        "See AGENTS.md nudge on every resolve_skills response."
    ),
)

_engine = SkillInjectEngine()


@mcp.tool()
def resolve_skills(request: dict[str, Any] | SkillInjectRequest) -> dict[str, Any]:
    """Resolve task requirements to indexed Agent Skills via hybrid retrieval.

    Pass a SkillInjectRequest (schema_version 1.0) with requirements[].
    Optional constraints.top_k (int >= 1) controls how many hybrid candidates
    are walked and returned as evidence per requirement; default is 5 when omitted
    (internal sparse/dense channel still uses retrieve_top_k, typically 20).
    Returns SkillInjectResponse. Never complete if required requirements are unmatched.
    """
    if isinstance(request, SkillInjectRequest):
        req = request
    else:
        req = SkillInjectRequest.model_validate(request)
    resp = _engine.resolve(req)
    payload = resp.model_dump(mode="json")
    # Agents.md-style nudge always present
    nudge = (
        "NUDGE: If match_status != complete, do not proceed as if skills are bound. "
        "Address gaps / validation_errors or call reindex_skills."
    )
    payload.setdefault("notes", [])
    if nudge not in payload["notes"]:
        payload["notes"].insert(0, nudge)
    return payload


@mcp.tool()
def reindex_skills(skills_dir: str | None = None) -> dict[str, Any]:
    """Rescan skills directory and rebuild sparse + dense indexes."""
    from pathlib import Path

    info = _engine.reindex(skills_dir=Path(skills_dir) if skills_dir else None)
    return info


@mcp.tool()
def get_skill_body(skill_id: str) -> dict[str, Any]:
    """Return the full SKILL.md body for a skill_id."""
    return _engine.get_skill_body(skill_id)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
