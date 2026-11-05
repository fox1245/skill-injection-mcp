"""Separate skills-about-skills (meta) from domain skills during retrieval."""
from __future__ import annotations

from typing import Iterable, Literal, TypeVar

from skill_inject_mcp.schemas import SkillMeta, ValidationErrorItem

SkillLayer = Literal["meta", "domain"]


def classify_skill_layer(
    skill: SkillMeta,
    errors: list[ValidationErrorItem] | None = None,
) -> SkillLayer:
    if "layer" not in skill.frontmatter or skill.frontmatter.get("layer") is None:
        return "domain"
    raw = skill.frontmatter.get("layer")
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in ("meta", "domain"):
            return normalized  # type: ignore[return-value]
    if errors is not None:
        errors.append(
            ValidationErrorItem(
                code="invalid_skill_layer",
                message=f"Invalid skill layer {raw!r} in frontmatter; expected 'meta' or 'domain'",
                path=skill.path,
            )
        )
    return "domain"


Hit = TypeVar("Hit")


def partition_hits(
    hits: Iterable[Hit],
    skills: dict[str, SkillMeta],
    wanted: SkillLayer,
) -> tuple[list[Hit], list[Hit]]:
    primary: list[Hit] = []
    secondary: list[Hit] = []
    for hit in hits:
        skill_id = getattr(hit, "skill_id", None)
        skill = skills.get(skill_id) if isinstance(skill_id, str) else None
        layer = skill.layer if skill is not None else "domain"
        (primary if layer == wanted else secondary).append(hit)
    return primary, secondary
