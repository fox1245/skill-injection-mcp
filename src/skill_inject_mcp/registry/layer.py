"""Separate skills-about-skills (meta) from domain skills during retrieval."""
from __future__ import annotations

import re
from typing import Iterable, Literal, TypeVar

from skill_inject_mcp.schemas import SkillMeta

SkillLayer = Literal["meta", "domain"]

KNOWN_META_IDS = {
    "skill-creator",
    "skill-installer",
    "skills-maintain",
    "find-skills",
    "skill-inspector",
    "agentx-codex-conductor",
    "plugin-creator",
    "template-creator",
    "ancient-grand-master-skill",
}

_KO = (
    "\uc2a4\ud0ac"
    r"\s*(?:"
    "\uc124\uce58|\uac10\uc0ac|\uac80\uc218|"
    "\uc0dd\uc131|\uc791\uc131|\uc720\uc9c0\ubcf4\uc218|\uba54\ud0c0)"
)
_REQ_META = re.compile(
    r"(?is)(?:skill\.md|skil{1,2}\s*injection|meta[- ]?skill|"
    r"create(?:\s+or\s+update)?\s+(?:a\s+)?(?:codex\s+)?skill|"
    r"install(?:able)?\s+(?:a\s+)?(?:codex\s+)?skill|"
    r"skill catalog|audit(?:ing)?\s+skills?|"
    r"inspect(?:ing)?\s+(?:a\s+)?skill|"
    + _KO
    + ")"
)


def classify_skill_layer(skill: SkillMeta) -> SkillLayer:
    raw = str(skill.frontmatter.get("layer") or "").strip().lower()
    if raw in ("meta", "domain"):
        return raw  # type: ignore[return-value]
    tags = {str(t).strip().lower() for t in skill.tags}
    if tags & {"meta", "meta-skill", "metaskill"}:
        return "meta"
    sid = skill.skill_id.lower()
    name = skill.name.lower()
    if sid in KNOWN_META_IDS or name in KNOWN_META_IDS:
        return "meta"
    markers = ("skill-creator", "skill-installer", "skill-inspector")
    if any(token in sid or token in name for token in markers):
        return "meta"
    blob = (skill.name + "\n" + skill.description).lower()
    needles = ("create", "install", "inspect", "audit", "catalog", "maintain")
    if "skill.md" in blob and any(word in blob for word in needles):
        return "meta"
    return "domain"


def classify_requirement_layer(description: str, search_query: str | None = None) -> SkillLayer:
    text = description + "\n" + (search_query or "")
    return "meta" if _REQ_META.search(text) else "domain"


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
