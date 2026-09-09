from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skill_inject_mcp.schemas import SkillMeta, ValidationErrorItem


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    body = parts[2].lstrip("\n")
    return meta, body


def scan_skills(skills_dir: Path) -> list[SkillMeta]:
    skills_dir = Path(skills_dir)
    results: list[SkillMeta] = []
    if not skills_dir.exists():
        return results
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        folder_id = skill_md.parent.name
        skill_id = str(fm.get("id") or fm.get("skill_id") or folder_id)
        name = str(fm.get("name") or skill_id)
        description = str(fm.get("description") or "")
        depends_on = fm.get("depends_on") or fm.get("dependencies") or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        tags = fm.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        results.append(
            SkillMeta(
                skill_id=skill_id,
                name=name,
                description=description,
                body=body,
                path=str(skill_md.resolve()),
                depends_on=[str(d) for d in depends_on],
                tags=[str(t) for t in tags],
                frontmatter=fm,
            )
        )
    return results


def validate_skill_graph(skills: list[SkillMeta]) -> list[ValidationErrorItem]:
    errors: list[ValidationErrorItem] = []
    seen: dict[str, str] = {}
    for s in skills:
        if s.skill_id in seen:
            errors.append(
                ValidationErrorItem(
                    code="duplicate_skill_id",
                    message=f"Duplicate skill_id '{s.skill_id}' at {s.path} (also {seen[s.skill_id]})",
                    path=s.path,
                )
            )
        else:
            seen[s.skill_id] = s.path

    ids = {s.skill_id for s in skills}
    graph = {s.skill_id: list(s.depends_on) for s in skills}

    # missing deps (warn as validation)
    for sid, deps in graph.items():
        for d in deps:
            if d not in ids:
                errors.append(
                    ValidationErrorItem(
                        code="missing_dependency",
                        message=f"Skill '{sid}' depends on unknown '{d}'",
                        path=sid,
                    )
                )

    # cycle detection (DFS)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {sid: WHITE for sid in graph}
    stack: list[str] = []

    def visit(u: str) -> bool:
        color[u] = GRAY
        stack.append(u)
        for v in graph.get(u, []):
            if v not in color:
                continue
            if color[v] == GRAY:
                cycle_path = stack[stack.index(v) :] + [v]
                errors.append(
                    ValidationErrorItem(
                        code="dependency_cycle",
                        message="Dependency cycle: " + " -> ".join(cycle_path),
                        path=u,
                    )
                )
                return True
            if color[v] == WHITE and visit(v):
                return True
        stack.pop()
        color[u] = BLACK
        return False

    for sid in list(graph):
        if color[sid] == WHITE:
            visit(sid)

    return errors


class SkillRegistry:
    def __init__(self) -> None:
        self.skills: dict[str, SkillMeta] = {}
        self.validation_errors: list[ValidationErrorItem] = []

    def load(self, skills_dir: Path) -> list[SkillMeta]:
        skills = scan_skills(skills_dir)
        self.validation_errors = validate_skill_graph(skills)
        # keep first occurrence on duplicates for indexing, but errors recorded
        self.skills = {}
        for s in skills:
            if s.skill_id not in self.skills:
                self.skills[s.skill_id] = s
        return list(self.skills.values())

    def get(self, skill_id: str) -> SkillMeta | None:
        return self.skills.get(skill_id)

    def all(self) -> list[SkillMeta]:
        return list(self.skills.values())
