from __future__ import annotations

import hashlib
from graphlib import CycleError, TopologicalSorter
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
    return meta, parts[2].lstrip("\n")


def scan_skills(skills_dir: Path) -> list[SkillMeta]:
    skills_dir = Path(skills_dir)
    results = []
    for skill_md in sorted(skills_dir.rglob("SKILL.md")):
        raw = skill_md.read_bytes()
        fm, body = _parse_frontmatter(raw.decode("utf-8-sig"))
        skill_id = str(fm.get("id") or fm.get("skill_id") or skill_md.parent.name)
        deps = fm.get("depends_on") or fm.get("dependencies") or []
        tags = fm.get("tags") or []
        if isinstance(deps, str):
            deps = [deps]
        if isinstance(tags, str):
            tags = [tags]
        results.append(SkillMeta(
            skill_id=skill_id, name=str(fm.get("name") or skill_id),
            description=str(fm.get("description") or ""), body=body,
            path=skill_md.relative_to(skills_dir).as_posix(),
            depends_on=[str(d) for d in deps], tags=[str(t) for t in tags],
            frontmatter=fm, content_hash=hashlib.sha256(raw).hexdigest(),
        ))
    return results


def validate_skill_graph(skills: list[SkillMeta]) -> list[ValidationErrorItem]:
    errors = []
    seen = {}
    for skill in skills:
        if skill.skill_id in seen:
            errors.append(ValidationErrorItem(
                code="duplicate_skill_id",
                message=f"Duplicate skill_id '{skill.skill_id}' at {skill.path} "
                        f"(also {seen[skill.skill_id]})", path=skill.path,
            ))
        seen[skill.skill_id] = skill.path
    graph = {s.skill_id: list(s.depends_on) for s in skills}
    for sid, deps in graph.items():
        for dep in deps:
            if dep not in graph:
                errors.append(ValidationErrorItem(
                    code="missing_dependency", message=f"Skill '{sid}' depends on unknown '{dep}'", path=sid,
                ))
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as exc:
        errors.append(ValidationErrorItem(
            code="dependency_cycle", message=f"Dependency cycle: {exc.args[1]}",
        ))
    return errors


class SkillRegistry:
    def __init__(self) -> None:
        self.skills: dict[str, SkillMeta] = {}
        self.validation_errors: list[ValidationErrorItem] = []
        self.duplicate_ids: set[str] = set()

    def load(self, skills_dir: Path) -> list[SkillMeta]:
        skills = scan_skills(skills_dir)
        self.validation_errors = validate_skill_graph(skills)
        self.skills = {}
        self.duplicate_ids = set()
        for skill in skills:
            if skill.skill_id in self.skills:
                self.duplicate_ids.add(skill.skill_id)
            else:
                self.skills[skill.skill_id] = skill
        return self.all()

    def get(self, skill_id: str) -> SkillMeta | None:
        return self.skills.get(skill_id)

    def all(self) -> list[SkillMeta]:
        return list(self.skills.values())

    def dependency_closure(self, skill_id: str) -> tuple[list[str], list[str]]:
        graph = {}
        pending = [skill_id]
        errors = []
        visited = set()
        while pending:
            sid = pending.pop()
            if sid in visited:
                continue
            visited.add(sid)
            skill = self.get(sid)
            if skill is None:
                errors.append(f"missing_dependency:{sid}")
                continue
            if sid in self.duplicate_ids:
                errors.append(f"duplicate_skill_id:{sid}")
            graph[sid] = skill.depends_on
            pending.extend(reversed(skill.depends_on))
        if errors:
            return [], errors
        try:
            return list(TopologicalSorter(graph).static_order()), []
        except CycleError:
            return [], [f"dependency_cycle:{skill_id}"]
