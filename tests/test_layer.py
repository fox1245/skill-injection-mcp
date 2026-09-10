from __future__ import annotations

from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.registry.layer import classify_requirement_layer, classify_skill_layer
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest, SkillMeta


def _write_skill(root: Path, name: str, description: str, extra: str = "") -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n# {name}\n\n{description}\n",
        encoding="utf-8",
    )


def _engine(tmp_path: Path, skills: Path) -> SkillInjectEngine:
    eng = SkillInjectEngine(
        Settings(skills_dir=skills, index_dir=tmp_path / "idx", use_fake_embedder=True, _env_file=None, rerank="off", multi_query=False)
    )
    eng.reindex()
    return eng


def test_explicit_frontmatter_and_requirement_layers():
    meta = SkillMeta(
        skill_id="custom-meta", name="custom-meta", description="does things",
        body="", path="custom-meta/SKILL.md", frontmatter={"layer": "meta"},
    )
    assert classify_skill_layer(meta) == "meta"
    assert classify_requirement_layer("Create a Codex skill with SKILL.md") == "meta"
    assert classify_requirement_layer("Install Python packages with pip") == "domain"


def test_domain_requirement_does_not_bind_meta_skill(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(skills, "skill-creator", "Create or update a Codex skill with SKILL.md frontmatter.")
    _write_skill(skills, "package-installer", "Install Python packages and project dependencies with pip.")
    eng = _engine(tmp_path, skills)
    try:
        resp = eng.resolve(SkillInjectRequest(requirements=[
            Requirement(id="pip", description="Install Python packages with pip", required=True),
        ]))
        assert resp.checks[0].skill_id == "package-installer"
        assert resp.checks[0].layer == "domain"
        assert resp.evidence
        assert resp.evidence[0].layer == "domain"
    finally:
        eng.close()


def test_meta_requirement_binds_meta_skill(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(skills, "skill-creator", "Create or update a Codex skill with SKILL.md frontmatter.")
    _write_skill(skills, "package-installer", "Install Python packages and project dependencies with pip.")
    eng = _engine(tmp_path, skills)
    try:
        resp = eng.resolve(SkillInjectRequest(requirements=[
            Requirement(id="author", description="Create a Codex skill with SKILL.md", required=True),
        ]))
        assert resp.checks[0].skill_id == "skill-creator"
        assert resp.checks[0].layer == "meta"
    finally:
        eng.close()
