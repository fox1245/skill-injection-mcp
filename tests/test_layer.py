from __future__ import annotations

from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.registry.layer import classify_skill_layer
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest, SkillMeta, ValidationErrorItem


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
    errors: list[ValidationErrorItem] = []
    meta = SkillMeta(
        skill_id="custom-meta", name="custom-meta", description="does things",
        body="", path="custom-meta/SKILL.md", frontmatter={"layer": "meta"},
    )
    domain_explicit = SkillMeta(
        skill_id="skill-creator", name="skill-creator", description="Create a skill with SKILL.md",
        body="", path="skill-creator/SKILL.md", frontmatter={"layer": "domain"}, tags=["meta"],
    )
    domain_default = SkillMeta(
        skill_id="agentx-codex-conductor", name="agentx-codex-conductor",
        description="Create or inspect SKILL.md catalog",
        body="", path="agentx-codex-conductor/SKILL.md", frontmatter={}, tags=["meta"],
    )
    assert classify_skill_layer(meta, errors) == "meta"
    assert classify_skill_layer(domain_explicit, errors) == "domain"
    assert classify_skill_layer(domain_default, errors) == "domain"
    assert errors == []

    invalid = SkillMeta(
        skill_id="bad-layer", name="bad-layer", description="bad layer value",
        body="", path="bad-layer/SKILL.md", frontmatter={"layer": "orchestrator"},
    )
    assert classify_skill_layer(invalid, errors) == "domain"
    assert len(errors) == 1
    assert errors[0].code == "invalid_skill_layer"
    assert errors[0].path == "bad-layer/SKILL.md"


def test_domain_requirement_does_not_bind_meta_skill(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(
        skills,
        "skill-creator",
        "Install Python packages with pip and create a skill.",
        extra="layer: meta\n",
    )
    _write_skill(skills, "package-installer", "Install Python packages and project dependencies with pip.")
    eng = _engine(tmp_path, skills)
    try:
        resp = eng.resolve(SkillInjectRequest(requirements=[
            Requirement(
                id="pip",
                description="Install Python packages with pip",
                required=True,
                layer="domain",
            ),
        ]))
        assert resp.checks[0].skill_id == "package-installer"
        assert resp.checks[0].layer == "domain"
        assert resp.evidence
        assert resp.evidence[0].layer == "domain"
    finally:
        eng.close()


def test_invalid_frontmatter_layer_reports_validation_error_and_unlayered_requirement_does_not_partition(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(
        skills,
        "bad-layer-skill",
        "Install Python packages and project dependencies with pip.",
        extra="layer: not-a-valid-layer\n",
    )
    _write_skill(
        skills,
        "meta-only",
        "Create or update a Codex skill with SKILL.md frontmatter.",
        extra="layer: meta\n",
    )
    eng = _engine(tmp_path, skills)
    try:
        info = eng.ensure_index()
        assert any(err["code"] == "invalid_skill_layer" for err in info["validation_errors"])
        bad_skill = eng.registry.get("bad-layer-skill")
        assert bad_skill is not None and bad_skill.layer == "domain"

        # With layer=None on Requirement, meta skills are not partitioned away.
        resp = eng.resolve(SkillInjectRequest(requirements=[
            Requirement(
                id="unlayered",
                description="Create or update a Codex skill with SKILL.md frontmatter",
                required=True,
            ),
        ]))
        assert resp.checks[0].skill_id == "meta-only"
        assert resp.checks[0].layer == "meta"
        assert any(err.code == "invalid_skill_layer" for err in resp.validation_errors)
    finally:
        eng.close()


def test_meta_requirement_binds_meta_skill(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(
        skills,
        "skill-creator",
        "Create or update a Codex skill with SKILL.md frontmatter.",
        extra="layer: meta\n",
    )
    _write_skill(
        skills,
        "package-installer",
        "Create or update a Codex skill with SKILL.md frontmatter and install packages.",
    )
    eng = _engine(tmp_path, skills)
    try:
        resp = eng.resolve(SkillInjectRequest(requirements=[
            Requirement(
                id="author",
                description="Create or update a Codex skill with SKILL.md",
                required=True,
                layer="meta",
            ),
        ]))
        assert resp.checks[0].skill_id == "skill-creator"
        assert resp.checks[0].layer == "meta"
        assert resp.evidence[0].layer == "meta"
    finally:
        eng.close()


def test_conductor_is_meta_and_covers_orchestration(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(
        skills,
        "agentx-codex-conductor",
        "Conduct Codex use of skill-injection-mcp, self-directing-mcp, and AgentX on one delegated production task.",
        extra="layer: meta\n",
    )
    _write_skill(skills, "package-installer", "Install Python packages and project dependencies with pip.")
    eng = _engine(tmp_path, skills)
    try:
        scanned = next(s for s in eng.registry.all() if s.skill_id == "agentx-codex-conductor")
        assert scanned.layer == "meta"
        resp = eng.resolve(SkillInjectRequest(requirements=[
            Requirement(
                id="conduct",
                description="Conduct Codex use of skill-injection-mcp, self-directing-mcp, and AgentX on one delegated production task.",
                required=True,
                layer="meta",
            ),
        ]))
        assert resp.checks[0].skill_id == "agentx-codex-conductor"
        assert resp.checks[0].layer == "meta"
    finally:
        eng.close()
