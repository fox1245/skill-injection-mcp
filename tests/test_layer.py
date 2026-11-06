from __future__ import annotations

import json
from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.registry.layer import classify_skill_layer
from skill_inject_mcp.registry.scan import SkillRegistry
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


def test_layer_overrides_file_precedence(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(skills, "plain-skill", "Create or update a skill with SKILL.md.")
    _write_skill(skills, "frontmatter-meta", "Install Python packages with pip.", extra="layer: meta\n")
    _write_skill(skills, "kept-meta", "Inspect skill catalogs.", extra="layer: meta\n")
    overrides = tmp_path / "layer_overrides.json"
    overrides.write_text(
        json.dumps({
            "schema_version": 1,
            "layers": {
                "plain-skill": "meta",
                "frontmatter-meta": "domain",
            },
        }),
        encoding="utf-8",
    )

    eng = SkillInjectEngine(
        Settings(
            skills_dir=skills,
            layer_overrides=overrides,
            index_dir=tmp_path / "idx",
            use_fake_embedder=True,
            _env_file=None,
            rerank="off",
            multi_query=False,
        )
    )
    try:
        info = eng.ensure_index()
        assert info["validation_errors"] == []
        assert eng.registry.get("plain-skill").layer == "meta"
        assert eng.registry.get("frontmatter-meta").layer == "domain"
        assert eng.registry.get("kept-meta").layer == "meta"
    finally:
        eng.close()


def test_layer_overrides_error_cases_yield_codes_and_apply_no_override(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(skills, "plain-skill", "Create or update a skill with SKILL.md.")
    _write_skill(skills, "frontmatter-meta", "Install Python packages with pip.", extra="layer: meta\n")

    missing_path = tmp_path / "missing_overrides.json"
    bad_json_path = tmp_path / "bad_json.json"
    bad_json_path.write_text("{not valid json", encoding="utf-8")
    extra_key_path = tmp_path / "extra_key.json"
    extra_key_path.write_text(
        json.dumps({"schema_version": 1, "layers": {"plain-skill": "meta"}, "note": "x"}),
        encoding="utf-8",
    )
    wrong_version_path = tmp_path / "wrong_version.json"
    wrong_version_path.write_text(
        json.dumps({"schema_version": 2, "layers": {"plain-skill": "meta", "frontmatter-meta": "domain"}}),
        encoding="utf-8",
    )
    bad_value_path = tmp_path / "bad_value.json"
    bad_value_path.write_text(
        json.dumps({"schema_version": 1, "layers": {"plain-skill": "meta", "frontmatter-meta": "invalid"}}),
        encoding="utf-8",
    )
    unknown_skill_path = tmp_path / "unknown_skill.json"
    unknown_skill_path.write_text(
        json.dumps({"schema_version": 1, "layers": {"no-such-skill": "meta"}}),
        encoding="utf-8",
    )

    cases = [
        (missing_path, "unreadable_layer_overrides"),
        (bad_json_path, "invalid_layer_overrides"),
        (extra_key_path, "invalid_layer_overrides"),
        (wrong_version_path, "invalid_layer_overrides_version"),
        (bad_value_path, "invalid_layer_override"),
        (unknown_skill_path, "unknown_layer_override"),
    ]
    for idx, (path, expected_code) in enumerate(cases):
        eng = SkillInjectEngine(
            Settings(
                skills_dir=skills,
                layer_overrides=path,
                index_dir=tmp_path / f"idx-{idx}",
                use_fake_embedder=True,
                _env_file=None,
                rerank="off",
                multi_query=False,
            )
        )
        try:
            info = eng.ensure_index()
            sig = eng._source_signature
            assert eng.ensure_index()["registry_snapshot"] == info["registry_snapshot"]
            assert eng._source_signature == sig
            codes = [err["code"] for err in info["validation_errors"]]
            assert expected_code in codes
            assert eng.registry.get("plain-skill").layer == "domain"
            assert eng.registry.get("frontmatter-meta").layer == "meta"
        finally:
            eng.close()

    # Deleting a previously readable invalid file invalidates the signature and reports unreadable.
    eng_del = SkillInjectEngine(
        Settings(
            skills_dir=skills,
            layer_overrides=bad_json_path,
            index_dir=tmp_path / "idx-del",
            use_fake_embedder=True,
            _env_file=None,
            rerank="off",
            multi_query=False,
        )
    )
    try:
        first_info = eng_del.ensure_index()
        assert first_info["validation_errors"][0]["code"] == "invalid_layer_overrides"
        bad_json_path.unlink()
        second_info = eng_del.ensure_index()
        assert second_info["validation_errors"][0]["code"] == "unreadable_layer_overrides"
    finally:
        eng_del.close()

    # An unknown skill_id reports unknown_layer_override without dropping valid overrides in the same file.
    mixed_unknown_path = tmp_path / "mixed_unknown.json"
    mixed_unknown_path.write_text(
        json.dumps({"schema_version": 1, "layers": {"plain-skill": "meta", "no-such-skill": "meta"}}),
        encoding="utf-8",
    )
    eng_mixed = SkillInjectEngine(
        Settings(
            skills_dir=skills,
            layer_overrides=mixed_unknown_path,
            index_dir=tmp_path / "idx-mixed",
            use_fake_embedder=True,
            _env_file=None,
            rerank="off",
            multi_query=False,
        )
    )
    try:
        mixed_info = eng_mixed.ensure_index()
        assert [err["code"] for err in mixed_info["validation_errors"]] == ["unknown_layer_override"]
        assert eng_mixed.registry.get("plain-skill").layer == "meta"
    finally:
        eng_mixed.close()


def test_layer_overrides_apply_to_manifest_registry(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(skills, "plain-skill", "Create or update a Codex skill with SKILL.md.")
    _write_skill(skills, "frontmatter-meta", "Install Python packages with pip.", extra="layer: meta\n")
    manifest = tmp_path / "catalog.json"
    manifest.write_text(
        json.dumps({
            "skills": [
                {"name": "plain-skill", "path": str(skills / "plain-skill" / "SKILL.md"), "enabled": True},
                {"name": "frontmatter-meta", "path": str(skills / "frontmatter-meta" / "SKILL.md"), "enabled": True},
                {"name": "disabled-skill", "path": str(skills / "disabled" / "SKILL.md"), "enabled": False},
            ]
        }),
        encoding="utf-8",
    )
    overrides = tmp_path / "layer_overrides.json"
    overrides.write_text(
        json.dumps({
            "schema_version": 1,
            "layers": {
                "plain-skill": "meta",
                "frontmatter-meta": "domain",
                "disabled-skill": "domain",
            },
        }),
        encoding="utf-8",
    )

    reg = SkillRegistry(layer_overrides=overrides)
    reg.load_manifest(manifest)
    assert reg.validation_errors == []
    assert reg.get("plain-skill").layer == "meta"
    assert reg.get("frontmatter-meta").layer == "domain"

    eng = SkillInjectEngine(
        Settings(
            skill_manifest=manifest,
            layer_overrides=overrides,
            index_dir=tmp_path / "idx",
            use_fake_embedder=True,
            _env_file=None,
            rerank="off",
            multi_query=False,
        )
    )
    try:
        resp = eng.resolve(SkillInjectRequest(requirements=[
            Requirement(
                id="author",
                description="Create or update a Codex skill with SKILL.md",
                required=True,
                layer="meta",
            ),
        ]))
        assert resp.checks[0].skill_id == "plain-skill"
        assert resp.checks[0].layer == "meta"
    finally:
        eng.close()


def test_editing_layer_overrides_reloads_on_next_resolve_without_restart(tmp_path: Path):
    skills = tmp_path / "skills"
    _write_skill(skills, "skill-alpha", "Create or update a Codex skill with SKILL.md frontmatter.")
    _write_skill(skills, "skill-beta", "Create or update a Codex skill with SKILL.md frontmatter.")
    overrides = tmp_path / "layer_overrides.json"
    overrides.write_text(
        json.dumps({
            "schema_version": 1,
            "layers": {"skill-alpha": "meta", "skill-beta": "domain"},
        }),
        encoding="utf-8",
    )

    eng = SkillInjectEngine(
        Settings(
            skills_dir=skills,
            layer_overrides=overrides,
            index_dir=tmp_path / "idx",
            use_fake_embedder=True,
            _env_file=None,
            rerank="off",
            multi_query=False,
        )
    )
    try:
        req = SkillInjectRequest(requirements=[
            Requirement(
                id="author",
                description="Create or update a Codex skill with SKILL.md frontmatter",
                required=True,
                layer="meta",
            ),
        ])
        first = eng.resolve(req)
        assert first.checks[0].skill_id == "skill-alpha"
        assert first.checks[0].layer == "meta"

        overrides.write_text(
            json.dumps({
                "schema_version": 1,
                "layers": {"skill-alpha": "domain", "skill-beta": "meta"},
            }),
            encoding="utf-8",
        )
        second = eng.resolve(req)
        assert second.registry_snapshot != first.registry_snapshot
        assert second.checks[0].skill_id == "skill-beta"
        assert second.checks[0].layer == "meta"
        assert eng.registry.get("skill-alpha").layer == "domain"
        assert eng.registry.get("skill-beta").layer == "meta"
    finally:
        eng.close()
