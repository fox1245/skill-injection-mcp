from __future__ import annotations

import json
from pathlib import Path

from skill_inject_mcp.codex_adapter import prompt_context
from skill_inject_mcp.registry.scan import SkillRegistry


def test_prompt_hook_returns_advisory_context(engine):
    result = prompt_context(engine, "Install Python project dependencies with pip")
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "UserPromptSubmit"
    assert "package-installer" in output["additionalContext"]
    assert "unverified" in output["additionalContext"]
    assert "resolve_skills" in output["additionalContext"]
    assert "decision" not in result


def test_acknowledgement_does_not_index(engine, monkeypatch):
    monkeypatch.setattr(engine, "ensure_index", lambda: (_ for _ in ()).throw(AssertionError("no indexing")))
    assert prompt_context(engine, "좋아")["hookSpecificOutput"]["additionalContext"] == ""


def test_hook_failure_does_not_block_user(engine, monkeypatch):
    monkeypatch.setattr(engine, "acquire_hook_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    result = prompt_context(engine, "Implement a retrieval service")
    assert "unavailable" in result["hookSpecificOutput"]["additionalContext"]
    assert "decision" not in result


def test_manifest_uses_codex_names_and_original_paths(tmp_path):
    original = tmp_path / "installed" / "different-folder" / "SKILL.md"
    original.parent.mkdir(parents=True)
    original.write_text("---\nname: Original name\ndescription: A real installed skill\n---\nInstructions", encoding="utf-8")
    manifest = tmp_path / "catalog.json"
    manifest.write_text(json.dumps({"skills": [
        {"name": "plugin:real-skill", "path": str(original), "enabled": True},
        {"name": "disabled", "path": str(tmp_path / "missing"), "enabled": False},
    ]}), encoding="utf-8")
    registry = SkillRegistry()
    loaded = registry.load_manifest(manifest)
    assert len(loaded) == 1
    assert loaded[0].skill_id == "plugin:real-skill"
    assert loaded[0].source_path == str(original.resolve())
    assert loaded[0].body == "Instructions"
    assert registry.validation_errors == []


def test_manifest_content_updates_are_seen(tmp_path):
    from skill_inject_mcp.config import Settings
    from skill_inject_mcp.engine import SkillInjectEngine
    original = tmp_path / "SKILL.md"
    original.write_text("---\ndescription: Install packages\n---\nInstall packages", encoding="utf-8")
    manifest = tmp_path / "catalog.json"
    manifest.write_text(json.dumps({"skills": [{"name": "installer", "path": str(original)}]}), encoding="utf-8")
    engine = SkillInjectEngine(Settings(_env_file=None, skill_manifest=manifest,
                                      index_dir=tmp_path / "index", use_fake_embedder=True))
    try:
        a = engine.ensure_index()
        original.write_text("---\ndescription: Install Python packages\n---\nInstall Python packages", encoding="utf-8")
        b = engine.ensure_index()
        assert a["registry_snapshot"] != b["registry_snapshot"]
        assert engine.get_skill_body("installer")["source_path"] == str(original.resolve())
    finally:
        engine.close()


def test_optional_prompt_hook_has_no_mandatory_retry_instructions(engine):
    result = prompt_context(engine, "Install Python project dependencies with pip")
    assert "retry_hints" not in result["hookSpecificOutput"]
    assert "once more" not in result["hookSpecificOutput"]["additionalContext"]
    assert "Optional suggestions" in result["hookSpecificOutput"]["additionalContext"]
