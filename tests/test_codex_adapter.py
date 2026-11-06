from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from skill_inject_mcp.codex_adapter import prompt_context, sync_catalog
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
    assert prompt_context(engine, "   \n\t ")["hookSpecificOutput"]["additionalContext"] == ""
    non_empty = prompt_context(engine, "좋아")["hookSpecificOutput"]["additionalContext"]
    assert "Skill Injection MCP discovery" in non_empty


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


def test_sync_catalog_keeps_loaded_skills_with_errors_and_fails_on_empty_or_rpc_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def make_fake_popen(skills_response: dict):
        class FakeProcess:
            def __init__(self, *args, **kwargs):
                r_fd, w_fd = os.pipe()
                self.stdout = os.fdopen(r_fd, "r", encoding="utf-8")
                self._out = os.fdopen(w_fd, "w", encoding="utf-8")
                self.stdin = self

            def write(self, data: str) -> int:
                for line in data.splitlines():
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("method") == "initialize":
                        self._out.write(json.dumps({"id": msg["id"], "result": {"serverInfo": {}}}) + "\n")
                        self._out.flush()
                    elif msg.get("method") == "skills/list":
                        reply = {"id": msg["id"], **skills_response}
                        self._out.write(json.dumps(reply) + "\n")
                        self._out.flush()
                return len(data)

            def flush(self) -> None:
                pass

            def terminate(self) -> None:
                try:
                    self._out.close()
                except OSError:
                    pass

            def wait(self, timeout: float | None = None) -> int:
                try:
                    self.stdout.close()
                except OSError:
                    pass
                return 0

        return FakeProcess

    dest = tmp_path / "catalog.json"
    monkeypatch.setattr(
        subprocess,
        "Popen",
        make_fake_popen({
            "result": {
                "data": [{
                    "skills": [
                        {"name": "user-skill", "path": "/skills/user-skill/SKILL.md", "scope": "user", "enabled": True},
                        {"name": "sys-skill", "path": "/skills/sys-skill/SKILL.md", "scope": "system", "enabled": True},
                        {"name": "disabled-skill", "path": "/skills/disabled/SKILL.md", "scope": "user", "enabled": False},
                        {"name": "repo-skill", "path": "/skills/repo/SKILL.md", "scope": "repo", "enabled": True},
                    ],
                    "errors": [
                        {"path": "/skills/broken/SKILL.md", "message": "missing YAML frontmatter delimited by ---"},
                    ],
                }]
            }
        }),
    )
    summary = sync_catalog("codex", tmp_path, dest)
    assert summary == {"skills": 2, "errors": 1, "manifest": str(dest)}
    written = json.loads(dest.read_text(encoding="utf-8"))
    assert written == {
        "skills": [
            {"name": "user-skill", "path": "/skills/user-skill/SKILL.md", "scope": "user", "enabled": True},
            {"name": "sys-skill", "path": "/skills/sys-skill/SKILL.md", "scope": "system", "enabled": True},
        ]
    }
    err_output = capsys.readouterr().err
    assert "/skills/broken/SKILL.md" in err_output
    assert "missing YAML frontmatter delimited by ---" in err_output

    # Empty enabled skill list still raises.
    empty_dest = tmp_path / "empty_catalog.json"
    monkeypatch.setattr(
        subprocess,
        "Popen",
        make_fake_popen({
            "result": {
                "data": [{
                    "skills": [
                        {"name": "disabled-skill", "path": "/skills/disabled/SKILL.md", "scope": "user", "enabled": False},
                    ],
                    "errors": [
                        {"path": "/skills/broken/SKILL.md", "message": "missing YAML frontmatter delimited by ---"},
                    ],
                }]
            }
        }),
    )
    with pytest.raises(RuntimeError, match="empty"):
        sync_catalog("codex", tmp_path, empty_dest)
    assert not empty_dest.exists()

    # JSON-RPC error still raises.
    monkeypatch.setattr(
        subprocess,
        "Popen",
        make_fake_popen({"error": {"code": -32000, "message": "server failure"}}),
    )
    with pytest.raises(RuntimeError, match="failed"):
        sync_catalog("codex", tmp_path, empty_dest)

    # Missing or non-list data in response still raises.
    for invalid_data in ([], {}):
        monkeypatch.setattr(
            subprocess,
            "Popen",
            make_fake_popen({"result": {"data": invalid_data}}),
        )
        with pytest.raises(RuntimeError, match="no skill catalog data"):
            sync_catalog("codex", tmp_path, empty_dest)

    # Malformed skill entry omitting required fields still raises before writing.
    monkeypatch.setattr(
        subprocess,
        "Popen",
        make_fake_popen({
            "result": {
                "data": [{
                    "skills": [
                        {"name": "user-skill", "path": "/skills/user-skill/SKILL.md", "scope": "user"},
                        {"name": "sys-skill", "path": "/skills/sys-skill/SKILL.md", "scope": "system", "enabled": True},
                    ],
                }]
            }
        }),
    )
    with pytest.raises(KeyError):
        sync_catalog("codex", tmp_path, empty_dest)
    assert not empty_dest.exists()
