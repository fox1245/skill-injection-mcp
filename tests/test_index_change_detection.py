from __future__ import annotations

import json
import os
from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.registry.scan import SkillRegistry
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest


def test_unchanged_catalog_does_not_reparse_skill_yaml(engine, monkeypatch):
    def fail(*args):
        raise AssertionError("Unchanged catalog must reuse the parsed snapshot")
    monkeypatch.setattr(SkillRegistry, "load", fail)
    snapshot = engine.ensure_index()["registry_snapshot"]
    assert engine.ensure_index()["registry_snapshot"] == snapshot


def test_manifest_addition_and_source_edit_refresh_without_restart(tmp_path):
    first, second = tmp_path / "first" / "SKILL.md", tmp_path / "second" / "SKILL.md"
    for path, desc in [(first, "Install packages"), (second, "Build indexes")]:
        path.parent.mkdir()
        path.write_text(f"---\nname: original\ndescription: {desc}\n---\n{desc}", encoding="utf-8")
    manifest = tmp_path / "catalog.json"
    def write(paths):
        manifest.write_text(json.dumps({"skills": [{"name": p.parent.name, "path": str(p)} for p in paths]}), encoding="utf-8")
    write([first])
    engine = SkillInjectEngine(Settings(_env_file=None, skill_manifest=manifest, index_dir=tmp_path / "index", use_fake_embedder=True))
    try:
        one = engine.ensure_index()
        write([first, second])
        two = engine.ensure_index()
        assert two["skills_indexed"] == 2 and two["registry_snapshot"] != one["registry_snapshot"]
        second.write_text(second.read_text().replace("Build indexes", "Build vectors"), encoding="utf-8")
        three = engine.ensure_index()
        assert three["registry_snapshot"] != two["registry_snapshot"]
        assert engine.get_skill_body("second")["description"] == "Build vectors"
    finally:
        engine.close()


def test_forced_reindex_detects_same_size_timestamp_preserved_edit(tmp_path):
    path = tmp_path / "skills" / "example" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\nname: example\ndescription: alpha\n---\nalpha", encoding="utf-8")
    engine = SkillInjectEngine(Settings(_env_file=None, skills_dir=path.parent.parent, index_dir=tmp_path / "index", use_fake_embedder=True))
    try:
        first = engine.ensure_index()
        stat = path.stat()
        path.write_text(path.read_text().replace("alpha", "bravo"), encoding="utf-8")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = engine.reindex()
        assert second["registry_snapshot"] != first["registry_snapshot"]
        assert engine.get_skill_body("example")["description"] == "bravo"
    finally:
        engine.close()


def test_manifest_with_removed_path_reports_validation_error_and_resolves_remaining(tmp_path):
    first = tmp_path / "first" / "SKILL.md"
    second = tmp_path / "second" / "SKILL.md"
    missing = tmp_path / "third" / "SKILL.md"
    for path, desc in [
        (first, "Install Python packages with pip"),
        (second, "Build BM25 sparse search indexes"),
    ]:
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {path.parent.name}\ndescription: {desc}\n---\n{desc}", encoding="utf-8")
    manifest = tmp_path / "catalog.json"
    manifest.write_text(
        json.dumps({
            "skills": [
                {"name": "first", "path": str(first)},
                {"name": "second", "path": str(second)},
                {"name": "third", "path": str(missing)},
            ]
        }),
        encoding="utf-8",
    )
    engine = SkillInjectEngine(
        Settings(_env_file=None, skill_manifest=manifest, index_dir=tmp_path / "index", use_fake_embedder=True, multi_query=False)
    )
    try:
        info = engine.ensure_index()
        assert info["skills_indexed"] == 2
        assert len(info["validation_errors"]) == 1
        assert info["validation_errors"][0]["code"] == "missing_skill_path"
        assert info["validation_errors"][0]["path"] == str(missing)

        resp = engine.resolve(
            SkillInjectRequest(
                requirements=[Requirement(id="r1", description="Install Python packages with pip", required=True)]
            )
        )
        assert resp.match_status == "complete"
        assert resp.checks[0].skill_id == "first"
        assert any(
            err.code == "missing_skill_path" and err.path == str(missing)
            for err in resp.validation_errors
        )
    finally:
        engine.close()


def test_manifest_file_removed_and_restored_between_ensure_index_calls(tmp_path):
    first = tmp_path / "first" / "SKILL.md"
    second = tmp_path / "second" / "SKILL.md"
    for path, desc in [
        (first, "Install Python packages with pip"),
        (second, "Build BM25 sparse search indexes"),
    ]:
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {path.parent.name}\ndescription: {desc}\n---\n{desc}", encoding="utf-8")
    manifest = tmp_path / "catalog.json"
    manifest.write_text(
        json.dumps({
            "skills": [
                {"name": "first", "path": str(first)},
                {"name": "second", "path": str(second)},
            ]
        }),
        encoding="utf-8",
    )
    engine = SkillInjectEngine(
        Settings(_env_file=None, skill_manifest=manifest, index_dir=tmp_path / "index", use_fake_embedder=True)
    )
    try:
        one = engine.ensure_index()
        sig_one = engine._source_signature
        assert one["skills_indexed"] == 2
        assert one["validation_errors"] == []

        saved_text = second.read_text(encoding="utf-8")
        second.unlink()
        two = engine.ensure_index()
        sig_two = engine._source_signature
        assert sig_two != sig_one
        assert two["registry_snapshot"] != one["registry_snapshot"]
        assert two["skills_indexed"] == 1
        assert engine.registry.get("second") is None
        assert len(two["validation_errors"]) == 1
        assert two["validation_errors"][0]["code"] == "missing_skill_path"
        assert two["validation_errors"][0]["path"] == str(second)

        second.write_text(saved_text, encoding="utf-8")
        three = engine.ensure_index()
        sig_three = engine._source_signature
        assert sig_three != sig_two
        assert three["registry_snapshot"] != two["registry_snapshot"]
        assert three["skills_indexed"] == 2
        assert engine.registry.get("second") is not None
        assert three["validation_errors"] == []
    finally:
        engine.close()


def test_directory_file_vanished_between_listing_and_reading_then_restored(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    first = skills_dir / "first" / "SKILL.md"
    second = skills_dir / "second" / "SKILL.md"
    for path, desc in [
        (first, "Install Python packages with pip"),
        (second, "Build BM25 sparse search indexes"),
    ]:
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {path.parent.name}\ndescription: {desc}\n---\n{desc}", encoding="utf-8")
    engine = SkillInjectEngine(
        Settings(_env_file=None, skills_dir=skills_dir, index_dir=tmp_path / "index", use_fake_embedder=True)
    )
    try:
        one = engine.ensure_index()
        sig_one = engine._source_signature
        assert one["skills_indexed"] == 2
        assert one["validation_errors"] == []

        stat = second.stat()
        os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        orig_read_bytes = Path.read_bytes
        vanish = True

        def flaky_read_bytes(self: Path) -> bytes:
            if vanish and self.resolve() == second.resolve():
                raise FileNotFoundError(f"Simulated vanish while reading {self}")
            return orig_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", flaky_read_bytes)
        two = engine.ensure_index()
        sig_two = engine._source_signature
        assert sig_two != sig_one
        assert two["registry_snapshot"] != one["registry_snapshot"]
        assert two["skills_indexed"] == 1
        assert engine.registry.get("second") is None
        assert len(two["validation_errors"]) == 1
        assert two["validation_errors"][0]["code"] == "missing_skill_path"
        assert two["validation_errors"][0]["path"] == str(second)

        vanish = False
        three = engine.ensure_index()
        sig_three = engine._source_signature
        assert sig_three != sig_two
        assert three["registry_snapshot"] != two["registry_snapshot"]
        assert three["skills_indexed"] == 2
        assert engine.registry.get("second") is not None
        assert three["validation_errors"] == []
    finally:
        engine.close()


def test_manifest_entry_pointing_to_directory_or_non_skill_md_reported(tmp_path):
    valid = tmp_path / "valid" / "SKILL.md"
    valid.parent.mkdir(parents=True)
    valid.write_text("---\nname: valid\ndescription: Valid skill\n---\nValid skill", encoding="utf-8")
    a_dir = tmp_path / "some_directory"
    a_dir.mkdir(parents=True)
    non_skill = tmp_path / "other" / "README.md"
    non_skill.parent.mkdir(parents=True)
    non_skill.write_text("not a skill", encoding="utf-8")

    manifest = tmp_path / "catalog.json"
    manifest.write_text(
        json.dumps({
            "skills": [
                {"name": "valid", "path": str(valid)},
                {"name": "dir-entry", "path": str(a_dir)},
                {"name": "readme-entry", "path": str(non_skill)},
            ]
        }),
        encoding="utf-8",
    )
    engine = SkillInjectEngine(
        Settings(_env_file=None, skill_manifest=manifest, index_dir=tmp_path / "index", use_fake_embedder=True)
    )
    try:
        info = engine.ensure_index()
        assert info["skills_indexed"] == 1
        error_paths = [
            err["path"] for err in info["validation_errors"] if err["code"] == "missing_skill_path"
        ]
        assert error_paths == [str(a_dir), str(non_skill)]
    finally:
        engine.close()
