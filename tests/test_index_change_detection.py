from __future__ import annotations

import json
import os
from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.registry.scan import SkillRegistry


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
