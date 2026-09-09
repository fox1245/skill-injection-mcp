from __future__ import annotations

from skill_inject_mcp.engine import SkillInjectEngine


def test_skill_paths_are_posix_relative(engine: SkillInjectEngine):
    """Skill paths must be portable POSIX-relative (no backslashes / drive letters)."""
    skills = engine.registry.all()
    assert skills
    for s in skills:
        assert "\\" not in s.path, s.path
        assert not (len(s.path) >= 2 and s.path[1] == ":"), s.path
        assert s.path.endswith("SKILL.md"), s.path
        assert "/" in s.path, s.path
