import asyncio
import json

from skill_inject_mcp.config import Settings


def test_prompt_hook_default_does_no_discovery(monkeypatch):
    from skill_inject_mcp import server
    class Engine:
        settings = Settings(_env_file=None)
        def acquire_hook_snapshot(self):
            raise AssertionError("Default hook must not search")
    monkeypatch.setattr(server, "_engine", Engine())
    result = asyncio.run(server.codex_prompt_hook("Implement a retrieval service"))
    assert result["hookSpecificOutput"]["additionalContext"] == ""


def test_explicit_opt_in_still_discovers(engine, monkeypatch):
    from skill_inject_mcp import server
    engine.settings.prompt_hook_enabled = True
    monkeypatch.setattr(server, "_engine", engine)
    result = asyncio.run(server.codex_prompt_hook("Install Python project dependencies with pip"))
    assert "package-installer" in json.dumps(result)
    assert "once more" not in json.dumps(result)
    assert "before committing" not in json.dumps(result)


def test_summary_preserves_verdict_gaps_and_full_details(engine, monkeypatch):
    from skill_inject_mcp import server
    from skill_inject_mcp.schemas import SkillInjectRequest, Requirement
    monkeypatch.setattr(server, "_engine", engine)
    request = SkillInjectRequest(requirements=[Requirement(id="r", description="Install Python project dependencies with pip")])
    summary = asyncio.run(server.resolve_skills(request))
    full = asyncio.run(server.get_resolution_details(summary.resolution_id))
    assert summary.match_status == full.match_status
    assert summary.gaps == full.gaps
    assert summary.verification_mode == full.verification_mode
    assert summary.registry_snapshot == full.registry_snapshot
    assert summary.validation_errors == full.validation_errors
    assert summary.verification_diagnostics == full.verification_diagnostics
    assert summary.checks[0].assessment == full.checks[0].assessment
    assert summary.checks[0].unmet_requirements == full.checks[0].unmet_requirements
    assert summary.execution == [] and summary.evidence == []
    assert full.execution and full.evidence
    assert len(summary.model_dump_json()) < len(full.model_dump_json())


def test_detail_cache_is_bounded_and_expiration_is_explicit():
    import pytest
    from skill_inject_mcp.presentation import ResolutionDetails
    from skill_inject_mcp.schemas import SkillInjectResponse
    now = [0.0]
    cache = ResolutionDetails(max_entries=1, ttl=1, clock=lambda: now[0])
    first = cache.present(SkillInjectResponse(match_status="no_match"))
    second = cache.present(SkillInjectResponse(match_status="partial"))
    with pytest.raises(KeyError):
        cache.get(first.resolution_id)
    assert cache.get(second.resolution_id).match_status == "partial"
    now[0] = 2
    with pytest.raises(KeyError):
        cache.get(second.resolution_id)


def test_owned_hook_removal_preserves_unrelated_positions():
    from skill_inject_mcp.codex_integration import disable_prompt_hook, merge_guidance
    hooks = {"hooks": {"UserPromptSubmit": [
        {"hooks": [{"server": "skill-injection", "tool": "codex_prompt_hook"}]},
        {"hooks": [{"server": "self-directing-mcp", "tool": "codex_session_hook"}]}]}}
    result = disable_prompt_hook(hooks)
    assert result["hooks"]["UserPromptSubmit"][0]["hooks"] == []
    assert result["hooks"]["UserPromptSubmit"][1] == hooks["hooks"]["UserPromptSubmit"][1]
    assert disable_prompt_hook(result) == result
    assert hooks["hooks"]["UserPromptSubmit"][0]["hooks"]
    guidance = merge_guidance("User-owned instructions\n")
    assert guidance.startswith("User-owned instructions\n")
    assert merge_guidance(guidance) == guidance
