from __future__ import annotations

import json

import httpx
import pytest

from skill_inject_mcp.retrieve.semantic import SemanticVerifier
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest, SkillMeta

DESCRIPTION = "Install Python packages and project dependencies with pip or uv."
KOREAN = "pip으로 파이썬 프로젝트의 의존성 패키지를 설치한다."


def skill():
    return SkillMeta(skill_id="installer", name="Installer", description=DESCRIPTION,
                     body="Use pip install for project dependencies.", path="installer/SKILL.md")


def verdict(s=None, assessment="supported", unmet=None):
    s = s or skill()
    return {
        "skill_id": s.skill_id, "assessment": assessment, "reason": "Source supports this requirement.",
        "evidence": [{"source_id": "c0:description"}],
        "unmet_requirements": unmet or [],
    }


def response(results, finish_reason="stop"):
    return httpx.Response(200, json={"choices": [{
        "finish_reason": finish_reason,
        "message": {"content": json.dumps({"results": results})},
    }]})


def test_cross_language_support_uses_original_and_source_quotes():
    seen = []
    def transport(req):
        seen.append(json.loads(req.content))
        return response([verdict()])
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        result = SemanticVerifier().verify(KOREAN, [skill()], api_key="test", client=client)
    assessment = result.assessments["installer"]
    assert assessment.matched
    assert assessment.assessment == "supported"
    assert assessment.verifier == "semantic"
    assert assessment.citations[0]["quote"] == DESCRIPTION
    data = json.loads(seen[0]["messages"][1]["content"])
    assert data["original_requirement"] == KOREAN
    assert data["candidates"][0]["sources"][1]["text"] == skill().body
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["provider"]["require_parameters"]
    assert "c0:description" in seen[0]["response_format"]["json_schema"]["schema"]["$defs"]["SourceReference"]["properties"]["source_id"]["enum"]


@pytest.mark.parametrize("assessment", ["partial", "unsupported", "unknown"])
def test_non_supported_never_becomes_matched(assessment):
    v = verdict(assessment=assessment, unmet=["Vector search is not supported"])
    with httpx.Client(transport=httpx.MockTransport(lambda req: response([v]))) as client:
        result = SemanticVerifier().verify("Install packages and implement vector search", [skill()],
                                          api_key="test", client=client)
    assert not result.degraded
    assert not result.assessments["installer"].matched
    assert result.assessments["installer"].assessment == assessment


@pytest.mark.parametrize("defect", [
    "fabricated_quote", "wrong_field", "missing_candidate", "unknown_candidate",
    "duplicate_candidate", "supported_with_gap", "partial_without_gap", "no_evidence", "extra_field",
])
def test_invalid_verdicts_fail_closed(defect):
    v = verdict()
    results = [v]
    if defect == "fabricated_quote":
        v["evidence"][0]["source_id"] = "invented-source"
    elif defect == "wrong_field":
        v["evidence"][0]["field"] = "body"
    elif defect == "missing_candidate":
        results = []
    elif defect == "unknown_candidate":
        v["skill_id"] = "other"
    elif defect == "duplicate_candidate":
        results.append(v.copy())
    elif defect == "supported_with_gap":
        v["unmet_requirements"] = ["missing vector search"]
    elif defect == "partial_without_gap":
        v["assessment"] = "partial"
    elif defect == "no_evidence":
        v["evidence"] = []
    else:
        v["confidence"] = 1.0
    with httpx.Client(transport=httpx.MockTransport(lambda req: response(results))) as client:
        verifier = SemanticVerifier()
        result = verifier.verify(KOREAN, [skill()], api_key="test", client=client)
    assert result.degraded
    assert not result.assessments["installer"].matched
    assert result.assessments["installer"].assessment == "unknown"
    assert not verifier._cache


@pytest.mark.parametrize("failure", ["timeout", "http_error", "invalid_json", "truncated", "invalid_choice"])
def test_transport_or_parse_errors_are_unknown(failure):
    def transport(req):
        if failure == "timeout":
            raise httpx.ReadTimeout("simulated", request=req)
        if failure == "http_error":
            return httpx.Response(503)
        if failure == "invalid_json":
            return httpx.Response(200, json={"choices": [{"message": {"content": "not JSON"}}]})
        if failure == "invalid_choice":
            return httpx.Response(200, json={"choices": ["not an object"]})
        return response([verdict()], finish_reason="length")
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        result = SemanticVerifier().verify(KOREAN, [skill()], api_key="test", client=client)
    assert result.degraded
    assert not result.assessments["installer"].matched


def test_missing_api_key_does_not_certify_support():
    result = SemanticVerifier().verify(DESCRIPTION, [skill()], api_key=None)
    assert result.degraded
    assert result.reason == "no_api_key"
    assert not result.assessments["installer"].matched


def test_oversized_source_is_not_silently_truncated():
    s = skill().model_copy(update={"body": "x" * 1000 + "\nDoes not install packages."})
    result = SemanticVerifier().verify(KOREAN, [s], api_key="test", max_source_chars=100)
    assert result.degraded
    assert not result.assessments["installer"].matched
    assert result.assessments["installer"].reason == "semantic_source_limit_exceeded"


def test_cache_keys_include_original_requirement_source_and_model():
    seen = []
    def transport(req):
        body = json.loads(json.loads(req.content)["messages"][1]["content"])
        s = body["candidates"][0]
        seen.append(body)
        return response([{
            **verdict(), "skill_id": s["skill_id"],
            "evidence": [{"source_id": s["sources"][0]["source_id"]}],
        }])
    verifier = SemanticVerifier()
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        kwargs = {"api_key": "test", "client": client}
        verifier.verify(KOREAN, [skill()], **kwargs)
        verifier.verify(KOREAN, [skill()], **kwargs)
        assert len(seen) == 1
        verifier.verify("Install dependencies", [skill()], **kwargs)
        verifier.verify(KOREAN, [skill().model_copy(update={"body": "New implementation"})], **kwargs)
        verifier.verify(KOREAN, [skill()], model="different-model", **kwargs)
    assert len(seen) == 4


def test_engine_semantic_path_has_no_lexical_gate(engine, monkeypatch):
    engine.settings.verification_mode = "semantic"
    engine.settings.openrouter_api_key = "test"
    captured = []
    def transport(req):
        data = json.loads(json.loads(req.content)["messages"][1]["content"])
        captured.append(data)
        return response([{
            "skill_id": s["skill_id"],
            "assessment": "supported" if s["skill_id"] == "package-installer" else "unsupported",
            "reason": "Mock cross-language semantic judgement",
            "evidence": [{"source_id": s["sources"][0]["source_id"]}],
            "unmet_requirements": [] if s["skill_id"] == "package-installer" else ["Unsupported task"],
        } for s in data["candidates"]])
    monkeypatch.setattr("skill_inject_mcp.engine.evaluate_candidate",
                        lambda *a, **k: pytest.fail("Semantic mode must not use lexical verification"))
    original_verify = engine._verifier.verify
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        monkeypatch.setattr(engine._verifier, "verify",
                            lambda *a, **k: original_verify(*a, **k, client=client))
        req = SkillInjectRequest(requirements=[
            Requirement(id="r", description=KOREAN, search_query="unrelated hint"),
        ])
        result = engine.resolve(req)
    assert result.match_status == "complete"
    assert result.verification_mode == "semantic"
    assert not result.verification_degraded
    assert result.checks[0].skill_id == "package-installer"
    assert captured[0]["original_requirement"] == KOREAN


def test_engine_does_not_fall_back_to_lexical_after_semantic_outage(engine, monkeypatch):
    engine.settings.verification_mode = "semantic"
    engine.settings.openrouter_api_key = "test"
    original_verify = engine._verifier.verify
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503))) as client:
        monkeypatch.setattr(engine._verifier, "verify",
                            lambda *a, **k: original_verify(*a, **k, client=client))
        result = engine.resolve(SkillInjectRequest(requirements=[
            Requirement(id="r", description=DESCRIPTION),
        ]))
    assert result.match_status != "complete"
    assert result.verification_degraded
    assert result.checks[0].assessment == "unknown"


def test_negative_condition_can_be_semantically_supported():
    s = skill().model_copy(update={
        "description": "Install Python packages from a local wheel cache without network access.",
    })
    with httpx.Client(transport=httpx.MockTransport(lambda req: response([verdict(s)]))) as client:
        result = SemanticVerifier().verify("네트워크 없이 로컬 패키지를 설치한다.", [s],
                                          api_key="test", client=client)
    assert result.assessments["installer"].matched


def test_default_does_not_enable_remote_verification_from_api_key_alone():
    from skill_inject_mcp.config import Settings
    settings = Settings(_env_file=None, OPENROUTER_API_KEY="test")
    assert settings.verification_mode == "lexical"


def test_source_fragments_preserve_complete_original_text():
    from skill_inject_mcp.retrieve.semantic import source_fragments
    s = skill().model_copy(update={"body": "First paragraph.\n" + "x" * 4000 + "\nDoes not support network access."})
    fragments = source_fragments(s, chunk_chars=120)
    body = "".join(f["text"] for f in fragments.values() if f["field"] == "body")
    assert body == s.body
    assert all(len(f["text"]) <= 120 for key, f in fragments.items() if key.startswith("body:"))
