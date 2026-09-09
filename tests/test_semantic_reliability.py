from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from skill_inject_mcp.config import Settings
from skill_inject_mcp.retrieve.semantic import SemanticVerifier
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest, SkillMeta


def candidate():
    return SkillMeta(skill_id="code", name="Frontend code", path="code/SKILL.md",
                     description="Implement runnable frontend source code.",
                     body="Write React and TypeScript. Do not substitute image mockups for code.")


def verdict(assessment="supported"):
    return {"skill_id": "code", "assessment": assessment, "reason": "Source checked.",
            "evidence": [{"source_id": "c0:description"}],
            "unmet_requirements": [] if assessment == "supported" else ["Cannot meet this requirement"]}


def reply(result=None, *, finish="stop", content=None, usage=None):
    if content is None:
        content = json.dumps({"results": [verdict() if result is None else result]})
    return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {"content": content}}],
                                    "usage": usage or {}})


@pytest.mark.parametrize("truncated_content", [None, '{"results":[{"skill_id":"code","assessment":"supp'])
def test_truncation_retries_once_with_more_budget_and_unchanged_sources(truncated_content):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            # Even parseable JSON is not safe to accept when the provider says it was cut off.
            return reply(finish="length", content=truncated_content, usage={"prompt_tokens": 42000, "completion_tokens": 4096,
                "completion_tokens_details": {"reasoning_tokens": 1971}})
        return reply(verdict("unsupported"))
    verifier = SemanticVerifier()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = verifier.verify("Build runnable code", [candidate()], api_key="test", client=client,
                                 max_tokens=4096)
    assert [r["max_tokens"] for r in requests] == [4096, 8192]
    assert requests[0]["messages"] == requests[1]["messages"]
    assert requests[0]["response_format"] == requests[1]["response_format"]
    assert not result.degraded
    assert result.assessments["code"].assessment == "unsupported"
    assert not result.assessments["code"].matched
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "response_truncated" and diagnostic.retrying
    assert diagnostic.finish_reason == "length"
    assert diagnostic.completion_tokens == 4096 and diagnostic.reasoning_tokens == 1971
    assert all(not item.matched for item in verifier._cache.values())


def test_repeated_truncation_is_bounded_and_never_cached():
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return reply(finish="length")
    verifier = SemanticVerifier()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = verifier.verify("Build code", [candidate()], api_key="test", client=client,
                                 max_tokens=1024)
    assert [r["max_tokens"] for r in requests] == [1024, 2048]
    assert result.degraded and result.assessments["code"].assessment == "unknown"
    assert result.reason == "semantic_verifier_error:response_truncated"
    assert [d.retrying for d in result.diagnostics] == [True, False]
    assert [d.attempt for d in result.diagnostics] == [1, 2]
    assert not verifier._cache


def test_retry_can_be_disabled():
    calls = []
    def respond(request):
        calls.append(request)
        return reply(finish="length")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = SemanticVerifier().verify("Build code", [candidate()], api_key="test", client=client,
                                           max_retries=0)
    assert len(calls) == 1 and not result.diagnostics[0].retrying
    assert result.assessments["code"].assessment == "unknown"


@pytest.mark.parametrize(("defect", "code"), [
    ("invalid_json", "invalid_verdict_json"),
    ("extra_field", "invalid_verdict_schema"),
    ("wrong_source", "invalid_source_reference"),
    ("missing_candidate", "candidate_set_mismatch"),
    ("duplicate_candidate", "candidate_set_mismatch"),
    ("supported_with_gap", "supported_with_unmet_requirements"),
    ("partial_without_gap", "partial_without_unmet_requirements"),
    ("no_evidence", "missing_evidence"),
    ("empty_content", "empty_response_content"),
    ("bad_envelope", "invalid_response_envelope"),
    ("filtered", "response_filtered"),
])
def test_non_truncation_failures_have_specific_codes_without_retries(defect, code):
    calls = []
    value = verdict()
    def respond(request):
        calls.append(request)
        if defect == "invalid_json":
            return reply(content="{not-json")
        if defect == "empty_content":
            return reply(content="")
        if defect == "bad_envelope":
            return httpx.Response(200, json={"choices": ["unexpected"]})
        if defect == "filtered":
            return reply(finish="content_filter")
        if defect == "missing_candidate":
            return reply(content='{"results": []}')
        if defect == "duplicate_candidate":
            return reply(content=json.dumps({"results": [value, value]}))
        if defect == "extra_field":
            value["injected"] = "unexpected"
        elif defect == "wrong_source":
            value["evidence"][0]["source_id"] = "c1:description"
        elif defect == "supported_with_gap":
            value["unmet_requirements"] = ["code"]
        elif defect == "partial_without_gap":
            value["assessment"] = "partial"
        elif defect == "no_evidence":
            value["evidence"] = []
        return reply(value)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = SemanticVerifier().verify("Build code", [candidate()], api_key="test", client=client)
    assert len(calls) == 1
    assert result.degraded and result.assessments["code"].assessment == "unknown"
    assert result.reason == f"semantic_verifier_error:{code}"
    assert result.diagnostics[0].code == code


def test_failed_retry_does_not_reuse_the_first_attempts_apparent_support():
    calls = []
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return reply(finish="length")
        raise httpx.ReadTimeout("SECRET: request credentials and private content", request=request)
    verifier = SemanticVerifier()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = verifier.verify("Build code", [candidate()], api_key="SECRET-KEY", client=client)
    assert len(calls) == 2 and result.degraded
    assert result.assessments["code"].assessment == "unknown"
    assert not verifier._cache
    assert result.diagnostics[-1].code == "http_timeout"
    assert "SECRET" not in repr(result)


def test_one_invalid_candidate_keeps_batch_validation_atomic():
    other = candidate().model_copy(update={"skill_id": "other"})
    invalid = {**verdict(), "skill_id": "other"}
    # c0 belongs to code; the second candidate may not borrow its evidence.
    with httpx.Client(transport=httpx.MockTransport(lambda request: reply(
        content=json.dumps({"results": [verdict(), invalid]})))) as client:
        verifier = SemanticVerifier()
        result = verifier.verify("Build code", [candidate(), other], api_key="test", client=client)
    assert result.reason == "semantic_verifier_error:invalid_source_reference"
    assert all(a.assessment == "unknown" for a in result.assessments.values())
    assert not verifier._cache


def test_retry_still_enforces_source_ownership():
    calls = []
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return reply(finish="length")
        invalid = verdict()
        invalid["evidence"] = [{"source_id": "invented"}]
        return reply(invalid)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = SemanticVerifier().verify("Build code", [candidate()], api_key="test", client=client)
    assert len(calls) == 2 and result.degraded
    assert result.assessments["code"].assessment == "unknown"
    assert [d.code for d in result.diagnostics] == ["response_truncated", "invalid_source_reference"]


def test_failure_diagnostics_do_not_echo_untrusted_response_fields():
    def respond(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "SECRET_FINISH_REASON",
            "message": {"content": "SECRET_CONTENT", "reasoning": "SECRET_REASONING"}}],
            "usage": {"prompt_tokens": "SECRET_USAGE", "completion_tokens_details": "SECRET_DETAILS"}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = SemanticVerifier().verify("PRIVATE_REQUIREMENT", [candidate()], api_key="SECRET_KEY", client=client)
    assert "SECRET" not in repr(result)
    assert "PRIVATE_REQUIREMENT" not in repr(result)
    assert result.diagnostics[0].code == "unexpected_finish_reason"


def test_engine_exposes_retry_diagnostics_and_configured_budget(engine, monkeypatch):
    engine.settings.verification_mode = "semantic"
    engine.settings.openrouter_api_key = "test"
    engine.settings.verification_max_tokens = 6000
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return reply(finish="length")
        candidates = json.loads(payload["messages"][1]["content"])["candidates"]
        return reply(content=json.dumps({"results": [{
            "skill_id": s["skill_id"], "assessment": "supported" if s["skill_id"] == "package-installer" else "unsupported",
            "reason": "Source checked.", "evidence": [{"source_id": s["sources"][0]["source_id"]}],
            "unmet_requirements": [] if s["skill_id"] == "package-installer" else ["Unsupported"],
        } for s in candidates]}))
    original = engine._verifier.verify
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(engine._verifier, "verify", lambda *a, **kw: original(*a, **kw, client=client))
        result = engine.resolve(SkillInjectRequest(requirements=[Requirement(id="r", description="Install Python packages with pip")]))
    assert result.match_status == "complete" and not result.verification_degraded
    assert [r["max_tokens"] for r in requests] == [6000, 12000]
    assert result.verification_diagnostics[0].requirement_id == "r"
    assert result.verification_diagnostics[0].code == "response_truncated"


@pytest.mark.parametrize(("field", "value"), [("verification_max_tokens", 0), ("verification_max_tokens", -1),
                                             ("verification_max_retries", -1), ("verification_max_retries", 2)])
def test_invalid_verification_budgets_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_default_and_environment_verification_budgets(monkeypatch):
    assert Settings(_env_file=None).verification_max_tokens == 8192
    assert Settings(_env_file=None).verification_max_retries == 1
    monkeypatch.setenv("SKILL_INJECT_VERIFICATION_MAX_TOKENS", "10000")
    monkeypatch.setenv("SKILL_INJECT_VERIFICATION_MAX_RETRIES", "0")
    settings = Settings(_env_file=None)
    assert settings.verification_max_tokens == 10000 and settings.verification_max_retries == 0
