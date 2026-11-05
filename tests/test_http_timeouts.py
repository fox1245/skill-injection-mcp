from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.embedder import build_embedder
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.retrieve.multi_query import expand_queries
from skill_inject_mcp.retrieve.semantic import SemanticVerifier
from skill_inject_mcp.schemas import SkillMeta

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skills"
_DEFAULTS = Settings(_env_file=None)
_HTTP_BUDGET_KW = {
    "connect_timeout_s": _DEFAULTS.http_connect_timeout_s,
    "write_timeout_s": _DEFAULTS.http_write_timeout_s,
    "pool_timeout_s": _DEFAULTS.http_pool_timeout_s,
}
_EMBEDDER_KW = {
    "model": _DEFAULTS.embedding_model,
    "base_url": _DEFAULTS.embedding_base_url,
    **_HTTP_BUDGET_KW,
}
_EXPAND_KW = {
    "model": _DEFAULTS.multi_query_model,
    "base_url": _DEFAULTS.chat_base_url,
    "provider_order": _DEFAULTS.chat_provider_order,
    **_HTTP_BUDGET_KW,
}
_VERIFY_KW = {
    "model": _DEFAULTS.verification_model,
    "base_url": _DEFAULTS.chat_base_url,
    "provider_order": _DEFAULTS.chat_provider_order,
    **_HTTP_BUDGET_KW,
}


@pytest.mark.parametrize("method", ["embed_documents", "embed_queries"])
@pytest.mark.parametrize("read_timeout", [180.0, 246.0])
def test_embedding_http_request_uses_configured_read_budget(monkeypatch, method, read_timeout):
    seen = []
    def respond(transport, request):
        seen.append(request.extensions["timeout"])
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))
        ]})
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    embedder, degraded = build_embedder(
        api_key="test",
        dim=2,
        timeout_s=read_timeout,
        **_EMBEDDER_KW,
    )
    vectors = getattr(embedder, method)(["Install packages"])
    assert len(vectors) == 1
    assert seen == [{"connect": 10.0, "read": read_timeout, "write": 30.0, "pool": 10.0}]
    assert not degraded


def test_snapshot_passes_embedding_read_budget_to_http(monkeypatch, tmp_path):
    seen = []
    def respond(transport, request):
        seen.append(request.extensions["timeout"]["read"])
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))
        ]})
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    engine = SkillInjectEngine(Settings(
        _env_file=None, skills_dir=FIXTURES, index_dir=tmp_path / "index",
        OPENROUTER_API_KEY="test", use_fake_embedder=False,
        embedding_dim=2, embedding_timeout_s=246,
    ))
    try:
        engine.reindex()
        assert seen and set(seen) == {246.0}
    finally:
        engine.close()


def test_embedding_timeout_names_budget_without_exposing_credentials(monkeypatch):
    def fail(transport, request):
        raise httpx.ReadTimeout("simulated outage", request=request)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    embedder, _ = build_embedder(
        api_key="secret-test-only",
        dim=2,
        timeout_s=246,
        **_EMBEDDER_KW,
    )
    with pytest.raises(RuntimeError, match="read_timeout_s=246") as error:
        embedder.embed_queries(["Install packages"])
    assert "ReadTimeout" in str(error.value)
    assert "secret-test-only" not in str(error.value)


def test_multi_query_read_budget_overrides_injected_client_default():
    seen = []
    def respond(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"choices": [
            {"message": {"content": '["pip install", "python dependencies"]'}}
        ]})
    with httpx.Client(transport=httpx.MockTransport(respond), timeout=1) as client:
        result = expand_queries(
            "Install packages",
            api_key="test",
            timeout_s=37,
            client=client,
            **_EXPAND_KW,
        )
    assert not result.skipped
    assert seen == [{"connect": 10.0, "read": 37, "write": 30.0, "pool": 10.0}]


def test_semantic_read_budget_overrides_injected_client_default():
    seen = []
    def respond(request):
        seen.append(request.extensions["timeout"])
        result = {"results": [{
            "skill_id": "installer", "assessment": "supported", "reason": "Supports installation.",
            "evidence": [{"source_id": "c0:description"}], "unmet_requirements": [],
        }]}
        return httpx.Response(200, json={"choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(result)}}
        ]})
    skill = SkillMeta(skill_id="installer", name="Installer", description="Install packages",
                      body="Installation instructions", path="installer/SKILL.md")
    with httpx.Client(transport=httpx.MockTransport(respond), timeout=1) as client:
        result = SemanticVerifier().verify(
            "Install packages",
            [skill],
            api_key="test",
            timeout_s=144,
            client=client,
            **_VERIFY_KW,
        )
    assert result.assessments["installer"].matched
    assert seen == [{"connect": 10.0, "read": 144, "write": 30.0, "pool": 10.0}]


@pytest.mark.parametrize("field", ["embedding_timeout_s", "multi_query_timeout_s", "verification_timeout_s"])
@pytest.mark.parametrize("value", [0, -1, float("inf")])
def test_invalid_read_budgets_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_embedding_read_budget_can_be_configured_by_environment(monkeypatch):
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_TIMEOUT_S", "275")
    assert Settings(_env_file=None).embedding_timeout_s == 275
