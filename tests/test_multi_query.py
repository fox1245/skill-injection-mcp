from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.retrieve.hybrid import HybridRetriever, reciprocal_rank_fusion
from skill_inject_mcp.retrieve.multi_query import expand_queries
from skill_inject_mcp.schemas import (
    Constraints,
    MatchStatus,
    Requirement,
    SkillInjectRequest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "skills"


def test_expand_queries_no_key_returns_original_only():
    mq = expand_queries(
        "Install Python packages with pip",
        None,
        api_key=None,
        enabled=True,
    )
    assert mq.skipped is True
    assert mq.reason == "no_api_key"
    assert mq.queries == ["Install Python packages with pip"]


def test_expand_queries_disabled_returns_original():
    mq = expand_queries(
        "BM25 sparse search",
        "FTS5PorterTokenXyz",
        api_key="sk-test",
        enabled=False,
    )
    assert mq.skipped is True
    assert mq.queries == ["FTS5PorterTokenXyz"]


def test_expand_queries_mock_llm_merges_original_first():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '["pip install dependencies", "uv add packages", "Python package installer skill"]'
                        }
                    }
                ]
            },
        )
    )
    client = httpx.Client(transport=transport)
    mq = expand_queries(
        "Install Python packages with pip",
        None,
        api_key="sk-test",
        enabled=True,
        client=client,
    )
    client.close()
    assert mq.skipped is False
    assert mq.queries[0] == "Install Python packages with pip"
    assert len(mq.queries) >= 3
    assert "pip install dependencies" in mq.queries


def test_expand_queries_timeout_degrades():
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    client = httpx.Client(transport=httpx.MockTransport(_raise))
    mq = expand_queries(
        "Install packages",
        None,
        api_key="sk-test",
        enabled=True,
        client=client,
        timeout_s=1.0,
    )
    client.close()
    assert mq.skipped is True
    assert mq.queries == ["Install packages"]
    assert mq.reason and mq.reason.startswith("error:")


def test_multi_query_merge_widens_candidates(tmp_path: Path):
    """Expanded queries feed one RRF; paraphrases can surface synonym skills."""
    eng = SkillInjectEngine(
        Settings(
            skills_dir=FIXTURES,
            index_dir=tmp_path / "idx",
            use_fake_embedder=True,
            multi_query=False,  # we inject queries directly via retriever
        )
    )
    eng.reindex()
    assert eng.retriever is not None
    # Single keyword-poor query vs expanded set including install synonyms
    single = eng.retriever.retrieve("Add project deps", top_k=5, queries=["Add project deps"])
    multi = eng.retriever.retrieve(
        "Add project deps",
        top_k=5,
        queries=[
            "Add project deps",
            "Install Python packages with pip",
            "package installer dependencies uv",
        ],
    )
    multi_ids = {h.skill_id for h in multi}
    assert "package-installer" in multi_ids
    # Multi should include at least as many unique candidates or rank installer highly
    assert any(h.skill_id == "package-installer" for h in multi[:3])


def test_engine_expander_failure_degrades(tmp_path: Path, monkeypatch):
    eng = SkillInjectEngine(
        Settings(
            skills_dir=FIXTURES,
            index_dir=tmp_path / "idx",
            use_fake_embedder=True,
            multi_query=True,
            openrouter_api_key="sk-test-key",
        )
    )
    eng.reindex()

    def _boom(*_a, **_k):
        from skill_inject_mcp.retrieve.multi_query import MultiQueryResult

        return MultiQueryResult(queries=["Install Python packages and project dependencies with pip"], skipped=True, reason="error:TimeoutException")

    monkeypatch.setattr("skill_inject_mcp.engine.expand_queries", _boom)
    resp = eng.resolve(
        SkillInjectRequest(
            requirements=[
                Requirement(id="r", description="Install Python packages and project dependencies with pip", required=True)
            ]
        )
    )
    assert resp.retriever_degraded is True
    assert any("multi_query_skipped" in n for n in resp.notes)
    # Still resolves via original query + match gate
    assert resp.match_status == MatchStatus.complete


def test_custom_top_k_changes_evidence_width(tmp_path: Path):
    eng = SkillInjectEngine(
        Settings(skills_dir=FIXTURES, index_dir=tmp_path / "idx", use_fake_embedder=True, multi_query=False)
    )
    eng.reindex()
    req = [
        Requirement(
            id="r",
            description="BM25 FTS5 sparse full-text search in Python",
            required=True,
        )
    ]
    narrow = eng.resolve(
        SkillInjectRequest(requirements=req, constraints=Constraints(top_k=1))
    )
    wide = eng.resolve(
        SkillInjectRequest(requirements=req, constraints=Constraints(top_k=3))
    )
    narrow_ev = [e for e in narrow.evidence if e.requirement_id == "r"]
    wide_ev = [e for e in wide.evidence if e.requirement_id == "r"]
    assert len(narrow_ev) == 1
    assert len(wide_ev) >= len(narrow_ev)
    assert len(wide_ev) <= 3
    # Default (omitted) uses resolve_top_k=5
    default = eng.resolve(SkillInjectRequest(requirements=req))
    default_ev = [e for e in default.evidence if e.requirement_id == "r"]
    assert len(default_ev) <= 5
    assert len(default_ev) >= len(narrow_ev)
