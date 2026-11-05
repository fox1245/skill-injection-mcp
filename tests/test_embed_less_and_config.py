from __future__ import annotations

import inspect
import json
from pathlib import Path
import sqlite3

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.embedder import OpenRouterEmbedder, build_embedder
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.index.cache import EmbeddingCache
from skill_inject_mcp.retrieve.multi_query import expand_queries
from skill_inject_mcp.retrieve.semantic import SemanticVerifier
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest
from tests.embed_server import EmbeddingServer


def _write_skill(
    root: Path,
    skill_id: str,
    *,
    name: str | None = None,
    description: str,
    body: str,
    tags: list[str] | None = None,
) -> Path:
    skill_name = name or skill_id
    tag_line = f"tags: {json.dumps(tags)}\n" if tags is not None else ""
    path = root / skill_id / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {skill_name}\ndescription: {description}\n{tag_line}---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_summary_embedding_bounds_input_and_reembeds_only_changed_summary(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    index_dir = tmp_path / "index"
    head_marker = "HEAD_MARKER_ALPHA "
    tail_marker = " TAIL_MARKER_OMEGA_UNIQUEBM25"
    filler = "x" * (20_000 - len(head_marker) - len(tail_marker))
    long_body = head_marker + filler + tail_marker
    assert len(long_body) == 20_000

    _write_skill(
        skills_root,
        "long-skill",
        name="Long Skill",
        description="Install Python packages with pip",
        body=long_body,
        tags=["pip", "packaging"],
    )
    _write_skill(
        skills_root,
        "short-skill",
        name="Short Skill",
        description="Short helper skill",
        body="Short body under one hundred characters.",
        tags=["helper"],
    )

    summary_chars_1 = 500
    with EmbeddingServer() as server:
        eng1 = SkillInjectEngine(
            Settings(
                _env_file=None,
                skills_dir=skills_root,
                index_dir=index_dir,
                embedding_base_url=server.base_url,
                embedding_api_key="emb-test-key",
                use_fake_embedder=False,
                multi_query=False,
                verification_mode="lexical",
                index_summary_chars=summary_chars_1,
            )
        )
        try:
            eng1.ensure_index()
            inputs_first = server.inputs
            assert len(inputs_first) == 2
            long_input = next(text for text in inputs_first if "Long Skill" in text)
            expected_max_len = (
                len("Long Skill")
                + len("Install Python packages with pip")
                + summary_chars_1
                + len("pip packaging")
                + 3  # three newlines joining the 4 sections
            )
            assert len(long_input) <= expected_max_len
            assert head_marker in long_input
            assert tail_marker not in long_input
            # Sparse BM25 index retains the full body including the tail token.
            sparse_hits = eng1.sparse.search("TAIL_MARKER_OMEGA_UNIQUEBM25", top_k=5)
            assert [sid for sid, _, _ in sparse_hits] == ["long-skill"]
        finally:
            eng1.close()

        before_count = len(server.inputs)
        summary_chars_2 = 900
        eng2 = SkillInjectEngine(
            Settings(
                _env_file=None,
                skills_dir=skills_root,
                index_dir=index_dir,
                embedding_base_url=server.base_url,
                embedding_api_key="emb-test-key",
                use_fake_embedder=False,
                multi_query=False,
                verification_mode="lexical",
                index_summary_chars=summary_chars_2,
            )
        )
        try:
            eng2.ensure_index()
            new_inputs = server.inputs[before_count:]
            # Only long-skill changed its summary; short-skill is shorter than 500 chars and hits EmbeddingCache.
            assert len(new_inputs) == 1
            assert "Long Skill" in new_inputs[0]
            assert head_marker in new_inputs[0]
            assert tail_marker not in new_inputs[0]
        finally:
            eng2.close()


def test_persistent_cache_enabled_by_default_across_engines(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    index_dir = tmp_path / "index"
    _write_skill(
        skills_root,
        "alpha",
        description="Install Python packages with pip",
        body="Alpha body text for package installation.",
    )
    _write_skill(
        skills_root,
        "beta",
        description="Build SQLite BM25 sparse indexes",
        body="Beta body text for sparse indexing.",
    )

    assert Settings(_env_file=None).persistent_embedding_cache is True

    with EmbeddingServer() as server:
        # Engine A indexes with default persistent_embedding_cache setting (not passed explicitly).
        eng_a = SkillInjectEngine(
            Settings(
                _env_file=None,
                skills_dir=skills_root,
                index_dir=index_dir,
                embedding_base_url=server.base_url,
                OPENROUTER_API_KEY="test-key",
                use_fake_embedder=False,
                multi_query=False,
                verification_mode="lexical",
            )
        )
        try:
            eng_a.ensure_index()
            assert len(server.inputs) == 2
        finally:
            eng_a.close()

        # Brand-new Engine B on the same index_dir sends zero document requests.
        eng_b = SkillInjectEngine(
            Settings(
                _env_file=None,
                skills_dir=skills_root,
                index_dir=index_dir,
                embedding_base_url=server.base_url,
                OPENROUTER_API_KEY="test-key",
                use_fake_embedder=False,
                multi_query=False,
                verification_mode="lexical",
            )
        )
        try:
            eng_b.ensure_index()
            assert len(server.inputs) == 2

            # Editing one skill file causes exactly one document to be embedded.
            _write_skill(
                skills_root,
                "beta",
                description="Build SQLite BM25 and FTS5 sparse indexes",
                body="Updated beta body text.",
            )
            eng_b.ensure_index()
            assert len(server.inputs) == 3
            assert "Updated beta body text." in server.inputs[-1]
        finally:
            eng_b.close()


def test_snapshot_reuse_invalidated_when_index_summary_chars_changes(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    index_dir = tmp_path / "index"
    _write_skill(
        skills_root,
        "installer",
        description="Install Python packages with pip",
        body="Detailed body " * 200,
    )

    eng = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=skills_root,
            index_dir=index_dir,
            use_fake_embedder=True,
            multi_query=False,
            verification_mode="lexical",
            index_summary_chars=400,
        )
    )
    try:
        info1 = eng.ensure_index()
        info1_again = eng.ensure_index()
        assert info1["registry_snapshot"] == info1_again["registry_snapshot"]
        snapshot1 = eng._snapshot
        assert snapshot1 is not None
        assert eng.hook_catalog_state(snapshot1) == "current"

        eng.settings.index_summary_chars = 800
        assert eng.acquire_hook_snapshot() is None
        assert eng.hook_catalog_state(snapshot1) == "last-known"

        info2 = eng.ensure_index()
        assert info2["registry_snapshot"] != info1["registry_snapshot"]
    finally:
        eng.close()


def test_endpoint_and_key_separation_between_embedding_and_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "package-installer",
        name="package-installer",
        description="Install Python packages and project dependencies with pip.",
        body="Use pip install to install Python packages.",
    )

    seen_requests: list[tuple[str, str, str]] = []

    def respond(transport, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        auth = request.headers.get("Authorization", "")
        seen_requests.append((request.method, url, auth))
        payload = json.loads(request.content)
        if url.endswith("/embeddings"):
            inputs = payload["input"]
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))]},
            )
        if url.endswith("/chat/completions"):
            if "response_format" not in payload:
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": '["pip install packages", "python deps"]'}}]},
                )
            candidates = json.loads(payload["messages"][1]["content"])["candidates"]
            verdicts = [
                {
                    "skill_id": c["skill_id"],
                    "assessment": "supported",
                    "reason": "Supports pip package installation.",
                    "evidence": [{"source_id": c["sources"][0]["source_id"]}],
                    "unmet_requirements": [],
                }
                for c in candidates
            ]
            return httpx.Response(
                200,
                json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"results": verdicts})}}]},
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)

    # Case 1: Dedicated embedding_api_key and chat_api_key with separate base URLs.
    eng1 = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=skills_root,
            index_dir=tmp_path / "index-1",
            embedding_base_url="http://tei.invalid/v1",
            embedding_api_key="emb-key",
            chat_base_url="http://chat.invalid/v1",
            chat_api_key="chat-key",
            use_fake_embedder=False,
            embedding_dim=2,
            multi_query=True,
            verification_mode="semantic",
        )
    )
    try:
        res1 = eng1.resolve(
            SkillInjectRequest(
                requirements=[Requirement(id="r1", description="Install Python packages with pip")]
            )
        )
        assert res1.match_status == "complete"
    finally:
        eng1.close()

    emb_calls = [(url, auth) for _, url, auth in seen_requests if url.endswith("/embeddings")]
    chat_calls = [(url, auth) for _, url, auth in seen_requests if url.endswith("/chat/completions")]
    assert len(emb_calls) >= 2  # document indexing + query embedding
    assert len(chat_calls) == 2  # multi-query expansion + semantic verification
    assert all(url == "http://tei.invalid/v1/embeddings" and auth == "Bearer emb-key" for url, auth in emb_calls)
    assert all(url == "http://chat.invalid/v1/chat/completions" and auth == "Bearer chat-key" for url, auth in chat_calls)
    assert not any("tei.invalid" in url for url, _ in chat_calls)
    assert not any("chat.invalid" in url for url, _ in emb_calls)

    # Case 2: Only OPENROUTER_API_KEY set; both embedding and chat keys fall back to it.
    seen_requests.clear()
    eng2 = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=skills_root,
            index_dir=tmp_path / "index-2",
            embedding_base_url="http://tei.invalid/v1",
            chat_base_url="http://chat.invalid/v1",
            OPENROUTER_API_KEY="shared-fallback-key",
            use_fake_embedder=False,
            embedding_dim=2,
            multi_query=True,
            verification_mode="semantic",
        )
    )
    try:
        res2 = eng2.resolve(
            SkillInjectRequest(
                requirements=[Requirement(id="r2", description="Install Python packages with pip")]
            )
        )
        assert res2.match_status == "complete"
    finally:
        eng2.close()

    emb_calls_2 = [(url, auth) for _, url, auth in seen_requests if url.endswith("/embeddings")]
    chat_calls_2 = [(url, auth) for _, url, auth in seen_requests if url.endswith("/chat/completions")]
    assert emb_calls_2 and chat_calls_2
    assert all(url == "http://tei.invalid/v1/embeddings" and auth == "Bearer shared-fallback-key" for url, auth in emb_calls_2)
    assert all(
        url == "http://chat.invalid/v1/chat/completions" and auth == "Bearer shared-fallback-key"
        for url, auth in chat_calls_2
    )


def test_http_budgets_come_from_settings_and_signatures_have_no_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "package-installer",
        name="package-installer",
        description="Install Python packages and project dependencies with pip.",
        body="Use pip install to install Python packages.",
    )

    timeouts_by_stage: dict[str, dict[str, float]] = {}

    def respond(transport, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        ext_timeout = dict(request.extensions["timeout"])
        payload = json.loads(request.content)
        if url.endswith("/embeddings"):
            timeouts_by_stage["embedder"] = ext_timeout
            inputs = payload["input"]
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))]},
            )
        if "response_format" not in payload:
            timeouts_by_stage["multi_query"] = ext_timeout
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '["pip install", "python packages"]'}}]},
            )
        timeouts_by_stage["verifier"] = ext_timeout
        candidates = json.loads(payload["messages"][1]["content"])["candidates"]
        verdicts = [
            {
                "skill_id": c["skill_id"],
                "assessment": "supported",
                "reason": "Supports installation.",
                "evidence": [{"source_id": c["sources"][0]["source_id"]}],
                "unmet_requirements": [],
            }
            for c in candidates
        ]
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"results": verdicts})}}]},
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)

    engine = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=skills_root,
            index_dir=tmp_path / "index",
            OPENROUTER_API_KEY="test-key",
            use_fake_embedder=False,
            embedding_dim=2,
            multi_query=True,
            verification_mode="semantic",
            http_connect_timeout_s=13.5,
            http_write_timeout_s=41.5,
            http_pool_timeout_s=17.5,
            embedding_timeout_s=210.0,
            multi_query_timeout_s=45.0,
            verification_timeout_s=155.0,
        )
    )
    try:
        result = engine.resolve(
            SkillInjectRequest(
                requirements=[Requirement(id="r", description="Install Python packages with pip")]
            )
        )
        assert result.match_status == "complete"
    finally:
        engine.close()

    assert timeouts_by_stage["embedder"] == {
        "connect": 13.5,
        "read": 210.0,
        "write": 41.5,
        "pool": 17.5,
    }
    assert timeouts_by_stage["multi_query"] == {
        "connect": 13.5,
        "read": 45.0,
        "write": 41.5,
        "pool": 17.5,
    }
    assert timeouts_by_stage["verifier"] == {
        "connect": 13.5,
        "read": 155.0,
        "write": 41.5,
        "pool": 17.5,
    }

    # Verify model names, base URLs, and timeouts have no defaults in function signatures.
    required_params = ("model", "base_url", "timeout_s", "connect_timeout_s", "write_timeout_s", "pool_timeout_s")
    for fn in (OpenRouterEmbedder.__init__, build_embedder, expand_queries, SemanticVerifier.verify):
        sig = inspect.signature(fn)
        for name in required_params:
            assert name in sig.parameters, f"{fn} missing parameter {name}"
            assert sig.parameters[name].default is inspect.Parameter.empty, (
                f"{fn} parameter {name} must not have a default value"
            )

    for bad_field in ("http_connect_timeout_s", "http_write_timeout_s", "http_pool_timeout_s"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{bad_field: 0.0})
    with pytest.raises(ValidationError):
        Settings(_env_file=None, index_summary_chars=0)


def test_embedding_cache_get_many_and_put_many_single_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = EmbeddingCache(tmp_path / "cache", ("openrouter", "model", 4, "http://emb.invalid"), dim=4)

    connect_calls = 0
    real_connect = sqlite3.connect

    def counting_connect(*args, **kwargs):
        nonlocal connect_calls
        connect_calls += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)

    v1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    v2 = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float64)
    v3 = np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float64)

    cache.put_many({"d1": v1, "d2": v2, "d3": v3})
    assert connect_calls == 1

    loaded = cache.get_many(["d1", "missing", "d3", "d2"])
    assert connect_calls == 2
    assert set(loaded.keys()) == {"d1", "d2", "d3"}
    np.testing.assert_allclose(loaded["d1"], v1)
    np.testing.assert_allclose(loaded["d2"], v2)
    np.testing.assert_allclose(loaded["d3"], v3)


def test_vectors_paid_for_before_a_failed_build_are_kept(tmp_path, monkeypatch):
    from skill_inject_mcp.embed.embedder import FakeEmbedder
    from skill_inject_mcp.registry.scan import SkillRegistry

    FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skills"
    total = len(SkillRegistry().load(FIXTURES))
    assert total >= 3
    calls: list[int] = []
    original = FakeEmbedder.embed_documents

    def fail_on_second_batch(self, texts, *, deadline=None):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("embedding server unavailable")
        return original(self, texts, deadline=deadline)

    def make_engine():
        return SkillInjectEngine(Settings(
            _env_file=None, skills_dir=FIXTURES, index_dir=tmp_path / "index",
            use_fake_embedder=True, multi_query=False, embedding_batch_size=1,
        ))

    monkeypatch.setattr(FakeEmbedder, "embed_documents", fail_on_second_batch)
    first = make_engine()
    try:
        with pytest.raises(RuntimeError):
            first.ensure_index()
    finally:
        first.close()
    assert calls == [1, 1]

    calls.clear()
    monkeypatch.setattr(FakeEmbedder, "embed_documents", lambda self, texts, *, deadline=None: (calls.append(len(texts)), original(self, texts, deadline=deadline))[1])
    second = make_engine()
    try:
        second.ensure_index()
    finally:
        second.close()
    assert sum(calls) == total - 1
