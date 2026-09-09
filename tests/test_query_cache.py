from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest

from skill_inject_mcp.embed.embedder import FakeEmbedder, OpenRouterEmbedder


def test_exact_query_cache_deduplicates_misses_preserves_order_and_is_bounded(monkeypatch):
    calls = []
    def respond(transport, request):
        inputs = json.loads(request.content)["input"]
        calls.append(inputs)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1., float(len(text))]} for i, text in enumerate(inputs)
        ]})
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    embedder = OpenRouterEmbedder(api_key="test", dim=2, query_cache_size=2)
    try:
        vectors = embedder.embed_queries(["alpha", "beta", "alpha"])
        assert len(calls) == 1 and len(calls[0]) == 2
        np.testing.assert_equal(vectors[0], vectors[2])
        original = vectors[0].copy()
        vectors[0][:] = 0
        np.testing.assert_equal(embedder.embed_queries(["alpha"])[0], original)
        embedder.embed_queries(["gamma"])
        embedder.embed_queries(["beta"])
        assert len(calls) == 3  # beta was least recently used and evicted.
    finally:
        embedder.close()


def test_query_cache_is_shared_between_async_hook_and_sync_resolve(monkeypatch):
    calls = []
    original = FakeEmbedder._embed_queries_uncached
    def record(self, texts):
        calls.append(list(texts))
        return original(self, texts)
    monkeypatch.setattr(FakeEmbedder, "_embed_queries_uncached", record)
    embedder = FakeEmbedder(dim=8)
    first = embedder.embed_queries(["same requirement"])[0]
    async def run():
        cached = await embedder.aembed_queries(["same requirement"])
        np.testing.assert_equal(cached[0], first)
        await embedder.aembed_queries(["new requirement"])
    asyncio.run(run())
    embedder.embed_queries(["new requirement"])
    assert calls == [["same requirement"], ["new requirement"]]


def test_invalid_embedding_batch_does_not_publish_partial_cache(monkeypatch):
    embedder = FakeEmbedder(dim=2)
    calls = []
    def broken(texts):
        calls.append(list(texts))
        return [np.array([1., 0.]), np.array([float("nan"), 0.])]
    monkeypatch.setattr(embedder, "_embed_queries_uncached", broken)
    for _ in range(2):
        with pytest.raises(ValueError, match="embedding"):
            embedder.embed_queries(["one", "two"])
    assert calls == [["one", "two"], ["one", "two"]]


def test_model_change_does_not_reuse_previous_query_vector(monkeypatch):
    models = []
    def respond(transport, request):
        models.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1., 0.]}]})
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    embedder = OpenRouterEmbedder(api_key="test", dim=2)
    try:
        embedder.embed_queries(["same query"])
        embedder.model = "different-model"
        embedder.embed_queries(["same query"])
        assert models == ["qwen/qwen3-embedding-8b", "different-model"]
    finally:
        embedder.close()


def test_sync_http_client_is_reused_and_closed(monkeypatch):
    clients = []
    original = httpx.Client
    class TrackingClient(original):
        def __init__(self, **kwargs):
            kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [
                {"index": i, "embedding": [1., 0.]} for i, _ in enumerate(json.loads(request.content)["input"])
            ]}))
            super().__init__(**kwargs)
            clients.append(self)
    monkeypatch.setattr(httpx, "Client", TrackingClient)
    embedder = OpenRouterEmbedder(api_key="test", dim=2)
    embedder.embed_documents(["document"])
    embedder.embed_queries(["first"])
    embedder.embed_queries(["second"])
    assert len(clients) == 1 and not clients[0].is_closed
    embedder.close()
    assert clients[0].is_closed


def test_async_http_client_is_reused_and_closed(monkeypatch):
    clients = []
    original = httpx.AsyncClient
    class TrackingClient(original):
        def __init__(self, **kwargs):
            kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [
                {"index": i, "embedding": [1., 0.]} for i, _ in enumerate(json.loads(request.content)["input"])
            ]}))
            super().__init__(**kwargs)
            clients.append(self)
    monkeypatch.setattr(httpx, "AsyncClient", TrackingClient)
    async def run():
        embedder = OpenRouterEmbedder(api_key="test", dim=2)
        try:
            await embedder.aembed_queries(["first"])
            await embedder.aembed_queries(["second"])
            assert len(clients) == 1 and not clients[0].is_closed
        finally:
            await embedder.aclose()
        assert clients[0].is_closed
    asyncio.run(run())


def test_cache_keeps_hash_keys_and_can_be_disabled(monkeypatch):
    embedder = FakeEmbedder(dim=2)
    embedder.embed_queries(["PRIVATE user input"])
    assert "PRIVATE" not in repr(embedder._query_cache.keys())
    uncached = FakeEmbedder(dim=2, query_cache_size=0)
    calls = []
    original = uncached._embed_queries_uncached
    monkeypatch.setattr(uncached, "_embed_queries_uncached", lambda texts: (calls.append(list(texts)), original(texts))[1])
    uncached.embed_queries(["same"])
    uncached.embed_queries(["same"])
    assert calls == [["same"], ["same"]] and not uncached._query_cache


def test_engine_shares_chat_connection_between_expansion_and_verification(engine, monkeypatch):
    from skill_inject_mcp.schemas import Requirement, SkillInjectRequest
    clients = []
    original = httpx.Client
    def respond(request):
        payload = json.loads(request.content)
        if "response_format" not in payload:
            return httpx.Response(200, json={"choices":[{"message":{"content":'["pip install", "project dependencies"]'}}]})
        candidates = json.loads(payload["messages"][1]["content"])["candidates"]
        verdicts = [{"skill_id": c["skill_id"], "assessment": "supported", "reason": "Mock source check",
                     "evidence": [{"source_id": c["sources"][0]["source_id"]}], "unmet_requirements": []} for c in candidates]
        return httpx.Response(200, json={"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"results":verdicts})}}]})
    class TrackingClient(original):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)
            self.calls = 0
            clients.append(self)
        def post(self, *args, **kwargs):
            self.calls += 1
            return super().post(*args, **kwargs)
    monkeypatch.setattr(httpx, "Client", TrackingClient)
    engine.settings.openrouter_api_key = "test"
    engine.settings.multi_query = True
    engine.settings.verification_mode = "semantic"
    result = engine.resolve(SkillInjectRequest(requirements=[Requirement(id="r", description="Install Python packages")]))
    assert result.match_status == "complete"
    assert len(clients) == 1 and clients[0].calls == 2
    engine.close()
    assert clients[0].is_closed


def test_expanded_queries_use_one_embedding_batch_without_changing_rrf_order():
    from skill_inject_mcp.retrieve.hybrid import HybridRetriever
    calls = []
    class Embeddings:
        def embed_queries(self, texts):
            calls.append(list(texts))
            return [np.array([1. if t == "alpha" else 2.]) for t in texts]
    class Dense:
        def search(self, vector, top_k):
            return [("a", .9), ("b", .8)] if vector[0] == 1 else [("c", .95), ("a", .7)]
    class Sparse:
        def search(self, query, top_k):
            return [("b", 2., 1), ("c", 1., 2)] if query == "alpha" else [("a", 3., 1), ("b", 1., 2)]
    retriever = HybridRetriever(sparse=Sparse(), dense=Dense(), embedder=Embeddings(), skills={})
    hits = retriever.retrieve("alpha", queries=["alpha", "beta"], top_k=3)
    assert calls == [["alpha", "beta"]]
    assert [h.skill_id for h in hits] == ["a", "c", "b"]
    assert [(h.dense_rank, h.sparse_rank) for h in hits] == [(2, 1), (1, 3), (3, 2)]
