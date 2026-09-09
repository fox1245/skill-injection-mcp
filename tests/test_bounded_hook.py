from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx

from skill_inject_mcp.codex_adapter import async_prompt_context
from skill_inject_mcp.embed.embedder import OpenRouterEmbedder
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest


QUERY = "Install Python project dependencies with pip"


def context(result):
    return result["hookSpecificOutput"]["additionalContext"]


def test_timed_out_http_request_is_cancelled_and_does_not_block_followup(engine):
    async def run():
        cancelled = asyncio.Event()
        calls = []
        async def respond(request):
            calls.append(request)
            if len(calls) == 1:
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.set()
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.] + [0.] * 1023}]})
        remote = OpenRouterEmbedder(api_key="test", dim=1024)
        remote._async_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        remote._async_loop = asyncio.get_running_loop()
        snapshot = engine._snapshot
        snapshot.embedder = remote
        engine.settings.hook_timeout_s = .05
        try:
            started = time.monotonic()
            first = await async_prompt_context(engine, QUERY)
            assert time.monotonic() - started < .4
            assert cancelled.is_set()
            assert "BM25-only" in context(first) and "unverified" in context(first)
            assert "package-installer" in context(first)
            assert not remote._query_cache
            second = await async_prompt_context(engine, QUERY)
            assert "hybrid" in context(second)
            assert len(calls) == 2  # The timed-out request never poisoned the query cache.
            result = await asyncio.wait_for(asyncio.to_thread(engine.resolve, SkillInjectRequest(requirements=[
                Requirement(id="r", description=QUERY)])), timeout=1)
            assert result.match_status == "complete"
        finally:
            await remote.aclose()
            remote.close()
    asyncio.run(run())


def test_hook_does_not_wait_for_long_resolve_lock(engine):
    entered, release = threading.Event(), threading.Event()
    def hold():
        with engine._lock:
            entered.set()
            release.wait(2)
    worker = threading.Thread(target=hold)
    worker.start()
    assert entered.wait(1)
    try:
        started = time.monotonic()
        result = asyncio.run(async_prompt_context(engine, QUERY))
        assert time.monotonic() - started < .5
        assert "package-installer" in context(result)
    finally:
        release.set()
        worker.join(2)


def test_timeout_omits_bm25_hits_without_positive_description_coverage(engine, monkeypatch):
    async def slow(texts):
        await asyncio.sleep(10)
    monkeypatch.setattr(engine._snapshot.embedder, "aembed_queries", slow)
    engine.settings.hook_timeout_s = .03
    result = asyncio.run(async_prompt_context(engine, "Use Python to fold origami cranes"))
    output = context(result)
    assert "BM25-only" in output
    assert json.loads(output.split("\n")[1]) == []


def test_snapshot_lease_survives_replacement_until_released(engine):
    leased = engine.acquire_hook_snapshot()
    original_directory = leased.directory
    engine.reindex()
    assert engine._snapshot is not leased and original_directory.exists()
    assert leased.sparse.search_readonly("Python packages")
    leased.release()
    assert not original_directory.exists()


def test_retired_index_file_cleanup_does_not_delay_hook_reply(engine, monkeypatch):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        snapshot = engine._snapshot
        original_embed = snapshot.embedder.aembed_queries
        original_dispose = snapshot._dispose
        async def delayed(texts):
            entered.set()
            await release.wait()
            return await original_embed(texts)
        def slow_dispose():
            time.sleep(.3)
            original_dispose()
        monkeypatch.setattr(snapshot.embedder, "aembed_queries", delayed)
        monkeypatch.setattr(snapshot, "_dispose", slow_dispose)
        task = asyncio.create_task(async_prompt_context(engine, QUERY))
        await entered.wait()
        await asyncio.to_thread(engine.reindex)
        started = time.monotonic()
        release.set()
        result = await task
        assert time.monotonic() - started < .2 and "package-installer" in context(result)
        assert snapshot.directory.exists()
        await asyncio.gather(*tuple(engine._cleanup_jobs))
        assert not snapshot.directory.exists()
    asyncio.run(run())


def test_cold_hook_returns_without_waiting_for_background_index(engine, monkeypatch):
    engine.close()
    # A new engine has no published snapshot yet.
    from skill_inject_mcp.engine import SkillInjectEngine
    cold = SkillInjectEngine(engine.settings.model_copy())
    entered, release = threading.Event(), threading.Event()
    original = cold.ensure_index
    def delayed():
        entered.set()
        release.wait(2)
        return original()
    monkeypatch.setattr(cold, "ensure_index", delayed)
    try:
        started = time.monotonic()
        result = asyncio.run(async_prompt_context(cold, QUERY))
        assert time.monotonic() - started < .3
        assert "unavailable" in context(result)
        assert entered.wait(1)
    finally:
        release.set()
        cold.close()


def test_explicit_tool_skill_directory_does_not_leak_into_global_hook(engine, tmp_path, monkeypatch):
    path = tmp_path / "foreign" / "private-fixture" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\nname: private-fixture\ndescription: Install packages\n---\nInstall packages", encoding="utf-8")
    engine.ensure_index(path.parent.parent)
    monkeypatch.setattr(engine, "request_background_refresh", lambda: None)
    result = asyncio.run(async_prompt_context(engine, QUERY))
    assert "private-fixture" not in context(result)
    assert "unavailable" in context(result)


def test_failed_refresh_uses_explicit_last_known_catalog(engine, monkeypatch):
    monkeypatch.setattr(engine, "hook_catalog_state", lambda snapshot: "last-known")
    result = asyncio.run(async_prompt_context(engine, QUERY))
    assert "last-known" in context(result)
    assert "package-installer" in context(result)


def test_mcp_event_loop_stays_responsive_during_sync_resolve(engine, monkeypatch):
    from skill_inject_mcp import server
    entered, release = threading.Event(), threading.Event()
    original = engine.resolve
    def delayed(request):
        entered.set()
        release.wait(1)
        return original(request)
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(engine, "resolve", delayed)
    async def run():
        task = asyncio.create_task(server.resolve_skills(SkillInjectRequest(requirements=[Requirement(id="r", description=QUERY)])))
        try:
            await asyncio.wait_for(asyncio.to_thread(entered.wait), .3)
            started = time.monotonic()
            hook = await asyncio.wait_for(server.codex_prompt_hook(QUERY), .3)
            assert time.monotonic() - started < .3 and "package-installer" in context(hook)
        finally:
            release.set()
            await task
    asyncio.run(run())
