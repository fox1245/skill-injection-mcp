from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import email.utils
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import pytest
from pydantic import ValidationError

from skill_inject_mcp.codex_adapter import async_prompt_context
from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.admission import EmbeddingUnavailable, build_admission
from skill_inject_mcp.embed.embedder import build_embedder
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.index.locks import FileLock
from tests.embed_server import EmbeddingServer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "skills"
CLIENT_SCRIPT = ROOT / "tests" / "embed_client.py"


def _minimal_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "OPENROUTER_API_KEY": "test-dummy-key",
        "SKILL_INJECT_SKILLS_DIR": str(FIXTURES),
        "SKILL_INJECT_INDEX_DIR": str(tmp_path / "index"),
        "SKILL_INJECT_DENSE_BACKEND": "numpy",
        "SKILL_INJECT_MULTI_QUERY": "false",
        "SKILL_INJECT_VERIFICATION_MODE": "lexical",
    }
    for key in ("SYSTEMROOT", "WINDIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(extra)
    return env


def _build_test_embedder(settings: Settings, *, api_key: str, dim: int = 2, admission=None):
    return build_embedder(
        api_key=api_key,
        model=settings.embedding_model,
        dim=dim,
        base_url=settings.embedding_base_url,
        timeout_s=settings.embedding_timeout_s,
        connect_timeout_s=settings.http_connect_timeout_s,
        write_timeout_s=settings.http_write_timeout_s,
        pool_timeout_s=settings.http_pool_timeout_s,
        admission=admission,
    )


def _run_client_processes(
    tmp_path: Path,
    *,
    num_children: int,
    max_concurrency: int,
    delay_s: float = 0.3,
    disable_queue: bool = False,
) -> EmbeddingServer:
    run_dir = tmp_path / f"run-{max_concurrency}-{int(disable_queue)}"
    ready_dir = run_dir / "ready"
    ready_dir.mkdir(parents=True)
    start_file = run_dir / "start.flag"
    queue_dir = run_dir / "queue"

    with EmbeddingServer(delay_s=delay_s) as server:
        env = _minimal_env(
            tmp_path,
            SKILL_INJECT_INDEX_DIR=str(run_dir / "index"),
            SKILL_INJECT_EMBEDDING_QUEUE_DIR=str(queue_dir),
            SKILL_INJECT_EMBEDDING_MAX_CONCURRENCY=str(max_concurrency),
            SKILL_INJECT_EMBEDDING_BASE_URL=server.base_url,
            SKILL_INJECT_EMBEDDING_DIM="16",
            SKILL_INJECT_LOCK_POLL_S="0.01",
        )
        procs: list[subprocess.Popen[str]] = []
        try:
            for idx in range(num_children):
                ready_file = ready_dir / f"client-{idx}.ready"
                cmd = [sys.executable, str(CLIENT_SCRIPT), str(ready_file), str(start_file)]
                if disable_queue:
                    cmd.append("--no-admission")
                procs.append(
                    subprocess.Popen(
                        cmd,
                        cwd=tmp_path,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            deadline = time.monotonic() + 15.0
            while len(list(ready_dir.glob("*.ready"))) < num_children:
                for proc in procs:
                    if proc.poll() is not None:
                        out, err = proc.communicate()
                        pytest.fail(f"Client exited before barrier: {out}\n{err}")
                if time.monotonic() >= deadline:
                    pytest.fail("Timed out waiting for client processes")
                time.sleep(0.01)

            start_file.write_text("go", encoding="utf-8")

            for proc in procs:
                out, err = proc.communicate(timeout=45)
                assert proc.returncode == 0, f"Client failed:\nstdout={out}\nstderr={err}"
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
        return server


@pytest.mark.parametrize("n_slots", [2, 1])
def test_shared_queue_limits_concurrency_across_processes(tmp_path: Path, n_slots: int) -> None:
    num_children = 3 * n_slots + 2
    server = _run_client_processes(
        tmp_path,
        num_children=num_children,
        max_concurrency=n_slots,
        delay_s=0.3,
        disable_queue=False,
    )
    assert server.request_count == 2 * num_children
    assert server.max_inflight == n_slots


def test_without_queue_exceeds_limit(tmp_path: Path) -> None:
    n_slots = 2
    num_children = 3 * n_slots + 2
    server = _run_client_processes(
        tmp_path,
        num_children=num_children,
        max_concurrency=n_slots,
        delay_s=0.3,
        disable_queue=True,
    )
    assert server.request_count == 2 * num_children
    assert server.max_inflight > n_slots


def test_retry_after_honoured_and_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_now = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)
    retry_date = email.utils.format_datetime(fixed_now + timedelta(seconds=2), usegmt=True)

    for max_wait, expected_sleeps in [(30.0, [3.0, 2.0]), (1.5, [1.5, 1.5])]:
        responses = [
            httpx.Response(429, headers={"Retry-After": "3"}, json={"error": "rate_limit"}),
            httpx.Response(429, headers={"Retry-After": retry_date}, json={"error": "rate_limit"}),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]}),
        ]
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda transport, request, r=responses: r.pop(0))
        sleeps: list[float] = []
        settings = Settings(
            _env_file=None,
            index_dir=tmp_path / f"index-{max_wait}",
            OPENROUTER_API_KEY="test-key",
            embedding_dim=2,
            embedding_max_concurrency=1,
            embedding_max_retries=3,
            embedding_retry_base_backoff_s=0.5,
            embedding_retry_max_wait_s=max_wait,
            embedding_retry_jitter=0.0,
        )
        admission = build_admission(settings, sleep=sleeps.append, now=lambda: fixed_now)
        embedder, _ = _build_test_embedder(
            settings,
            api_key="test-key",
            dim=2,
            admission=admission,
        )
        try:
            vectors = embedder.embed_queries(["install packages"])
            assert len(vectors) == 1
            assert sleeps == expected_sleeps
            assert admission.queue is not None
            for slot_path in admission.queue.slot_paths():
                probe = FileLock(slot_path)
                assert probe.try_acquire()
                probe.release()
            # Query cache hits must not acquire a slot even when all slots are held.
            held = [FileLock(p) for p in admission.queue.slot_paths()]
            for lock in held:
                assert lock.try_acquire()
            try:
                cached_vectors = embedder.embed_queries(["install packages"])
                assert len(cached_vectors) == 1
            finally:
                for lock in held:
                    lock.release()
        finally:
            embedder.close()


def test_deadline_stops_before_retry_wait_exceeding_remaining_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = 0

    def always_429(transport, request):
        nonlocal requests
        requests += 1
        return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": "rate_limit"})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", always_429)
    sleeps: list[float] = []
    now = [100.0]
    settings = Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        OPENROUTER_API_KEY="secret-token-do-not-leak",
        embedding_dim=2,
        embedding_max_retries=3,
        embedding_retry_jitter=0.0,
    )
    admission = build_admission(settings, clock=lambda: now[0], sleep=sleeps.append)
    embedder, _ = _build_test_embedder(settings, api_key="secret-token-do-not-leak", dim=2, admission=admission)
    try:
        with pytest.raises(EmbeddingUnavailable) as exc_info:
            embedder.embed_documents(["document"], deadline=now[0] + 1.0)
        assert exc_info.value.reason == "deadline"
        assert "secret-token-do-not-leak" not in str(exc_info.value)
        assert requests == 1
        assert sleeps == []
    finally:
        embedder.close()


def test_retries_exhausted_and_non_retried_status_raises_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = 0

    def always_503(transport, request):
        nonlocal requests
        requests += 1
        return httpx.Response(503, json={"error": "unavailable"})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", always_503)
    sleeps: list[float] = []
    settings = Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        OPENROUTER_API_KEY="test-key",
        embedding_dim=2,
        embedding_max_retries=3,
        embedding_retry_base_backoff_s=0.5,
        embedding_retry_max_wait_s=1.5,
        embedding_retry_jitter=0.0,
    )
    admission = build_admission(settings, sleep=sleeps.append)
    embedder, _ = _build_test_embedder(settings, api_key="test-key", dim=2, admission=admission)
    try:
        with pytest.raises(EmbeddingUnavailable) as exc_info:
            embedder.embed_documents(["document"])
        assert exc_info.value.reason == "retries_exhausted"
        assert requests == 4
        assert sleeps == [0.5, 1.0, 1.5]

        sleeps.clear()
        bad_requests = 0

        def always_400(transport, request):
            nonlocal bad_requests
            bad_requests += 1
            return httpx.Response(400, json={"error": "bad_request"})

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", always_400)
        with pytest.raises(httpx.HTTPStatusError):
            embedder.embed_documents(["document"])
        assert bad_requests == 1
        assert sleeps == []
    finally:
        embedder.close()


def test_slot_wait_deadline_and_async_non_blocking_acquire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[object] = []
    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        lambda transport, request: (requests.append(request), httpx.Response(200, json={"data": []}))[1],
    )
    settings = Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        OPENROUTER_API_KEY="test-key",
        embedding_dim=2,
        embedding_max_concurrency=2,
        embedding_queue_max_wait_s=0.15,
        lock_poll_s=0.01,
    )
    admission = build_admission(settings)
    assert admission.queue is not None
    embedder, _ = _build_test_embedder(settings, api_key="test-key", dim=2, admission=admission)
    held_locks = [FileLock(path) for path in admission.queue.slot_paths()]
    for lock in held_locks:
        assert lock.try_acquire()
    try:
        started = time.monotonic()
        with pytest.raises(EmbeddingUnavailable) as exc_info:
            embedder.embed_documents(["doc"], deadline=started + 0.2)
        elapsed = time.monotonic() - started
        assert exc_info.value.reason == "deadline"
        assert elapsed < 1.0
        assert requests == []

        with pytest.raises(EmbeddingUnavailable) as queue_exc:
            admission.queue.acquire(deadline=None)
        assert queue_exc.value.reason == "queue_timeout"

        async def run_async_variant() -> None:
            ticks = 0
            stop_ticker = asyncio.Event()

            async def ticker() -> None:
                nonlocal ticks
                while not stop_ticker.is_set():
                    await asyncio.sleep(0.01)
                    ticks += 1

            async def release_one_slot() -> None:
                await asyncio.sleep(0.08)
                held_locks[0].release()

            ticker_task = asyncio.create_task(ticker())
            releaser_task = asyncio.create_task(release_one_slot())
            try:
                acquired = await admission.queue.aacquire(deadline=time.monotonic() + 2.0)
                try:
                    assert acquired.held
                    assert ticks >= 2
                finally:
                    acquired.release()
            finally:
                stop_ticker.set()
                await asyncio.gather(ticker_task, releaser_task)

        asyncio.run(run_async_variant())
    finally:
        for lock in held_locks:
            lock.release()
        embedder.close()


def test_slot_is_released_on_every_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None,
        index_dir=tmp_path / "index",
        OPENROUTER_API_KEY="test-key",
        embedding_dim=2,
        embedding_max_concurrency=2,
        lock_poll_s=0.01,
    )
    admission = build_admission(settings)
    assert admission.queue is not None
    embedder, _ = _build_test_embedder(settings, api_key="test-key", dim=2, admission=admission)

    def assert_all_slots_free() -> None:
        assert admission.queue is not None
        for path in admission.queue.slot_paths():
            probe = FileLock(path)
            assert probe.try_acquire(), f"Slot still locked: {path}"
            probe.release()

    try:
        # 1. Read timeout
        def fail_timeout(transport, request):
            raise httpx.ReadTimeout("simulated read timeout", request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail_timeout)
        with pytest.raises(RuntimeError, match="read_timeout_s="):
            embedder.embed_documents(["doc"])
        assert_all_slots_free()

        # 2. Exception from _decode
        monkeypatch.setattr(
            httpx.HTTPTransport,
            "handle_request",
            lambda transport, request: httpx.Response(200, json={"data": [{"index": 5, "embedding": [1.0, 0.0]}]}),
        )
        with pytest.raises(ValueError, match="Invalid embedding response"):
            embedder.embed_documents(["doc"])
        assert_all_slots_free()

        # 3. asyncio.wait_for cancellation of an in-flight async request
        async def run_cancel() -> None:
            in_flight = asyncio.Event()

            async def slow_async(transport, request):
                in_flight.set()
                await asyncio.sleep(10.0)
                return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

            monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", slow_async)
            task = asyncio.create_task(embedder.aembed_queries(["async query"]))
            await in_flight.wait()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=0.03)
            await embedder.aclose()

        asyncio.run(run_cancel())
        assert_all_slots_free()
    finally:
        embedder.close()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("embedding_max_concurrency", 0),
        ("embedding_max_concurrency", -2),
        ("embedding_queue_max_wait_s", 0.0),
        ("embedding_queue_max_wait_s", -1.0),
        ("embedding_max_retries", -1),
        ("embedding_retry_base_backoff_s", 0.0),
        ("embedding_retry_base_backoff_s", -0.5),
        ("embedding_retry_max_wait_s", 0.0),
        ("embedding_retry_max_wait_s", -2.0),
        ("embedding_retry_jitter", -0.1),
        ("embedding_retry_jitter", 1.1),
    ],
)
def test_settings_validation_rejects_invalid_queue_and_retry_values(field: str, bad_value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: bad_value})


def test_settings_can_be_configured_via_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qdir = tmp_path / "custom_queue"
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_MAX_CONCURRENCY", "5")
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_QUEUE_DIR", str(qdir))
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_QUEUE_MAX_WAIT_S", "42.5")
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_MAX_RETRIES", "4")
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_RETRY_STATUSES", "[429, 503]")
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_RETRY_BASE_BACKOFF_S", "0.25")
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_RETRY_MAX_WAIT_S", "12.0")
    monkeypatch.setenv("SKILL_INJECT_EMBEDDING_RETRY_JITTER", "0.5")

    cfg = Settings(_env_file=None)
    assert cfg.embedding_max_concurrency == 5
    assert cfg.embedding_queue_dir == qdir
    assert cfg.embedding_queue_max_wait_s == 42.5
    assert cfg.embedding_max_retries == 4
    assert cfg.embedding_retry_statuses == [429, 503]
    assert cfg.embedding_retry_base_backoff_s == 0.25
    assert cfg.embedding_retry_max_wait_s == 12.0
    assert cfg.embedding_retry_jitter == 0.5


def test_hook_with_429_long_retry_after_returns_promptly_after_one_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sync_ok(transport, request):
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))]},
        )

    async_requests = 0

    async def async_always_429(transport, request):
        nonlocal async_requests
        async_requests += 1
        return httpx.Response(429, headers={"Retry-After": "30"}, json={"error": "rate_limit"})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", sync_ok)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_always_429)

    engine = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=FIXTURES,
            index_dir=tmp_path / "index",
            OPENROUTER_API_KEY="test-key",
            use_fake_embedder=False,
            embedding_dim=2,
            hook_timeout_s=2.0,
            multi_query=False,
        )
    )
    try:
        engine.ensure_index()
        started = time.monotonic()
        result = asyncio.run(async_prompt_context(engine, "Install Python project dependencies with pip"))
        elapsed = time.monotonic() - started
        context = result["hookSpecificOutput"]["additionalContext"]
        assert elapsed < 0.8
        assert async_requests == 1
        assert "BM25-only fallback (embedding unavailable: deadline)" in context
    finally:
        engine.close()
