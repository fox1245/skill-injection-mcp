from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skill_inject_mcp.codex_adapter import async_prompt_context
from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.embedder import FakeEmbedder
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.index.locks import FileLock
from skill_inject_mcp.registry.scan import SkillRegistry
from tests.embed_server import EmbeddingServer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "skills"


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


def _snapshot_dirs(index_dir: Path) -> list[Path]:
    if not index_dir.is_dir():
        return []
    return [p for p in index_dir.iterdir() if p.is_dir() and p.name.startswith("snapshot-")]


def test_concurrent_processes_embed_the_catalog_once(tmp_path: Path) -> None:
    index_dir = tmp_path / "shared_index"
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir(parents=True)
    start_file = tmp_path / "start.flag"
    expected_skills = len(SkillRegistry().load(FIXTURES))
    assert expected_skills > 0

    worker_script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from skill_inject_mcp.config import Settings\n"
        "from skill_inject_mcp.engine import SkillInjectEngine\n"
        "ready_file = Path(sys.argv[1])\n"
        "start_file = Path(sys.argv[2])\n"
        "ready_file.write_text('ready', encoding='utf-8')\n"
        "deadline = time.monotonic() + 15.0\n"
        "while not start_file.exists():\n"
        "    if time.monotonic() >= deadline:\n"
        "        raise TimeoutError('Start signal not received')\n"
        "    time.sleep(0.01)\n"
        "engine = SkillInjectEngine(Settings(_env_file=None))\n"
        "try:\n"
        "    engine.ensure_index()\n"
        "finally:\n"
        "    engine.close()\n"
    )

    with EmbeddingServer(delay_s=0.3) as server:
        env = _minimal_env(
            tmp_path,
            SKILL_INJECT_INDEX_DIR=str(index_dir),
            SKILL_INJECT_RETRIEVAL_MODE="hybrid",
            SKILL_INJECT_USE_FAKE_EMBEDDER="false",
            SKILL_INJECT_PERSISTENT_EMBEDDING_CACHE="true",
            SKILL_INJECT_EMBEDDING_BASE_URL=server.base_url,
            SKILL_INJECT_EMBEDDING_DIM="16",
        )
        procs: list[subprocess.Popen[str]] = []
        try:
            for idx in range(4):
                ready_file = ready_dir / f"worker-{idx}.ready"
                procs.append(
                    subprocess.Popen(
                        [sys.executable, "-c", worker_script, str(ready_file), str(start_file)],
                        cwd=tmp_path,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )

            deadline = time.monotonic() + 15.0
            while len(list(ready_dir.glob("*.ready"))) < 4:
                for proc in procs:
                    if proc.poll() is not None:
                        out, err = proc.communicate()
                        pytest.fail(f"Worker exited before barrier: {out}\n{err}")
                if time.monotonic() >= deadline:
                    pytest.fail("Timed out waiting for workers to reach start barrier")
                time.sleep(0.02)

            start_file.write_text("go", encoding="utf-8")

            for proc in procs:
                out, err = proc.communicate(timeout=30)
                assert proc.returncode == 0, f"Worker failed:\nstdout={out}\nstderr={err}"
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)

        received = server.inputs
        assert len(received) == expected_skills
        assert len(set(received)) == expected_skills


def test_startup_sweep_removes_only_orphans(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    dead_nolease = index_dir / "snapshot-dead-nolease"
    dead_nolease.mkdir(parents=True)

    dead_lease = index_dir / "snapshot-dead-lease"
    dead_lease.mkdir(parents=True)
    unlocked = FileLock(dead_lease / "lease.lock")
    assert unlocked.try_acquire()
    unlocked.release()

    live_dir = index_dir / "snapshot-live"
    live_dir.mkdir(parents=True)
    live_lock = FileLock(live_dir / "lease.lock")
    assert live_lock.try_acquire()

    keep_dir = index_dir / "keep-me"
    keep_dir.mkdir(parents=True)
    from skill_inject_mcp.index.cache import EmbeddingCache
    EmbeddingCache(index_dir, ("test",), 4)
    cache_file = index_dir / "embedding-cache.sqlite"
    assert cache_file.exists()

    settings = Settings(
        _env_file=None,
        skills_dir=FIXTURES,
        index_dir=index_dir,
        use_fake_embedder=True,
        multi_query=False,
    )
    engine = SkillInjectEngine(settings)
    try:
        engine.ensure_index()
        assert not dead_nolease.exists()
        assert not dead_lease.exists()
        assert live_dir.exists()
        assert keep_dir.exists()
        assert cache_file.exists()
    finally:
        engine.close()
        live_lock.release()

    killed_dir = index_dir / "snapshot-killed-child"
    killed_dir.mkdir(parents=True)
    child_ready = tmp_path / "child_lease.ready"
    child_script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from skill_inject_mcp.index.locks import FileLock\n"
        "lease = FileLock(Path(sys.argv[1]))\n"
        "assert lease.try_acquire()\n"
        "Path(sys.argv[2]).write_text('ready', encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child_script, str(killed_dir / "lease.lock"), str(child_ready)],
        cwd=tmp_path,
        env=_minimal_env(tmp_path, SKILL_INJECT_INDEX_DIR=str(index_dir)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not child_ready.exists():
            if proc.poll() is not None:
                _, stderr = proc.communicate()
                pytest.fail(f"Child exited early: {stderr}")
            if time.monotonic() >= deadline:
                pytest.fail("Timed out waiting for child lease acquisition")
            time.sleep(0.02)
        proc.kill()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert killed_dir.exists()
    restarted = SkillInjectEngine(settings.model_copy())
    try:
        restarted.ensure_index()
        assert not killed_dir.exists()
        assert not live_dir.exists()
        assert keep_dir.exists()
        assert cache_file.exists()
    finally:
        restarted.close()


def test_failed_publication_disposes_the_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = tmp_path / "index"
    engine = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=FIXTURES,
            index_dir=index_dir,
            use_fake_embedder=True,
            multi_query=False,
        )
    )

    def fail_retriever(*args, **kwargs):
        raise RuntimeError("simulated retriever construction failure")

    monkeypatch.setattr("skill_inject_mcp.engine.HybridRetriever", fail_retriever)
    try:
        with pytest.raises(RuntimeError, match="retriever construction failure"):
            engine.ensure_index()
        assert engine._snapshot is None
        assert _snapshot_dirs(index_dir) == []
    finally:
        engine.close()


def test_close_during_build_does_not_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = tmp_path / "index"
    engine = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=FIXTURES,
            index_dir=index_dir,
            use_fake_embedder=True,
            multi_query=False,
        )
    )
    original_embed = FakeEmbedder.embed_documents

    def close_during_embed(self, texts):
        vectors = original_embed(self, texts)
        engine.close()
        return vectors

    monkeypatch.setattr(FakeEmbedder, "embed_documents", close_during_embed)
    with pytest.raises(RuntimeError, match="closed"):
        engine.ensure_index()
    assert engine._closed
    assert engine._snapshot is None
    assert _snapshot_dirs(index_dir) == []


def test_hook_never_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    engine = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=FIXTURES,
            index_dir=index_dir,
            use_fake_embedder=True,
            multi_query=False,
        )
    )

    def forbid_refresh() -> None:
        raise AssertionError("Hook path must never request a background refresh")

    monkeypatch.setattr(engine, "request_background_refresh", forbid_refresh)
    try:
        result = asyncio.run(async_prompt_context(engine, "Install Python project dependencies with pip"))
        context = result["hookSpecificOutput"]["additionalContext"]
        assert "unavailable; no published index snapshot is ready" in context
        assert _snapshot_dirs(index_dir) == []
        assert not (index_dir / "build.lock").exists()
    finally:
        engine.close()


def test_agy_hook_does_not_import_engine_or_create_index(tmp_path: Path) -> None:
    index_dir = tmp_path / "agy_index"
    env = _minimal_env(tmp_path, SKILL_INJECT_INDEX_DIR=str(index_dir))

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "agy_hook.py")],
        input=json.dumps({"hook_event_name": "UserPromptSubmit"}),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload.get("continue") is True
    assert not index_dir.exists()

    check_import = (
        "import sys\n"
        "import scripts.agy_hook as agy_hook\n"
        "agy_hook.handle_hook_event({'hook_event_name': 'UserPromptSubmit'})\n"
        "assert 'skill_inject_mcp.engine' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", check_import],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert not index_dir.exists()


def test_hook_paths_leave_no_generation_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = tmp_path / "index"
    engine = SkillInjectEngine(
        Settings(
            _env_file=None,
            skills_dir=FIXTURES,
            index_dir=index_dir,
            use_fake_embedder=True,
            multi_query=False,
        )
    )
    try:
        engine.ensure_index()
        leased = engine.acquire_hook_snapshot()
        assert leased is not None
        try:
            async def raise_error(texts, deadline=None):
                raise RuntimeError("embedding boom")

            monkeypatch.setattr(engine._snapshot.embedder, "aembed_queries", raise_error)
            res_err = asyncio.run(async_prompt_context(engine, "Install Python project dependencies with pip"))
            assert "BM25-only fallback" in res_err["hookSpecificOutput"]["additionalContext"]

            async def raise_timeout(texts, deadline=None):
                await asyncio.sleep(1.0)

            monkeypatch.setattr(engine._snapshot.embedder, "aembed_queries", raise_timeout)
            engine.settings.hook_timeout_s = 0.02
            res_timeout = asyncio.run(async_prompt_context(engine, "Install Python project dependencies with pip"))
            assert "BM25-only fallback" in res_timeout["hookSpecificOutput"]["additionalContext"]
        finally:
            leased.release()
    finally:
        engine.close()

    assert _snapshot_dirs(index_dir) == []


def test_sweep_requires_the_build_lock_of_this_index_directory(tmp_path: Path) -> None:
    from skill_inject_mcp.index.snapshot import sweep_orphan_generations

    index_dir = tmp_path / "index"
    (index_dir / "snapshot-orphan").mkdir(parents=True)
    other = FileLock(tmp_path / "other" / "build.lock")
    assert other.try_acquire()
    try:
        with pytest.raises(RuntimeError):
            sweep_orphan_generations(index_dir, other)
        with pytest.raises(RuntimeError):
            sweep_orphan_generations(index_dir, FileLock(index_dir / "build.lock"))  # not held
    finally:
        other.release()
    assert (index_dir / "snapshot-orphan").exists()


def test_discarding_an_unpublished_generation_survives_a_failing_close(tmp_path: Path) -> None:
    from skill_inject_mcp.index.snapshot import _create_generation, _discard_generation

    directory, lease = _create_generation(tmp_path / "index")

    class Broken:
        def close(self) -> None:
            raise OSError("close failed")

    _discard_generation(directory, tmp_path / "index", lease, Broken(), None)
    assert not directory.exists()
    assert not lease.held


def test_a_generation_already_removed_by_another_process_is_not_an_error(tmp_path: Path) -> None:
    import shutil

    from skill_inject_mcp.index.snapshot import _create_generation, _remove_generation

    directory, lease = _create_generation(tmp_path / "index")
    lease.release()
    shutil.rmtree(directory)
    _remove_generation(directory, tmp_path / "index")  # must not raise


def test_one_unusable_lease_file_does_not_stop_the_sweep_or_the_build(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    bad = index_dir / "snapshot-bad-lease"
    (bad / "lease.lock").mkdir(parents=True)  # a lease path that cannot be opened as a lock file
    orphan = index_dir / "snapshot-orphan"
    orphan.mkdir(parents=True)
    engine = SkillInjectEngine(Settings(
        _env_file=None, skills_dir=FIXTURES, index_dir=index_dir,
        use_fake_embedder=True, multi_query=False,
    ))
    try:
        engine.ensure_index()
        assert not orphan.exists()
        assert bad.exists()
    finally:
        engine.close()
