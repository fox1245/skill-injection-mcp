from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skill_inject_mcp.index.locks import FileLock

ROOT = Path(__file__).resolve().parents[1]


def _minimal_env(tmp_path: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "OPENROUTER_API_KEY": "test-dummy-key",
        "SKILL_INJECT_DENSE_BACKEND": "numpy",
        "SKILL_INJECT_MULTI_QUERY": "false",
        "SKILL_INJECT_VERIFICATION_MODE": "lexical",
    }
    for key in ("SYSTEMROOT", "WINDIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def test_second_lock_cannot_try_acquire_while_first_held_and_can_after_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "resource.lock"
    first = FileLock(lock_path)
    second = FileLock(lock_path)
    try:
        assert not first.held
        assert first.try_acquire()
        assert first.held
        assert first.try_acquire()
        assert not second.try_acquire()
        assert not second.held

        first.release()
        assert not first.held
        first.release()

        assert second.try_acquire()
        assert second.held
    finally:
        first.release()
        second.release()


def test_acquire_times_out_while_lock_is_held(tmp_path: Path) -> None:
    lock_path = tmp_path / "resource.lock"
    with FileLock(lock_path, timeout_s=0.2, poll_s=0.01) as holder:
        assert holder.held
        contender = FileLock(lock_path)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            contender.acquire(timeout_s=0.2, poll_s=0.01)
        assert time.monotonic() - started >= 0.15
        assert not contender.held


def test_lock_held_by_child_process_is_acquirable_after_child_is_killed(tmp_path: Path) -> None:
    lock_path = tmp_path / "child.lock"
    ready_path = tmp_path / "ready.flag"
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from skill_inject_mcp.index.locks import FileLock\n"
        "lock = FileLock(Path(sys.argv[1]))\n"
        "assert lock.try_acquire()\n"
        "Path(sys.argv[2]).write_text('ready', encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path), str(ready_path)],
        cwd=tmp_path,
        env=_minimal_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not ready_path.exists():
            if proc.poll() is not None:
                _, stderr = proc.communicate()
                pytest.fail(f"Child exited early: {stderr}")
            if time.monotonic() >= deadline:
                pytest.fail("Timed out waiting for child process to acquire lock")
            time.sleep(0.02)

        parent_lock = FileLock(lock_path)
        assert not parent_lock.try_acquire()

        proc.kill()
        proc.wait(timeout=10)

        parent_lock.acquire(timeout_s=2.0, poll_s=0.02)
        try:
            assert parent_lock.held
        finally:
            parent_lock.release()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
