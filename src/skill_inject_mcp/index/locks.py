"""Cross-process advisory file locks released automatically on process exit."""
from __future__ import annotations

import errno
import os
import sys
import time
from pathlib import Path


# errno values that mean "the lock is held by someone else"; any other OSError is a real failure and propagates.
_BUSY_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK})


def _is_busy(exc: OSError) -> bool:
    return isinstance(exc, (BlockingIOError, PermissionError)) or exc.errno in _BUSY_ERRNOS


if sys.platform == "win32":
    import msvcrt

    def _lock_fd(fd: int) -> bool:
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\x00")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if _is_busy(exc):
                return False
            raise

    def _unlock_fd(fd: int) -> None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _lock_fd(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if _is_busy(exc):
                return False
            raise

    def _unlock_fd(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


class FileLock:
    """Process-scoped exclusive advisory lock backed by an OS file descriptor."""

    def __init__(
        self,
        path: Path | str,
        *,
        timeout_s: float | None = None,
        poll_s: float | None = None,
    ) -> None:
        self.path = Path(path)
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOINHERIT"):
            flags |= os.O_NOINHERIT
        fd = os.open(self.path, flags, 0o600)
        try:
            if _lock_fd(fd):
                self._fd = fd
                return True
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
        return False

    def acquire(
        self,
        timeout_s: float | None = None,
        poll_s: float | None = None,
    ) -> FileLock:
        if self.try_acquire():
            return self
        wait_timeout = self._timeout_s if timeout_s is None else timeout_s
        wait_poll = self._poll_s if poll_s is None else poll_s
        if wait_timeout is None or wait_timeout <= 0:
            raise TimeoutError(f"Timed out acquiring lock: {self.path}")
        if wait_poll is None or wait_poll <= 0:
            wait_poll = wait_timeout
        deadline = time.monotonic() + wait_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out acquiring lock: {self.path}")
            time.sleep(min(wait_poll, remaining))
            if self.try_acquire():
                return self

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            _unlock_fd(fd)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> FileLock:
        if not self.held:
            self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
