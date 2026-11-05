"""Cross-process slot queue, retry policy, and deadline admission for embeddings."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import email.utils
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Awaitable, Callable, Mapping

import httpx

from skill_inject_mcp.config import Settings
from skill_inject_mcp.index.locks import FileLock


class EmbeddingUnavailable(RuntimeError):
    """Raised when an embedding request cannot proceed within its queue, retry, or deadline budget.

    ``reason`` is one of ``deadline``, ``queue_timeout``, ``retries_exhausted`` and ``retry_after``.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or f"Embedding service unavailable ({reason})")


class SlotQueue:
    """Process-shared N-slot semaphore backed by advisory file locks per embedding endpoint."""

    def __init__(
        self,
        queue_dir: Path,
        base_url: str,
        max_concurrency: int,
        max_wait_s: float,
        poll_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.base_url = base_url.rstrip("/")
        digest = hashlib.sha256(self.base_url.encode("utf-8")).hexdigest()[:16]
        self.slot_dir = self.queue_dir / digest
        self.max_concurrency = max_concurrency
        self.max_wait_s = max_wait_s
        self.poll_s = poll_s
        self._clock = clock
        self._sleep = sleep
        self._asleep = asleep

    def slot_paths(self) -> list[Path]:
        return [self.slot_dir / f"slot-{idx}.lock" for idx in range(self.max_concurrency)]

    def _try_acquire_any(self) -> FileLock | None:
        for path in self.slot_paths():
            lock = FileLock(path)
            if lock.try_acquire():
                return lock
        return None

    def acquire(self, deadline: float | None = None) -> FileLock:
        started = self._clock()
        if deadline is not None and started >= deadline:
            raise EmbeddingUnavailable("deadline")
        limit, reason = started + self.max_wait_s, "queue_timeout"
        if deadline is not None and deadline <= limit:
            limit, reason = deadline, "deadline"
        while True:
            slot = self._try_acquire_any()
            if slot is not None:
                return slot
            remaining = limit - self._clock()
            if remaining <= 0:
                raise EmbeddingUnavailable(reason)
            self._sleep(min(self.poll_s, remaining))
            if self._clock() >= limit:
                raise EmbeddingUnavailable(reason)

    async def aacquire(self, deadline: float | None = None) -> FileLock:
        started = self._clock()
        if deadline is not None and started >= deadline:
            raise EmbeddingUnavailable("deadline")
        limit, reason = started + self.max_wait_s, "queue_timeout"
        if deadline is not None and deadline <= limit:
            limit, reason = deadline, "deadline"
        while True:
            slot = self._try_acquire_any()
            if slot is not None:
                return slot
            remaining = limit - self._clock()
            if remaining <= 0:
                raise EmbeddingUnavailable(reason)
            await self._asleep(min(self.poll_s, remaining))
            if self._clock() >= limit:
                raise EmbeddingUnavailable(reason)


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RetryPolicy:
    """Retry backoff computation honouring Retry-After headers, cap, and additive jitter."""

    max_retries: int
    retry_statuses: tuple[int, ...]
    base_backoff_s: float
    max_wait_s: float
    jitter: float
    now: Callable[[], datetime] | datetime | float = field(default=_default_now, repr=False)
    rng: random.Random = field(default_factory=random.Random, repr=False)

    def _current_datetime(self) -> datetime:
        current = self.now() if callable(self.now) else self.now
        if isinstance(current, (int, float)):
            return datetime.fromtimestamp(float(current), tz=timezone.utc)
        if isinstance(current, datetime):
            return current if current.tzinfo is not None else current.replace(tzinfo=timezone.utc)
        raise TypeError("Unsupported clock value for RetryPolicy.now")

    def _parse_retry_after(self, headers: Mapping[str, str] | None) -> float | None:
        if headers is None:
            return None
        raw = headers.get("retry-after")
        if raw is None:
            for key, value in headers.items():
                if key.lower() == "retry-after":
                    raw = value
                    break
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except (ValueError, TypeError):
            try:
                parsed_dt = email.utils.parsedate_to_datetime(text)
            except (ValueError, TypeError, IndexError, OverflowError):
                return None
            if parsed_dt is None:
                return None
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            delta = (parsed_dt - self._current_datetime()).total_seconds()
            if math.isfinite(delta) and delta >= 0.0:
                return float(delta)
            return None
        else:
            if math.isfinite(seconds) and seconds >= 0.0:
                return float(seconds)
            return None

    def delay_s(self, attempt: int, headers: Mapping[str, str] | None = None) -> float:
        """Wait before the next attempt.

        The server's ``Retry-After`` is a floor: the wait is the larger of it and the exponential backoff.
        A ``Retry-After`` beyond ``max_wait_s`` cannot be honoured within the configured bound, so the caller
        gives up (``EmbeddingUnavailable("retry_after")``) instead of retrying early. Jitter only adds to the wait.
        """
        retry_after = self._parse_retry_after(headers)
        if retry_after is not None and retry_after > self.max_wait_s:
            raise EmbeddingUnavailable("retry_after")
        backoff = min(self.base_backoff_s * (2.0 ** attempt), self.max_wait_s)
        base = max(backoff, retry_after) if retry_after is not None else backoff
        if self.jitter <= 0.0:
            return float(base)
        u = self.rng.random() if hasattr(self.rng, "random") else float(self.rng())
        return float(base * (1.0 + self.jitter * u))


class EmbeddingAdmission:
    """Coordinate slot acquisition, bounded retries, and deadlines for embedding HTTP calls."""

    def __init__(
        self,
        queue: SlotQueue | None,
        policy: RetryPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.queue = queue
        self.policy = policy
        self.clock = clock
        self.sleep = sleep
        self.asleep = asleep

    def execute(
        self,
        send: Callable[[float | None], httpx.Response],
        *,
        deadline: float | None = None,
    ) -> httpx.Response:
        attempt = 0
        retry_statuses = set(self.policy.retry_statuses)
        while True:
            slot = self.queue.acquire(deadline) if self.queue is not None else None
            try:
                remaining: float | None = None
                if deadline is not None:
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        raise EmbeddingUnavailable("deadline")
                response = send(remaining)
            finally:
                if slot is not None:
                    slot.release()
            if response.status_code in retry_statuses:
                if attempt >= self.policy.max_retries:
                    raise EmbeddingUnavailable("retries_exhausted")
                wait = self.policy.delay_s(attempt, response.headers)
                if deadline is not None and self.clock() + wait > deadline:
                    raise EmbeddingUnavailable("deadline")
                self.sleep(wait)
                attempt += 1
                continue
            response.raise_for_status()
            return response

    async def aexecute(
        self,
        send: Callable[[float | None], Awaitable[httpx.Response]],
        *,
        deadline: float | None = None,
    ) -> httpx.Response:
        attempt = 0
        retry_statuses = set(self.policy.retry_statuses)
        while True:
            slot = await self.queue.aacquire(deadline) if self.queue is not None else None
            try:
                remaining: float | None = None
                if deadline is not None:
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        raise EmbeddingUnavailable("deadline")
                response = await send(remaining)
            finally:
                if slot is not None:
                    slot.release()
            if response.status_code in retry_statuses:
                if attempt >= self.policy.max_retries:
                    raise EmbeddingUnavailable("retries_exhausted")
                wait = self.policy.delay_s(attempt, response.headers)
                if deadline is not None and self.clock() + wait > deadline:
                    raise EmbeddingUnavailable("deadline")
                await self.asleep(wait)
                attempt += 1
                continue
            response.raise_for_status()
            return response


def build_admission(
    settings: Settings,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    asleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], datetime] | datetime | float | None = None,
    rng: random.Random | None = None,
) -> EmbeddingAdmission:
    queue_dir = (
        Path(settings.embedding_queue_dir).expanduser().resolve()
        if settings.embedding_queue_dir is not None
        else (Path(settings.index_dir).expanduser().resolve() / "embedding-queue")
    )
    queue = SlotQueue(
        queue_dir=queue_dir,
        base_url=settings.embedding_base_url,
        max_concurrency=settings.embedding_max_concurrency,
        max_wait_s=settings.embedding_queue_max_wait_s,
        poll_s=settings.lock_poll_s,
        clock=clock,
        sleep=sleep,
        asleep=asleep,
    )
    policy_kwargs: dict[str, object] = {
        "max_retries": settings.embedding_max_retries,
        "retry_statuses": tuple(settings.embedding_retry_statuses),
        "base_backoff_s": settings.embedding_retry_base_backoff_s,
        "max_wait_s": settings.embedding_retry_max_wait_s,
        "jitter": settings.embedding_retry_jitter,
    }
    if now is not None:
        policy_kwargs["now"] = now
    if rng is not None:
        policy_kwargs["rng"] = rng
    policy = RetryPolicy(**policy_kwargs)
    return EmbeddingAdmission(queue=queue, policy=policy, clock=clock, sleep=sleep, asleep=asleep)
