"""A bounded gate for route handlers that do blocking work.

Both routes in this repo block: ``POST /v1/execute`` waits on ``subprocess.run`` for up to the
sandbox timeout plus a grace, and ``POST /v1/search`` runs a whole budgeted search - up to sixteen
of those executions, plus synchronous HTTP to the runner. Declaring such a handler ``async def``
puts that work on the event loop itself, so one in-flight request stalls ``/healthz``, ``/readyz``,
``/metrics`` and the rate-limit middleware for its whole duration and the health probes flap under
ordinary load. The handlers are therefore plain ``def``, which makes Starlette dispatch them to its
worker threadpool.

That moves the problem rather than solving it: the threadpool is finite (40 threads by default),
and a caller who opens more concurrent searches than that starves every other route the same way. So
each route holds a :class:`ConcurrencyLimiter` sized for what the process can do at once, and a
request that cannot get a slot is refused with 503 straight away instead of queueing behind a
threadpool that has run out. Refusing quickly is the useful answer: a caller can retry, and the
probes keep answering.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from verifier_guided_search.core.errors import AppError

__all__ = ["ConcurrencyLimiter"]

_RETRY_AFTER: Final = "1"


class ConcurrencyLimiter:
    """At most ``limit`` callers inside :meth:`slot` at once; the rest get a 503.

    ``wait_seconds`` is a small amount of queueing for the case where a slot is about to free up.
    It defaults to zero, because a caller waiting in the threadpool is itself a consumed thread.
    """

    def __init__(self, limit: int, *, name: str, wait_seconds: float = 0.0) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if wait_seconds < 0:
            raise ValueError("wait_seconds must not be negative")
        self._semaphore = threading.BoundedSemaphore(limit)
        self._limit = limit
        self._name = name
        self._wait = wait_seconds
        self._refusals = 0
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def refusals(self) -> int:
        """How many requests this limiter has refused. Read by tests and by operators."""
        with self._lock:
            return self._refusals

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold one slot, or raise :class:`AppError` 503 without doing any work."""
        acquired = (
            self._semaphore.acquire(timeout=self._wait)
            if self._wait
            else (self._semaphore.acquire(blocking=False))
        )
        if not acquired:
            with self._lock:
                self._refusals += 1
            raise AppError(
                503,
                "busy",
                f"too many concurrent {self._name} requests; retry shortly",
                headers={"Retry-After": _RETRY_AFTER},
            )
        try:
            yield
        finally:
            self._semaphore.release()
