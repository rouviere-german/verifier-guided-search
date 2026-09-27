"""The bounded gate the two blocking routes hold.

Its job is narrow: let ``limit`` callers through, refuse the rest immediately with a 503 rather than
parking them in a threadpool that has already run out of threads, and always give the slot back.
"""

from __future__ import annotations

import threading

import pytest

from verifier_guided_search.core.concurrency import ConcurrencyLimiter
from verifier_guided_search.core.errors import AppError

pytestmark = pytest.mark.unit


def test_a_limiter_admits_up_to_its_limit_and_refuses_the_next() -> None:
    limiter = ConcurrencyLimiter(2, name="execute")
    with limiter.slot(), limiter.slot(), pytest.raises(AppError) as refused:
        limiter.slot().__enter__()
    assert refused.value.status_code == 503
    assert refused.value.code == "busy"
    assert "execute" in refused.value.message
    assert refused.value.headers["Retry-After"] == "1"
    assert limiter.refusals == 1


def test_a_slot_is_returned_even_when_the_body_raises() -> None:
    limiter = ConcurrencyLimiter(1, name="search")
    with pytest.raises(RuntimeError, match="boom"), limiter.slot():
        raise RuntimeError("boom")
    # The slot came back, so the next caller is admitted rather than refused.
    with limiter.slot():
        pass
    assert limiter.refusals == 0


def test_waiting_briefly_admits_a_caller_whose_slot_frees_up() -> None:
    """``wait_seconds`` is the small queue for a slot that is about to be released."""
    limiter = ConcurrencyLimiter(1, name="search", wait_seconds=5.0)
    holding = threading.Event()
    admitted = threading.Event()

    def hold() -> None:
        with limiter.slot():
            holding.set()
            admitted.wait(timeout=5)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert holding.wait(timeout=5)
        admitted.set()
        with limiter.slot():
            pass
    finally:
        admitted.set()
        worker.join(timeout=10)
    assert limiter.refusals == 0


def test_a_nonsensical_limiter_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        ConcurrencyLimiter(0, name="execute")
    with pytest.raises(ValueError, match="wait_seconds must not be negative"):
        ConcurrencyLimiter(1, name="execute", wait_seconds=-1.0)
