"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import structlog
from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


# --------------------------------------------------------------------------- project fixtures
#
# Sandbox executions cost roughly 0.13s each (a fresh interpreter per program), so anything that
# needs real verdicts shares one session-scoped sandbox with a warm verdict cache. Tests that assert
# on cache or sandbox internals take their own instance instead.


@pytest.fixture(scope="session")
def tiny_suite() -> TaskSuite:
    """Two families, one instance each: enough to exercise a search, cheap enough to do it often."""
    from verifier_guided_search.tasks.families import FAMILIES
    from verifier_guided_search.tasks.suite import build_suite

    return build_suite(instances_per_family=1, families=FAMILIES[:2])


@pytest.fixture(scope="session")
def small_suite() -> TaskSuite:
    """One instance of every family: the smallest suite that covers all 15 families."""
    from verifier_guided_search.tasks.suite import build_suite

    return build_suite(instances_per_family=1)


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalSandbox:
    """A fresh sandbox with its own workroot. Use when a test asserts on sandbox behaviour."""
    from verifier_guided_search.sandbox.local import LocalSandbox

    return LocalSandbox(tmp_path / "sandbox")


@pytest.fixture(scope="session")
def shared_sandbox(tmp_path_factory: pytest.TempPathFactory) -> CachingSandbox:
    """One caching sandbox for the whole session, so repeated programs cost a dict lookup."""
    from verifier_guided_search.sandbox.client import CachingSandbox
    from verifier_guided_search.sandbox.local import LocalSandbox

    root = tmp_path_factory.mktemp("shared-sandbox")
    return CachingSandbox(LocalSandbox(root))


if TYPE_CHECKING:
    from verifier_guided_search.sandbox.client import CachingSandbox
    from verifier_guided_search.sandbox.local import LocalSandbox
    from verifier_guided_search.tasks.suite import TaskSuite
