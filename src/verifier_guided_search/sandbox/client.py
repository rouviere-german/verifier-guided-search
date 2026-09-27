"""Sandbox clients: the protocol, the caching wrapper, and the HTTP client for the runner.

The search never holds a :class:`LocalSandbox` directly in production. It holds a
:class:`SandboxClient`, which is either

* :class:`~verifier_guided_search.sandbox.local.LocalSandbox` - one fresh interpreter per program on
  this machine, used by the tests, the demo and the benches; or
* :class:`HttpSandbox` - a POST to the runner container over a Compose network declared
  ``internal: true``, which is how untrusted code ends up somewhere with no egress, a read-only
  root filesystem and no Docker socket.

:class:`CachingSandbox` sits in front of either one and is also where sandbox-second accounting and
the verdict metrics are recorded, so every execution is charged exactly once whichever client is
underneath.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx
from pydantic import ValidationError

from verifier_guided_search.sandbox.cache import VerdictCache
from verifier_guided_search.sandbox.protocol import ExecJob, Verdict

if TYPE_CHECKING:
    from verifier_guided_search.metrics import SearchMetrics

__all__ = [
    "CachingSandbox",
    "ExecutionRecord",
    "HttpSandbox",
    "SandboxClient",
    "SandboxUnavailable",
]

_DEFAULT_TIMEOUT = 30.0


class SandboxUnavailable(RuntimeError):
    """The runner could not be reached or answered with something unusable."""


@runtime_checkable
class SandboxClient(Protocol):
    """Anything that turns an :class:`ExecJob` into a :class:`Verdict`."""

    def execute(self, job: ExecJob) -> Verdict: ...

    def ping(self) -> bool: ...


class ExecutionRecord:
    """Mutable tally of what the sandbox did, for budgets, metrics and results files."""

    __slots__ = ("cache_hits", "executions", "seconds")

    def __init__(self) -> None:
        self.executions = 0
        self.cache_hits = 0
        self.seconds = 0.0

    @property
    def lookups(self) -> int:
        return self.executions + self.cache_hits

    @property
    def cache_hit_ratio(self) -> float:
        return self.cache_hits / self.lookups if self.lookups else 0.0


class CachingSandbox:
    """Answers from the cache when it can, and records what it had to run when it cannot."""

    def __init__(
        self,
        inner: SandboxClient,
        *,
        cache: VerdictCache | None = None,
        metrics: SearchMetrics | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache if cache is not None else VerdictCache()
        self._metrics = metrics
        self.record = ExecutionRecord()

    @property
    def cache(self) -> VerdictCache:
        return self._cache

    @property
    def inner(self) -> SandboxClient:
        """The client underneath: the runner over HTTP, or a local sandbox."""
        return self._inner

    def ping(self) -> bool:
        return self._inner.ping()

    def execute(self, job: ExecJob) -> Verdict:
        cached = self._cache.get(job)
        if cached is not None:
            self.record.cache_hits += 1
            self._publish(cached, cache_hit=True)
            return cached
        verdict = self._inner.execute(job)
        self._cache.put(job, verdict)
        self.record.executions += 1
        self.record.seconds += verdict.duration_seconds
        self._publish(verdict, cache_hit=False)
        return verdict

    def _publish(self, verdict: Verdict, *, cache_hit: bool) -> None:
        if self._metrics is None:
            return
        self._metrics.verifier_verdicts_total.labels(verdict.status).inc()
        if cache_hit:
            self._metrics.verdict_cache_hits_total.inc()
        else:
            self._metrics.verdict_cache_misses_total.inc()
            self._metrics.sandbox_exec_seconds.observe(verdict.duration_seconds)
        self._metrics.verdict_cache_hit_ratio.set(self.record.cache_hit_ratio)


class HttpSandbox:
    """Talks to the runner container over the internal Compose network."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._base_url = base_url.rstrip("/")
        self._headers = headers

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpSandbox:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ping(self) -> bool:
        try:
            response = self._client.get(f"{self._base_url}/healthz", headers=self._headers)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def execute(self, job: ExecJob) -> Verdict:
        try:
            response = self._client.post(
                f"{self._base_url}/v1/execute",
                json=job.model_dump(mode="json"),
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            raise SandboxUnavailable(f"runner unreachable: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise SandboxUnavailable(f"runner returned HTTP {response.status_code}")
        try:
            verdict = Verdict.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise SandboxUnavailable("runner returned an unusable verdict") from exc
        if verdict.program_sha256 != job.program_sha256:
            raise SandboxUnavailable("runner answered about a different program")
        return verdict
