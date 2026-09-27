"""The runner service: the one process that executes candidate programs, and nothing else.

One working route, ``POST /v1/execute``, which takes an
:class:`~verifier_guided_search.sandbox.protocol.ExecJob` and returns a
:class:`~verifier_guided_search.sandbox.protocol.Verdict`. Everything the route does is delegated to
:class:`~verifier_guided_search.sandbox.local.LocalSandbox`, so the code path that runs untrusted
programs is identical in the container, in the tests and in the offline demo. The only difference is
where it runs, and that difference is the point:

* the service is reached only over a compose network with ``internal: true``, so a program that gets
  past the in-process guard still has nowhere to send anything;
* the container has a read-only root filesystem with a tmpfs at the sandbox workroot, so the
writable
  jail is the only writable path in the image;
* it runs as a non-root UID with ``cap_drop: ALL`` and ``no-new-privileges``, under pids, memory and
  CPU limits, so a fork bomb or a memory hog is the container runtime's problem, not Python's;
* it holds no secrets: no ``ANTHROPIC_API_KEY``, no database URL, nothing. The only credential it
  knows is the digest of the key the search service presents.

Hardening that belongs to *this* process rather than to the container:

``no readiness dependency on the search service``
    ``/readyz`` checks that the harness is in place and a child interpreter starts. It never calls
    back out.
``request size``
    ``max_body_bytes`` caps the payload; a program plus its tests is a few kilobytes, so the default
    is already generous. The job model caps the program at 64,000 characters and each field
    separately, before any of it is written to disk.
``limits are clamped, not trusted``
    A caller can only ask for limits *at or below* the service's configured ceiling, so a
    compromised search service cannot ask for a 60-second timeout or a gigabyte of memory.
``verdicts are not free text``
    What comes back is the validated verdict model. A candidate's stdout and exception messages are
    counted and redacted into ``detail``, never forwarded verbatim into a prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from verifier_guided_search.core.app import create_app
from verifier_guided_search.core.config import Settings, get_settings
from verifier_guided_search.core.logging import get_logger
from verifier_guided_search.core.security import AuthenticatedPrincipal
from verifier_guided_search.metrics import SearchMetrics, create_search_metrics
from verifier_guided_search.sandbox.local import LocalSandbox
from verifier_guided_search.sandbox.protocol import ExecJob, ExecLimits, Verdict

__all__ = ["clamp_limits", "create_runner_app"]

_log: Final = get_logger("verifier_guided_search.runner")


def clamp_limits(requested: ExecLimits, settings: Settings) -> ExecLimits:
    """Reduce the caller's limits to the service's ceiling. Never raises them.

    A caller may ask for less than the ceiling - a cheap task does not need 1.5 seconds - but the
    ceiling itself is the operator's decision, configured on the runner, not negotiated per request.
    """
    return ExecLimits(
        timeout_seconds=min(requested.timeout_seconds, settings.sandbox_timeout_seconds),
        memory_bytes=min(requested.memory_bytes, settings.sandbox_memory_bytes),
        max_processes=min(requested.max_processes, ExecLimits().max_processes),
        max_file_bytes=min(requested.max_file_bytes, ExecLimits().max_file_bytes),
        max_output_bytes=min(requested.max_output_bytes, ExecLimits().max_output_bytes),
    )


def create_runner_app(
    settings: Settings | None = None,
    *,
    workroot: Path | None = None,
    sandbox: LocalSandbox | None = None,
    registry: CollectorRegistry | None = None,
    metrics: SearchMetrics | None = None,
) -> FastAPI:
    """Build the runner application. ``workroot`` must be on a writable tmpfs in the container."""
    cfg = settings if settings is not None else get_settings()
    root = workroot if workroot is not None else cfg.runner_workroot
    box = sandbox if sandbox is not None else LocalSandbox(root)
    search_metrics = metrics if metrics is not None else create_search_metrics(registry)

    def routes(app: FastAPI) -> None:
        @app.post("/v1/execute", response_model=Verdict)
        async def execute(job: ExecJob, principal: AuthenticatedPrincipal) -> Verdict:
            """Run one candidate program against one task's tests in a fresh child interpreter."""
            bounded = job.model_copy(update={"limits": clamp_limits(job.limits, cfg)})
            verdict = box.execute(bounded)
            search_metrics.verifier_verdicts_total.labels(verdict.status).inc()
            search_metrics.sandbox_exec_seconds.observe(verdict.duration_seconds)
            # The program hash, not the program: a log line is not a place for untrusted source.
            _log.info(
                "candidate_executed",
                key_id=principal.key_id,
                program_sha256=verdict.program_sha256,
                status=verdict.status,
                violations=len(verdict.violations),
                duration_seconds=round(verdict.duration_seconds, 4),
            )
            return verdict

    app = create_app(
        cfg,
        readiness_checks={"sandbox": box.ping},
        registry=search_metrics.registry,
        routes=routes,
    )
    app.state.sandbox = box
    app.state.search_metrics = search_metrics
    return app
