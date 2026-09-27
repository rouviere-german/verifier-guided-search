"""The search service: accepts a search request, never executes a candidate itself.

One working route, ``POST /v1/search``: pick a benchmark task, a strategy and a candidate budget,
and get back what the search committed to. Candidate execution goes to the runner over the internal
network, so this process never runs untrusted code even when it is the one holding the credentials.

Readiness, as the spec asks for it
----------------------------------
``/healthz`` is liveness only: the process answers. ``/readyz`` runs two dependency checks and
reports their names without their errors:

``runner``
    The verifier is reachable. Without it a search would return "solved nothing" rather than fail,
    which is the worst possible failure mode for this service - a silent zero.
``value_model``
    The configured value model loads *and re-derives its own content hash*. A model file whose bytes
    no longer match the version in its name is refused
    (:meth:`~verifier_guided_search.value.registry.ValueModelRegistry.verify`), so the service will
    not serve searches guided by an artifact somebody edited. When no registry is configured the
    check reports ready and the service runs on visible pass fractions, which is the documented
    value-model-off mode rather than a degraded one.

What the route will not do
--------------------------
It will not take a program or a test suite from the caller. The task comes from the benchmark by id,
so a request cannot smuggle arbitrary code in through the search service and have the runner execute
it on the caller's terms - the tests that a candidate is measured against are always the
benchmark's. Budgets are clamped to ``search_max_candidates``, so one request cannot spend an
unbounded amount of verifier time.
"""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field

from verifier_guided_search.core.app import ReadinessCheck, create_app
from verifier_guided_search.core.config import Settings, get_settings
from verifier_guided_search.core.errors import AppError
from verifier_guided_search.core.logging import get_logger
from verifier_guided_search.core.security import AuthenticatedPrincipal
from verifier_guided_search.metrics import SearchMetrics, create_search_metrics
from verifier_guided_search.sandbox.client import CachingSandbox, HttpSandbox, SandboxClient
from verifier_guided_search.sandbox.local import LocalSandbox
from verifier_guided_search.sandbox.protocol import ExecLimits
from verifier_guided_search.search.budget import BudgetAccountant, BudgetSpec
from verifier_guided_search.search.generator import MutationGenerator, token_budget_for
from verifier_guided_search.search.strategies import (
    STRATEGIES,
    SearchConfig,
    SearchContext,
    run_search,
)
from verifier_guided_search.tasks.suite import TaskSuite, default_suite
from verifier_guided_search.value.model import GradientBoostedValueModel
from verifier_guided_search.value.registry import ValueModelRegistry

__all__ = [
    "SearchRequest",
    "SearchResponse",
    "build_sandbox",
    "create_search_app",
    "exec_limits",
]

_log: Final = get_logger("verifier_guided_search.service")
_SANDBOX_SLACK: Final = 4.0
_USD_CAP: Final = 1.0


class SearchRequest(BaseModel):
    """A search request. No program and no tests: the task is named, not supplied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+/[0-9]{2}$")
    strategy: str | None = Field(default=None, pattern=r"^[a-z_]{1,32}$")
    max_candidates: int | None = Field(default=None, ge=1, le=256)
    children_per_expansion: int = Field(default=4, ge=1, le=16)
    c_puct: float = Field(default=1.4, ge=0, le=10)
    use_value_model: bool = True


class SearchResponse(BaseModel):
    """What the search committed to, and what it cost. Hidden outcomes are reported here.

    The search never saw them: they come from the verdict of the candidate it already chose, which
    is the whole point of the visible/hidden split. A caller that wants to grade the search gets the
    grade; a caller that wants to cheat cannot, because the choice was already made.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    family: str
    strategy: str
    stop_reason: str
    evaluations: int = Field(ge=0)
    nodes: int = Field(ge=0)
    expansions: int = Field(ge=0)
    solved_visible: bool
    solved_hidden: bool
    false_pass: bool
    visible_passed: int = Field(ge=0)
    visible_total: int = Field(ge=0)
    hidden_passed: int = Field(ge=0)
    hidden_total: int = Field(ge=0)
    program_sha256: str | None = None
    usage: dict[str, int]
    value_model_version: str | None = None


def build_sandbox(settings: Settings, *, metrics: SearchMetrics | None = None) -> SandboxClient:
    """The verifier client: the runner over the internal network, or a local one when unset.

    A deployment always sets ``runner_url``. The local fallback is what makes the tests, the demo
    and the benches work without Docker, and it runs the very same child harness.
    """
    inner: SandboxClient
    if settings.runner_url:
        key = settings.runner_api_key.get_secret_value() if settings.runner_api_key else None
        inner = HttpSandbox(
            settings.runner_url, api_key=key, timeout=settings.runner_timeout_seconds
        )
    else:
        inner = LocalSandbox(settings.data_dir / "sandbox")
    return CachingSandbox(inner, metrics=metrics)


def _value_model(settings: Settings) -> GradientBoostedValueModel | None:
    registry = ValueModelRegistry(settings.value_model_root)
    wanted = settings.value_model_version or registry.current_version
    if wanted is None:
        return None
    return registry.load(wanted)


def _value_model_check(settings: Settings) -> ReadinessCheck:
    def check() -> bool:
        registry = ValueModelRegistry(settings.value_model_root)
        wanted = settings.value_model_version or registry.current_version
        if wanted is None:
            # No model configured: the service runs on visible pass fractions, and says so.
            return settings.value_model_version is None
        return registry.verify(wanted)

    return check


def create_search_app(
    settings: Settings | None = None,
    *,
    suite: TaskSuite | None = None,
    sandbox: SandboxClient | None = None,
    value_model: GradientBoostedValueModel | None = None,
    registry: CollectorRegistry | None = None,
    metrics: SearchMetrics | None = None,
) -> FastAPI:
    """Build the search application."""
    cfg = settings if settings is not None else get_settings()
    search_metrics = metrics if metrics is not None else create_search_metrics(registry)
    tasks = suite if suite is not None else default_suite()
    verifier = sandbox if sandbox is not None else build_sandbox(cfg, metrics=search_metrics)
    model = value_model if value_model is not None else _load_quietly(cfg)
    generator = MutationGenerator()

    def routes(app: FastAPI) -> None:
        @app.post("/v1/search", response_model=SearchResponse)
        async def search(body: SearchRequest, principal: AuthenticatedPrincipal) -> SearchResponse:
            """Run one budgeted search over one benchmark task."""
            try:
                task = tasks.get(body.task_id)
            except KeyError as missing:
                raise AppError(404, "unknown_task", "unknown benchmark task") from missing
            strategy = body.strategy or cfg.search_default_strategy
            if strategy not in STRATEGIES:
                raise AppError(422, "unknown_strategy", "unknown search strategy")
            cap = min(body.max_candidates or cfg.search_max_candidates, cfg.search_max_candidates)
            config = SearchConfig(
                strategy=strategy,
                children_per_expansion=body.children_per_expansion,
                c_puct=body.c_puct,
                use_value_model=body.use_value_model,
                limits=exec_limits(cfg),
            )
            accountant = BudgetAccountant(
                BudgetSpec.of(
                    candidates=cap,
                    tokens=token_budget_for(
                        generator, expansions=cap, k=body.children_per_expansion
                    ),
                    sandbox_seconds=cap * cfg.sandbox_timeout_seconds * _SANDBOX_SLACK,
                    usd=_USD_CAP,
                )
            )
            outcome = run_search(
                task,
                SearchContext(
                    sandbox=verifier,
                    generator=generator,
                    accountant=accountant,
                    config=config,
                    value_model=model if body.use_value_model else None,
                    metrics=search_metrics,
                    actor=f"api:{principal.key_id}",
                ),
            )
            answer = outcome.answer
            _log.info(
                "search_completed",
                key_id=principal.key_id,
                task_id=task.task_id,
                strategy=strategy,
                evaluations=len(outcome.evaluations),
                stop_reason=outcome.stop_reason,
                solved_hidden=bool(answer is not None and answer.solved_hidden),
            )
            return SearchResponse(
                task_id=task.task_id,
                family=task.family,
                strategy=strategy,
                stop_reason=outcome.stop_reason,
                evaluations=len(outcome.evaluations),
                nodes=outcome.nodes,
                expansions=outcome.expansions,
                solved_visible=bool(answer is not None and answer.solved_visible),
                solved_hidden=bool(answer is not None and answer.solved_hidden),
                false_pass=bool(answer is not None and answer.false_pass),
                visible_passed=answer.feedback.visible.passed if answer else 0,
                visible_total=answer.feedback.visible.total if answer else 0,
                hidden_passed=answer.verdict.hidden.passed if answer else 0,
                hidden_total=answer.verdict.hidden.total if answer else 0,
                program_sha256=answer.program_sha256 if answer else None,
                usage=accountant.committed.as_dict(),
                value_model_version=model.version if model is not None else None,
            )

    app = create_app(
        cfg,
        readiness_checks={
            "runner": verifier.ping,
            "value_model": _value_model_check(cfg),
        },
        registry=search_metrics.registry,
        routes=routes,
    )
    app.state.sandbox = verifier
    app.state.value_model = model
    app.state.suite = tasks
    app.state.search_metrics = search_metrics
    return app


def exec_limits(settings: Settings) -> ExecLimits:
    """The per-candidate limits this service asks for. The runner clamps them further."""
    return ExecLimits(
        timeout_seconds=settings.sandbox_timeout_seconds,
        memory_bytes=settings.sandbox_memory_bytes,
    )


def _load_quietly(settings: Settings) -> GradientBoostedValueModel | None:
    """Load the configured value model, or None when there is not a usable one.

    A missing or corrupt model is not a startup failure: ``/readyz`` reports it, the service runs on
    visible pass fractions, and an operator fixes the registry without a crash loop.
    """
    from verifier_guided_search.errors import ValueModelError  # noqa: PLC0415

    try:
        return _value_model(settings)
    except ValueModelError as exc:
        _log.warning("value_model_unavailable", error_type=type(exc).__name__)
        return None
