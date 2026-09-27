"""The budget sweep: one run per (task, strategy), read out at five budget levels.

Why one run and not five
------------------------
Every strategy is a generator that yields evaluations in a fixed order and never inspects the
remaining budget (see :mod:`verifier_guided_search.search.strategies`). A run whose candidate cap is
``B`` therefore produces exactly the first ``B`` evaluations of a run with a larger cap, truncated
at the first candidate that satisfies every visible test. So the sweep runs each pair once at the
largest level and reads the smaller levels off as prefixes. That is five times less compute *and* a
stronger comparison: at every level the strategies are being judged on literally the same work they
did at the level above, with no resampling noise between levels.

The claim is not assumed. ``tests/integration/test_budget_prefix.py`` runs the search again with a
real, smaller budget and asserts the evaluations are identical to the prefix.

What a budget is
----------------
The accountant enforces four dimensions at once (candidates, generator tokens, sandbox microseconds,
USD). The sweep varies the *candidate* cap - the number of programs handed to the verifier - and
holds the other three at a level generous enough that they never bind, so a level is a single number
and "compute to reach 60%" is well defined. The tokens and sandbox seconds actually spent are
recorded at every level anyway, so a reader can see what the candidate axis cost in each currency.

What is scored
--------------
The answer at a level is the candidate the strategy would have committed to: one that passed every
visible test if it found one, otherwise the best visible pass fraction. That answer is then scored
on the *hidden* tests, which no strategy ever saw. Three numbers come out of that per level:

``hidden_solve_rate``
    Share of tasks whose committed answer passes every hidden test. The headline metric.
``visible_solve_rate``
    Share where the verifier was satisfied. Always at least the hidden rate.
``false_pass_rate``
    Share where the verifier was satisfied and the hidden tests were not: the cost of trusting a
    partial test suite, and the reason the benchmark keeps two groups. The *task ids* are reported
    alongside it (``false_pass_task_ids``), because the rate alone invites the reading that more
    search keeps producing new wrong answers, while the ids show a fixed, small set of tasks whose
    visible suite is satisfiable by a wrong program - one of them at depth 0, before any search.

Confidence intervals are percentile bootstrap over *tasks* (1000 resamples by default), because two
evaluations of the same task are not independent observations.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from verifier_guided_search.agentic.audit import AuditLog
from verifier_guided_search.errors import SearchError
from verifier_guided_search.metrics import SearchMetrics
from verifier_guided_search.sandbox.client import SandboxClient
from verifier_guided_search.sandbox.protocol import ExecLimits
from verifier_guided_search.search.budget import BudgetAccountant, BudgetSpec
from verifier_guided_search.search.generator import (
    CandidateGenerator,
    MutationGenerator,
    token_budget_for,
)
from verifier_guided_search.search.strategies import (
    STRATEGIES,
    Evaluation,
    SearchConfig,
    SearchContext,
    SearchOutcome,
    best_evaluation,
    run_search,
)
from verifier_guided_search.search.trace import TraceStore
from verifier_guided_search.tasks.suite import TaskSuite
from verifier_guided_search.value.metrics import Interval, bootstrap_ci
from verifier_guided_search.value.model import GradientBoostedValueModel

__all__ = [
    "DEFAULT_BUDGET_LEVELS",
    "DEFAULT_STRATEGIES",
    "LevelStats",
    "SweepConfig",
    "SweepDeps",
    "SweepResults",
    "TaskStrategyRun",
    "aggregate",
    "compute_to_target",
    "default_generator",
    "run_matrix",
    "run_sweep",
    "summarise",
]

#: Five levels, doubling. Level 1 is "one sample, no search", which is the baseline the repo argues
#: against; level 16 is still seconds per task on a CPU.
DEFAULT_BUDGET_LEVELS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
DEFAULT_STRATEGIES: Final[tuple[str, ...]] = ("greedy", "best_of_n", "beam", "mcts")
#: Sandbox seconds and USD are set from the candidate cap so they never bind before it does.
_SANDBOX_SECONDS_SLACK: Final = 4.0
_USD_SLACK: Final = 10.0
DEFAULT_TARGET_SOLVE_RATE: Final = 0.60


class SweepConfig(BaseModel):
    """Everything needed to reproduce a sweep. Written verbatim into the results file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    instances_per_family: int = Field(ge=1)
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    budget_levels: tuple[int, ...] = DEFAULT_BUDGET_LEVELS
    children_per_expansion: int = Field(default=4, ge=1, le=32)
    beam_width: int = Field(default=3, ge=1, le=32)
    c_puct: float = Field(default=1.4, ge=0)
    use_value_model: bool = True
    generator_noise: float = Field(default=0.55, ge=0)
    generator_seed: int = 7
    timeout_seconds: float = Field(default=1.5, gt=0)
    bootstrap_resamples: int = Field(default=1000, ge=0)
    target_solve_rate: float = Field(default=DEFAULT_TARGET_SOLVE_RATE, gt=0, le=1)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        unknown = [name for name in self.strategies if name not in STRATEGIES]
        if unknown:
            raise SearchError(f"unknown strategies: {', '.join(unknown)}")
        if not self.budget_levels or sorted(self.budget_levels) != list(self.budget_levels):
            raise SearchError("budget_levels must be non-empty and ascending")
        if len(set(self.budget_levels)) != len(self.budget_levels):
            raise SearchError("budget_levels must be distinct")

    @property
    def max_candidates(self) -> int:
        return self.budget_levels[-1]

    @property
    def limits(self) -> ExecLimits:
        return ExecLimits(timeout_seconds=self.timeout_seconds)

    def search_config(self, strategy: str) -> SearchConfig:
        return SearchConfig(
            strategy=strategy,
            children_per_expansion=self.children_per_expansion,
            beam_width=self.beam_width,
            c_puct=self.c_puct,
            use_value_model=self.use_value_model,
            limits=self.limits,
        )

    def budget(self, generator: CandidateGenerator, *, candidates: int | None = None) -> BudgetSpec:
        """The budget for one run. Only ``max_candidates`` changes between levels."""
        cap = self.max_candidates if candidates is None else candidates
        expansions = max(self.max_candidates, 1)
        return BudgetSpec.of(
            candidates=cap,
            tokens=token_budget_for(
                generator, expansions=expansions, k=self.children_per_expansion
            ),
            sandbox_seconds=self.max_candidates * self.timeout_seconds * _SANDBOX_SECONDS_SLACK,
            usd=_USD_SLACK,
        )


@dataclass(frozen=True, slots=True)
class TaskStrategyRun:
    """One (task, strategy) run at the largest budget, plus how long it took on the wall clock."""

    task_id: str
    family: str
    strategy: str
    outcome: SearchOutcome
    wall_seconds: float

    def answer_at(self, candidates: int) -> Evaluation | None:
        """The candidate this run would have committed to with a cap of ``candidates``."""
        return best_evaluation(self.outcome.prefix(candidates))

    def usage_at(self, candidates: int) -> tuple[int, float, int]:
        """(candidates evaluated, sandbox seconds, generator tokens) up to ``candidates``.

        Sandbox seconds are summed from the verdicts in the prefix. Generator tokens are not
        attributable to a single evaluation, so they are pro-rated by the share of evaluations in
        the prefix - an approximation, and the reason the headline compute axis is candidates.
        """
        prefix = self.outcome.prefix(candidates)
        seconds = sum(item.verdict.duration_seconds for item in prefix)
        total = len(self.outcome.evaluations)
        share = len(prefix) / total if total else 0.0
        tokens = round(self.outcome.usage.generator_tokens * share)
        return len(prefix), seconds, tokens


class LevelStats(BaseModel):
    """Aggregated outcome of one strategy at one budget level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str
    candidates_budget: int = Field(ge=1)
    tasks: int = Field(ge=1)
    hidden_solved: int = Field(ge=0)
    visible_solved: int = Field(ge=0)
    false_passes: int = Field(ge=0)
    hidden_solve_rate: float = Field(ge=0, le=1)
    hidden_ci_low: float = Field(ge=0, le=1)
    hidden_ci_high: float = Field(ge=0, le=1)
    visible_solve_rate: float = Field(ge=0, le=1)
    false_pass_rate: float = Field(ge=0, le=1)
    mean_candidates_used: float = Field(ge=0)
    mean_sandbox_seconds: float = Field(ge=0)
    mean_generator_tokens: float = Field(ge=0)
    #: Which tasks the verifier was fooled on. Reported because the aggregate rate on its own reads
    #: as "search keeps finding new wrong answers", and the ids show it is the same small set of
    #: tasks whose visible suite a wrong program can satisfy.
    false_pass_task_ids: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.strategy}@{self.candidates_budget}"


class SweepResults(BaseModel):
    """The results file: what was run, on what, and what came out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    config: SweepConfig
    suite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    family_count: int = Field(ge=1)
    generator: str
    generator_model: str
    value_model_version: str | None = None
    levels: tuple[LevelStats, ...] = Field(min_length=1)
    compute_to_target: dict[str, int | None] = Field(default_factory=dict)
    total_evaluations: int = Field(ge=0)
    total_sandbox_seconds: float = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    cache_hit_ratio: float = Field(ge=0, le=1)

    def for_strategy(self, strategy: str) -> tuple[LevelStats, ...]:
        return tuple(item for item in self.levels if item.strategy == strategy)

    def level(self, strategy: str, candidates: int) -> LevelStats:
        for item in self.levels:
            if item.strategy == strategy and item.candidates_budget == candidates:
                return item
        raise KeyError(f"no level {strategy}@{candidates}")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(units: Sequence[tuple[str, bool]]) -> float:
    return sum(1 for _, flag in units if flag) / len(units) if units else 0.0


def aggregate(
    runs: Sequence[TaskStrategyRun],
    *,
    strategy: str,
    candidates: int,
    resamples: int = 1000,
    seed: int = 0,
) -> LevelStats:
    """Aggregate one strategy at one level over the tasks, with a bootstrap CI over tasks."""
    selected = [run for run in runs if run.strategy == strategy]
    if not selected:
        raise SearchError(f"no runs for strategy {strategy!r}")
    hidden_units: list[tuple[str, bool]] = []
    visible_units: list[tuple[str, bool]] = []
    false_units: list[tuple[str, bool]] = []
    used: list[float] = []
    seconds: list[float] = []
    tokens: list[float] = []
    for run in selected:
        answer = run.answer_at(candidates)
        hidden_units.append((run.task_id, answer is not None and answer.solved_hidden))
        visible_units.append((run.task_id, answer is not None and answer.solved_visible))
        false_units.append((run.task_id, answer is not None and answer.false_pass))
        count, sandbox_seconds, generator_tokens = run.usage_at(candidates)
        used.append(float(count))
        seconds.append(sandbox_seconds)
        tokens.append(float(generator_tokens))
    interval: Interval = bootstrap_ci(hidden_units, _rate, resamples=resamples, seed=seed)
    return LevelStats(
        strategy=strategy,
        candidates_budget=candidates,
        tasks=len(selected),
        hidden_solved=sum(1 for _, flag in hidden_units if flag),
        visible_solved=sum(1 for _, flag in visible_units if flag),
        false_passes=sum(1 for _, flag in false_units if flag),
        hidden_solve_rate=interval.estimate,
        hidden_ci_low=interval.low,
        hidden_ci_high=interval.high,
        visible_solve_rate=_rate(visible_units),
        false_pass_rate=_rate(false_units),
        mean_candidates_used=_mean(used),
        mean_sandbox_seconds=_mean(seconds),
        mean_generator_tokens=_mean(tokens),
        false_pass_task_ids=tuple(task_id for task_id, flag in false_units if flag),
    )


def compute_to_target(levels: Sequence[LevelStats], target: float) -> int | None:
    """The smallest candidate budget whose hidden solve rate reaches ``target``, or None.

    A point estimate, not a confidence statement: with 5 levels and a doubling ladder the honest
    reading is "between this level and the one below", which the report says explicitly.
    """
    for item in sorted(levels, key=lambda stats: stats.candidates_budget):
        if item.hidden_solve_rate >= target:
            return item.candidates_budget
    return None


@dataclass
class SweepDeps:
    """The collaborators a sweep needs. Supplied by the caller so nothing here builds a sandbox."""

    sandbox: SandboxClient
    generator: CandidateGenerator
    value_model: GradientBoostedValueModel | None = None
    traces: TraceStore | None = None
    metrics: SearchMetrics | None = None
    audit: AuditLog | None = None
    progress: Callable[[str], None] | None = None
    extra: Mapping[str, str] = field(default_factory=dict)


def run_matrix(suite: TaskSuite, config: SweepConfig, deps: SweepDeps) -> list[TaskStrategyRun]:
    """Run every (task, strategy) pair once, at the largest configured budget.

    Returned separately from :func:`summarise` because the aggregate throws away which program won
    which task, and the demo needs exactly that to propose a patch.
    """
    runs: list[TaskStrategyRun] = []
    for strategy in config.strategies:
        for task in suite:
            accountant = BudgetAccountant(config.budget(deps.generator))
            context = SearchContext(
                sandbox=deps.sandbox,
                generator=deps.generator,
                accountant=accountant,
                config=config.search_config(strategy),
                value_model=deps.value_model,
                metrics=deps.metrics,
                audit=deps.audit,
                traces=deps.traces,
                actor=f"sweep:{strategy}",
            )
            run_started = time.perf_counter()
            outcome = run_search(task, context)
            runs.append(
                TaskStrategyRun(
                    task_id=task.task_id,
                    family=task.family,
                    strategy=strategy,
                    outcome=outcome,
                    wall_seconds=time.perf_counter() - run_started,
                )
            )
            if not accountant.invariant_holds():  # pragma: no cover - defensive
                raise SearchError(f"budget invariant broken on {task.task_id}/{strategy}")
        if deps.progress is not None:
            deps.progress(f"{strategy}: {len(suite)} tasks")
    return runs


def summarise(
    suite: TaskSuite,
    config: SweepConfig,
    deps: SweepDeps,
    runs: Sequence[TaskStrategyRun],
    *,
    wall_seconds: float,
) -> SweepResults:
    """Aggregate a finished matrix into the results file, one row per (strategy, level)."""
    levels = tuple(
        aggregate(
            runs,
            strategy=strategy,
            candidates=candidates,
            resamples=config.bootstrap_resamples,
            seed=config.seed,
        )
        for strategy in config.strategies
        for candidates in config.budget_levels
    )
    to_target = {
        strategy: compute_to_target(
            [item for item in levels if item.strategy == strategy], config.target_solve_rate
        )
        for strategy in config.strategies
    }
    record = getattr(deps.sandbox, "record", None)
    return SweepResults(
        config=config,
        suite_sha256=suite.sha256,
        task_count=len(suite),
        family_count=len(suite.families),
        generator=deps.generator.name,
        generator_model=deps.generator.model,
        value_model_version=(deps.value_model.version if deps.value_model is not None else None),
        levels=levels,
        compute_to_target=to_target,
        total_evaluations=sum(len(run.outcome.evaluations) for run in runs),
        total_sandbox_seconds=round(
            sum(item.verdict.duration_seconds for run in runs for item in run.outcome.evaluations),
            3,
        ),
        wall_seconds=round(wall_seconds, 3),
        cache_hit_ratio=round(getattr(record, "cache_hit_ratio", 0.0), 4),
    )


def run_sweep(suite: TaskSuite, config: SweepConfig, deps: SweepDeps) -> SweepResults:
    """Run the matrix and aggregate it. The common entry point."""
    started = time.perf_counter()
    runs = run_matrix(suite, config, deps)
    return summarise(suite, config, deps, runs, wall_seconds=time.perf_counter() - started)


def default_generator(config: SweepConfig) -> MutationGenerator:
    """The offline generator configured from a sweep config."""
    return MutationGenerator(seed=config.generator_seed, noise=config.generator_noise)
