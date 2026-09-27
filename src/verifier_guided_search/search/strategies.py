"""The four strategies, and the driver that runs one of them under a budget.

All four are written as generators that yield one :class:`Evaluation` at a time and never look at
how much budget is left. That gives the property the whole evaluation rests on: **a run at budget B
is exactly the first B evaluations of a run at any larger budget**. So a budget sweep costs one run
per
(task, strategy) at the largest level instead of one per level, every level is compared on identical
work, and ``tests/integration/test_budget_prefix.py`` checks the claim directly.

The strategies:

``greedy``
    Ask for one candidate, take it, move there, repeat. A single chain with no backtracking - the
    "one sample per step" baseline.
``best_of_n``
    Expand the root over and over and verify each proposal in rank order. Never goes deeper; this is
    best-of-N sampling with a verifier.
``beam``
    Keep ``width`` nodes, expand each for ``k`` children, verify children as they appear, then keep
    the best ``width`` of them by value estimate.
``mcts``
    PUCT from the root to a leaf, expand, verify one child, back the value up the path. The value
    comes from the value model when one is configured and from the visible pass fraction otherwise,
    which is exactly the "value model on/off" ablation.

Every strategy stops as soon as a candidate passes all *visible* tests. That is what a real system
would do, and it is what makes the verifier false-pass rate a real number: the search commits to an
answer the visible tests approve of, and the hidden tests then say whether it was right.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Final

from verifier_guided_search.agentic.audit import AuditLog
from verifier_guided_search.agentic.errors import ProviderError
from verifier_guided_search.errors import BudgetError, BudgetExhausted, SearchError
from verifier_guided_search.metrics import SearchMetrics
from verifier_guided_search.mutate import hamming_distance
from verifier_guided_search.sandbox.client import SandboxClient
from verifier_guided_search.sandbox.protocol import ExecJob, ExecLimits, Verdict, VisibleFeedback
from verifier_guided_search.search.budget import BudgetAccountant, BudgetUsage
from verifier_guided_search.search.generator import CandidateGenerator
from verifier_guided_search.search.node import SearchNode, SearchTree
from verifier_guided_search.search.puct import DEFAULT_C_PUCT, ChildStats, select_index
from verifier_guided_search.search.trace import TraceRow, TraceStore
from verifier_guided_search.tasks.models import Task
from verifier_guided_search.value.features import FEATURE_NAMES, FeatureContext, node_features
from verifier_guided_search.value.model import GradientBoostedValueModel

__all__ = [
    "STRATEGIES",
    "Evaluation",
    "SearchConfig",
    "SearchContext",
    "SearchOutcome",
    "run_search",
]

_MAX_SELECT_ATTEMPTS: Final = 64
_MAX_DEPTH: Final = 32


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One verified candidate, in the order the strategy produced it."""

    order: int
    node_index: int
    depth: int
    program: str
    program_sha256: str
    edit_distance_from_root: int
    feedback: VisibleFeedback
    verdict: Verdict
    features: tuple[float, ...]
    value_estimate: float
    cache_hit: bool

    @property
    def solved_visible(self) -> bool:
        return self.feedback.solved

    @property
    def solved_hidden(self) -> bool:
        return self.verdict.hidden_solved

    @property
    def false_pass(self) -> bool:
        return self.verdict.false_pass


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Knobs the ablations sweep. ``use_value_model`` is the value-model on/off switch."""

    strategy: str = "mcts"
    children_per_expansion: int = 4
    beam_width: int = 3
    c_puct: float = DEFAULT_C_PUCT
    use_value_model: bool = True
    # ExecLimits is a frozen pydantic model, so one shared instance cannot be mutated by a run.
    limits: ExecLimits = ExecLimits()  # noqa: RUF009

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGY_NAMES:
            raise SearchError(
                f"unknown strategy {self.strategy!r}; known: {', '.join(STRATEGY_NAMES)}"
            )
        if self.children_per_expansion < 1:
            raise SearchError("children_per_expansion must be at least 1")
        if self.beam_width < 1:
            raise SearchError("beam_width must be at least 1")


@dataclass
class SearchContext:
    """Everything one search needs. Built fresh per (task, strategy) run."""

    sandbox: SandboxClient
    generator: CandidateGenerator
    accountant: BudgetAccountant
    config: SearchConfig = field(default_factory=SearchConfig)
    value_model: GradientBoostedValueModel | None = None
    metrics: SearchMetrics | None = None
    audit: AuditLog | None = None
    traces: TraceStore | None = None
    actor: str = "search"


@dataclass
class SearchOutcome:
    """What a run produced: the ordered evaluations, the answer, and what it cost."""

    strategy: str
    task_id: str
    family: str
    evaluations: tuple[Evaluation, ...]
    stop_reason: str
    usage: BudgetUsage
    nodes: int
    expansions: int
    best_order: int | None
    generator_calls: int

    @property
    def solved_visible(self) -> bool:
        return any(item.solved_visible for item in self.evaluations)

    @property
    def answer(self) -> Evaluation | None:
        return best_evaluation(self.evaluations)

    def prefix(self, candidates: int) -> tuple[Evaluation, ...]:
        """The evaluations a run with ``candidates`` of budget would have produced."""
        if candidates < 0:
            raise SearchError("candidates must not be negative")
        out: list[Evaluation] = []
        for item in self.evaluations[:candidates]:
            out.append(item)
            if item.solved_visible:
                break
        return tuple(out)


def best_evaluation(evaluations: tuple[Evaluation, ...]) -> Evaluation | None:
    """The answer: a visible-solving candidate if any, else the best visible pass fraction."""
    if not evaluations:
        return None
    solved = [item for item in evaluations if item.solved_visible]
    pool = solved or list(evaluations)
    return min(pool, key=lambda item: (-item.feedback.pass_fraction, item.depth, item.order))


class _Runner:
    """Shared machinery: evaluating a node, expanding a node, scoring, tracing and auditing."""

    def __init__(self, task: Task, context: SearchContext) -> None:
        self.task = task
        self.ctx = context
        self.tree = SearchTree(task.start_program, tuple(task.start_state))
        self.order = 0
        self.expansions = 0
        self.generator_calls = 0
        self.stop_reason = "exhausted"

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, node: SearchNode) -> Evaluation:
        """Verify one candidate. Raises :class:`BudgetExhausted` before spending anything."""
        job = ExecJob(
            program=node.program,
            entry_point=self.task.entry_point,
            visible=self.task.visible.cases,
            hidden=self.task.hidden.cases,
            limits=self.ctx.config.limits,
        )
        reservation = self.ctx.accountant.reserve_evaluation(
            timeout_seconds=self.ctx.config.limits.timeout_seconds
        )
        cached = job.cache_key in _cache_keys(self.ctx)
        try:
            verdict = self.ctx.sandbox.execute(job)
        except BaseException:
            reservation.release()
            raise
        self.ctx.accountant.settle_evaluation(reservation, seconds=verdict.duration_seconds)
        feedback = verdict.visible_only()
        node.feedback = feedback
        parent_pass = node.parent.pass_fraction if node.parent is not None else 0.0
        distance = hamming_distance(node.state, self.tree.root.state)
        features = node_features(
            feedback,
            FeatureContext(
                depth=node.depth,
                edit_distance_from_root=distance,
                program_chars=len(node.program),
                parent_pass_fraction=parent_pass,
            ),
        )
        node.value = self.score(features, feedback)
        evaluation = Evaluation(
            order=self.order,
            node_index=node.index,
            depth=node.depth,
            program=node.program,
            program_sha256=node.program_sha256,
            edit_distance_from_root=distance,
            feedback=feedback,
            verdict=verdict,
            features=features,
            value_estimate=node.value,
            cache_hit=cached,
        )
        self.order += 1
        self._publish(evaluation)
        return evaluation

    def score(self, features: tuple[float, ...], feedback: VisibleFeedback) -> float:
        """Value of a node: the value model's probability, or the visible pass fraction."""
        if self.ctx.config.use_value_model and self.ctx.value_model is not None:
            return self.ctx.value_model.score_one(features)
        return feedback.pass_fraction

    def _publish(self, evaluation: Evaluation) -> None:
        strategy = self.ctx.config.strategy
        if self.ctx.metrics is not None:
            self.ctx.metrics.programs_evaluated_total.labels(strategy).inc()
        if self.ctx.audit is not None:
            self.ctx.audit.append(
                self.ctx.actor,
                "verifier.verdict",
                {
                    "task_id": self.task.task_id,
                    "strategy": strategy,
                    "program_sha256": evaluation.program_sha256,
                    "status": evaluation.verdict.status,
                    "visible_passed": evaluation.feedback.visible.passed,
                    "visible_total": evaluation.feedback.visible.total,
                    "hidden_passed": evaluation.verdict.hidden.passed,
                    "hidden_total": evaluation.verdict.hidden.total,
                    "cache_hit": evaluation.cache_hit,
                    "violations": list(evaluation.verdict.violations),
                },
            )
        if self.ctx.traces is not None:
            self.ctx.traces.append(
                TraceRow(
                    task_id=self.task.task_id,
                    family=self.task.family,
                    strategy=strategy,
                    order=evaluation.order,
                    program_sha256=evaluation.program_sha256,
                    features=evaluation.features,
                    feature_names=FEATURE_NAMES,
                    visible_all_passed=evaluation.feedback.visible.all_passed,
                    hidden_all_passed=evaluation.verdict.hidden.all_passed,
                    status=evaluation.verdict.status,
                    cache_hit=evaluation.cache_hit,
                )
            )

    # ---------------------------------------------------------------- expansion
    def expand(self, node: SearchNode, k: int) -> list[SearchNode]:
        """Ask the generator for children of ``node`` and attach them (unevaluated)."""
        if node.depth >= _MAX_DEPTH:
            node.terminal = True
            return []
        input_tokens = self.ctx.generator.estimate_input_tokens(self.task, node, k)
        reservation = self.ctx.accountant.reserve_generation(
            model=self.ctx.generator.model,
            input_tokens=input_tokens,
            max_output_tokens=self.ctx.generator.max_output_tokens(k),
        )
        try:
            generation = self.ctx.generator.expand(self.task, node, k)
        except BaseException:
            reservation.release()
            raise
        self.ctx.accountant.settle_generation(
            reservation,
            model=generation.model,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
        )
        node.expansions += 1
        self.expansions += 1
        self.generator_calls += 1
        if self.ctx.metrics is not None:
            self.ctx.metrics.nodes_expanded.labels(self.ctx.config.strategy).inc()
        if not generation.candidates:
            node.terminal = True
            return []
        return [
            self.tree.add_child(
                node, program=candidate.program, state=candidate.state, prior=candidate.prior
            )
            for candidate in generation.candidates
        ]


def _cache_keys(context: SearchContext) -> frozenset[str]:
    """Cache keys known before this execution, so an evaluation can report a cache hit."""
    cache = getattr(context.sandbox, "cache", None)
    if cache is None:
        return frozenset()
    return frozenset(cache.keys())


# ------------------------------------------------------------------------- strategies


def _greedy(runner: _Runner) -> Iterator[Evaluation]:
    node = runner.tree.root
    evaluation = runner.evaluate(node)
    yield evaluation
    if evaluation.solved_visible:
        return
    while True:
        children = runner.expand(node, 1)
        if not children:
            return
        node = children[0]
        evaluation = runner.evaluate(node)
        yield evaluation
        if evaluation.solved_visible:
            return


def _best_of_n(runner: _Runner) -> Iterator[Evaluation]:
    root = runner.tree.root
    evaluation = runner.evaluate(root)
    yield evaluation
    if evaluation.solved_visible:
        return
    k = runner.ctx.config.children_per_expansion
    while True:
        children = runner.expand(root, k)
        if not children:
            return
        for child in children:
            evaluation = runner.evaluate(child)
            yield evaluation
            if evaluation.solved_visible:
                return


def _beam(runner: _Runner) -> Iterator[Evaluation]:
    root = runner.tree.root
    evaluation = runner.evaluate(root)
    yield evaluation
    if evaluation.solved_visible:
        return
    config = runner.ctx.config
    frontier = [root]
    while frontier:
        produced: list[SearchNode] = []
        for parent in frontier:
            children = runner.expand(parent, config.children_per_expansion)
            for child in children:
                evaluation = runner.evaluate(child)
                yield evaluation
                if evaluation.solved_visible:
                    return
                produced.append(child)
        if not produced:
            return
        produced.sort(key=lambda node: (-node.value, -node.pass_fraction, node.index))
        frontier = produced[: config.beam_width]


def _mcts(runner: _Runner) -> Iterator[Evaluation]:
    root = runner.tree.root
    evaluation = runner.evaluate(root)
    yield evaluation
    if evaluation.solved_visible:
        return
    # c_puct and k are read inside _select_leaf, which owns the descent.
    while True:
        target = _select_leaf(runner, root)
        if target is None:
            return
        evaluation = runner.evaluate(target)
        target.backup(target.value)
        yield evaluation
        if evaluation.solved_visible:
            return


def _select_leaf(runner: _Runner, root: SearchNode) -> SearchNode | None:
    """Walk down by PUCT to an unevaluated child, expanding leaves on the way."""
    c_puct = runner.ctx.config.c_puct
    k = runner.ctx.config.children_per_expansion
    for _ in range(_MAX_SELECT_ATTEMPTS):
        node = root
        while True:
            pending = [child for child in node.children if not child.evaluated]
            if pending:
                return max(pending, key=lambda child: (child.prior, -child.index))
            if not node.children and not node.terminal:
                if not runner.expand(node, k):
                    break  # dead end: restart the descent, this node is now terminal
                continue
            options = [child for child in node.children if not child.terminal]
            if not options:
                node.terminal = True
                break
            stats = [
                ChildStats(
                    prior=child.prior,
                    visits=child.visits,
                    total_value=child.total_value,
                    fallback_value=node.value,
                )
                for child in options
            ]
            node = options[select_index(stats, node.visits, c_puct)]
        if root.terminal:
            return None
    return None


Strategy = Callable[[_Runner], Iterator[Evaluation]]
STRATEGIES: Mapping[str, Strategy] = {
    "greedy": _greedy,
    "best_of_n": _best_of_n,
    "beam": _beam,
    "mcts": _mcts,
}
STRATEGY_NAMES: tuple[str, ...] = tuple(STRATEGIES)


def run_search(task: Task, context: SearchContext) -> SearchOutcome:
    """Run one strategy on one task until it solves, runs out of budget or runs out of
    candidates."""
    runner = _Runner(task, context)
    strategy = STRATEGIES[context.config.strategy]
    evaluations: list[Evaluation] = []
    stop_reason = "exhausted"
    try:
        for evaluation in strategy(runner):
            evaluations.append(evaluation)
            if evaluation.solved_visible:
                stop_reason = "solved"
                break
    except BudgetExhausted as exhausted:
        stop_reason = f"budget:{exhausted.dimension}"
        if context.metrics is not None:
            context.metrics.budget_exhausted_total.labels(exhausted.dimension).inc()
    except BudgetError as mistake:
        # Not an exhausted budget: the accountant was used incorrectly (settled twice, a settle that
        # does not fit its reservation). That is a bug, and it is reported as a stop reason so it
        # surfaces in the results instead of aborting a whole sweep with a traceback.
        stop_reason = f"budget_error:{type(mistake).__name__}"
    except ProviderError as failure:
        stop_reason = f"provider:{type(failure).__name__}"
    best = best_evaluation(tuple(evaluations))
    outcome = SearchOutcome(
        strategy=context.config.strategy,
        task_id=task.task_id,
        family=task.family,
        evaluations=tuple(evaluations),
        stop_reason=stop_reason,
        usage=context.accountant.committed,
        nodes=len(runner.tree),
        expansions=runner.expansions,
        best_order=best.order if best is not None else None,
        generator_calls=runner.generator_calls,
    )
    if context.metrics is not None:
        outcome_label = "solved" if outcome.solved_visible else stop_reason.split(":")[0]
        context.metrics.searches_total.labels(context.config.strategy, outcome_label).inc()
    if context.audit is not None:
        context.audit.append(
            context.actor,
            "search.finished",
            {
                "task_id": task.task_id,
                "strategy": context.config.strategy,
                "stop_reason": stop_reason,
                "evaluations": len(evaluations),
                "usage": dict(context.accountant.committed.as_dict()),
                "generator_token_overshoot": context.accountant.generator_token_overshoot,
            },
        )
    return outcome
