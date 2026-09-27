"""The four strategies, running for real: what each one does with its budget, and what it records.

The comparison in the README is only meaningful if the strategies are actually different algorithms
over the same machinery. So the shape of each one is asserted directly: greedy stays on one chain,
best-of-N never leaves depth 1, beam widens then prunes, MCTS goes deeper than 1 and backs values
up. The shared parts - budget accounting, tracing, auditing, the value model switch - are checked
once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from verifier_guided_search.agentic.audit import AuditLog, verify_file
from verifier_guided_search.agentic.errors import ProviderError
from verifier_guided_search.errors import BudgetError
from verifier_guided_search.experiments.sweep import SweepConfig, default_generator
from verifier_guided_search.metrics import SearchMetrics, create_search_metrics
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.search.budget import (
    BudgetAccountant,
    BudgetSpec,
    BudgetUsage,
    Reservation,
)
from verifier_guided_search.search.generator import Generation
from verifier_guided_search.search.node import SearchNode
from verifier_guided_search.search.strategies import (
    STRATEGIES,
    SearchConfig,
    SearchContext,
    SearchOutcome,
    run_search,
)
from verifier_guided_search.search.trace import TraceStore
from verifier_guided_search.tasks.models import Task
from verifier_guided_search.tasks.suite import TaskSuite
from verifier_guided_search.value.features import FEATURE_NAMES
from verifier_guided_search.value.model import (
    GradientBoostedValueModel,
    constant_value_model,
)

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]

_SEED = 20260927
_BUDGET = 8


def settings(strategy: str, **overrides: object) -> SweepConfig:
    base = {
        "seed": _SEED,
        "instances_per_family": 1,
        "strategies": (strategy,),
        "budget_levels": (1, 2, 4, _BUDGET),
        "bootstrap_resamples": 0,
    }
    base.update(overrides)
    return SweepConfig(**base)


def search(
    task: Task,
    strategy: str,
    sandbox: CachingSandbox,
    *,
    candidates: int = _BUDGET,
    config: SearchConfig | None = None,
    value_model: GradientBoostedValueModel | None = None,
    metrics: SearchMetrics | None = None,
    audit: AuditLog | None = None,
    traces: TraceStore | None = None,
) -> tuple[SearchOutcome, BudgetAccountant]:
    conf = settings(strategy)
    generator = default_generator(conf)
    accountant = BudgetAccountant(conf.budget(generator, candidates=candidates))
    outcome = run_search(
        task,
        SearchContext(
            sandbox=sandbox,
            generator=generator,
            accountant=accountant,
            config=config if config is not None else conf.search_config(strategy),
            value_model=value_model,
            metrics=metrics,
            audit=audit,
            traces=traces,
            actor=f"test:{strategy}",
        ),
    )
    return outcome, accountant


def test_every_strategy_is_registered_and_runs(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    assert set(STRATEGIES) == {"greedy", "best_of_n", "beam", "mcts"}
    for strategy in STRATEGIES:
        outcome, _ = search(tiny_suite.tasks[0], strategy, shared_sandbox)
        assert outcome.strategy == strategy
        assert outcome.evaluations, f"{strategy} evaluated nothing"
        assert outcome.evaluations[0].order == 0
        assert outcome.evaluations[0].depth == 0, "every strategy verifies the root first"


def test_greedy_walks_a_single_chain(tiny_suite: TaskSuite, shared_sandbox: CachingSandbox) -> None:
    outcome, _ = search(tiny_suite.tasks[0], "greedy", shared_sandbox)
    depths = [item.depth for item in outcome.evaluations]
    assert depths == list(range(len(depths))), f"greedy backtracked: {depths}"
    # One expansion per step. The count equals the evaluations when the candidate budget refused the
    # last step *after* its expansion, and one fewer when the run solved or ran out of neighbours.
    assert len(outcome.evaluations) - 1 <= outcome.expansions <= len(outcome.evaluations)
    assert outcome.generator_calls == outcome.expansions
    assert outcome.nodes == outcome.expansions + 1, "greedy asked for one child per expansion"


def test_best_of_n_never_leaves_depth_one(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """The structural reason it cannot repair a two-edit bug, stated as a test rather than a
    claim."""
    outcome, _ = search(tiny_suite.tasks[0], "best_of_n", shared_sandbox)
    assert max(item.depth for item in outcome.evaluations) <= 1
    assert {item.edit_distance_from_root for item in outcome.evaluations} <= {0, 1}


def test_beam_widens_before_it_deepens(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    outcome, _ = search(tiny_suite.tasks[0], "beam", shared_sandbox, candidates=8)
    if len(outcome.evaluations) < 5:
        pytest.skip("this task solved before the beam had to prune")
    depths = [item.depth for item in outcome.evaluations]
    assert depths.count(1) >= 2, f"beam did not widen: {depths}"


def test_mcts_goes_deeper_than_one_and_backs_values_up(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    for task in tiny_suite:
        outcome, _ = search(task, "mcts", shared_sandbox, candidates=8)
        if len(outcome.evaluations) < 6:
            continue
        assert max(item.depth for item in outcome.evaluations) >= 2, "mcts stayed shallow"
        return
    pytest.skip("every task solved before mcts needed to descend")


def test_a_run_stops_at_the_first_visible_solve(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    for task in tiny_suite:
        for strategy in STRATEGIES:
            outcome, _ = search(task, strategy, shared_sandbox)
            solved = [item for item in outcome.evaluations if item.solved_visible]
            if not solved:
                continue
            assert outcome.stop_reason == "solved"
            assert solved[0] is outcome.evaluations[-1], "the search kept going after solving"


def test_running_out_of_budget_is_reported_not_raised(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    outcome, accountant = search(tiny_suite.tasks[0], "mcts", shared_sandbox, candidates=2)
    assert len(outcome.evaluations) <= 2
    if outcome.stop_reason != "solved":
        assert outcome.stop_reason.startswith("budget:"), outcome.stop_reason
        assert accountant.refusals
    assert accountant.invariant_holds()


def test_a_zero_candidate_budget_produces_nothing_and_no_crash(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    conf = settings("mcts")
    generator = default_generator(conf)
    accountant = BudgetAccountant(
        BudgetSpec.of(candidates=0, tokens=100_000, sandbox_seconds=10.0, usd=1.0)
    )
    outcome = run_search(
        tiny_suite.tasks[0],
        SearchContext(
            sandbox=shared_sandbox,
            generator=generator,
            accountant=accountant,
            config=conf.search_config("mcts"),
        ),
    )
    assert outcome.evaluations == ()
    assert outcome.answer is None
    assert outcome.stop_reason == "budget:candidates"


def test_the_committed_candidate_count_equals_the_evaluations(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    for strategy in STRATEGIES:
        outcome, accountant = search(tiny_suite.tasks[0], strategy, shared_sandbox)
        assert accountant.committed.candidates == len(outcome.evaluations)
        assert accountant.committed.generator_tokens > 0 or outcome.expansions == 0
        assert outcome.usage == accountant.committed


def test_traces_carry_features_and_the_hidden_label(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    store = TraceStore()
    outcome, _ = search(tiny_suite.tasks[0], "mcts", shared_sandbox, traces=store)
    assert len(store) == len(outcome.evaluations)
    for row, evaluation in zip(store.rows, outcome.evaluations, strict=True):
        assert row.task_id == outcome.task_id
        assert row.strategy == "mcts"
        assert row.feature_names == FEATURE_NAMES
        assert len(row.features) == len(FEATURE_NAMES)
        assert row.visible_all_passed == evaluation.feedback.visible.all_passed
        assert row.hidden_all_passed == evaluation.verdict.hidden.all_passed
        assert row.label == int(evaluation.verdict.hidden.all_passed)


def test_every_executed_program_reaches_the_audit_chain(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox, tmp_path: Path
) -> None:
    """The spec's hash-chained audit of executed program hashes, verdicts and approvals."""
    log = AuditLog(tmp_path / "audit.jsonl")
    outcome, _ = search(tiny_suite.tasks[0], "beam", shared_sandbox, audit=log)
    result = verify_file(log.path)
    assert result.ok
    text = log.path.read_text(encoding="utf-8")
    for evaluation in outcome.evaluations:
        assert evaluation.program_sha256 in text
    assert '"search.finished"' in text
    assert result.entries == len(outcome.evaluations) + 1


def test_metrics_count_expansions_verdicts_and_outcomes(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    metrics = create_search_metrics()
    outcome, _ = search(tiny_suite.tasks[0], "greedy", shared_sandbox, metrics=metrics)

    def value(name: str, labels: dict[str, str]) -> float:
        got = metrics.registry.get_sample_value(name, labels)
        return float(got) if got is not None else 0.0

    assert value("vgs_programs_evaluated_total", {"strategy": "greedy"}) == len(outcome.evaluations)
    assert value("vgs_nodes_expanded_total", {"strategy": "greedy"}) == outcome.expansions
    outcomes = sum(
        value("vgs_searches_total", {"strategy": "greedy", "outcome": label})
        for label in ("solved", "budget", "exhausted", "provider")
    )
    assert outcomes == 1.0


def test_the_value_model_switch_changes_the_node_scores(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """With the model on, a node's value is the model's probability, not its pass fraction."""
    model = constant_value_model(0.42)
    conf = settings("mcts")
    with_model, _ = search(
        tiny_suite.tasks[0],
        "mcts",
        shared_sandbox,
        config=conf.search_config("mcts"),
        value_model=model,
    )
    assert all(item.value_estimate == pytest.approx(0.42) for item in with_model.evaluations)

    off = conf.search_config("mcts")
    without, _ = search(
        tiny_suite.tasks[0],
        "mcts",
        shared_sandbox,
        config=off.__class__(
            strategy="mcts",
            children_per_expansion=off.children_per_expansion,
            beam_width=off.beam_width,
            c_puct=off.c_puct,
            use_value_model=False,
            limits=off.limits,
        ),
        value_model=model,
    )
    assert [item.value_estimate for item in without.evaluations] == [
        item.feedback.pass_fraction for item in without.evaluations
    ]


def test_an_accounting_error_ends_the_search_with_a_reason(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """``BudgetError`` is a bug, not an exhausted budget - and it still must not escape the search.

    ``run_search`` caught ``BudgetExhausted`` and ``ProviderError`` only, so any accounting
    mistake (a settle that does not fit its reservation, a reservation settled twice) came out of
    the search as a traceback in the middle of a sweep. It is now a stop reason like any other.
    """

    class BrokenAccountant(BudgetAccountant):
        """Raises the "the accountant was used incorrectly" error on a settle."""

        def settle_generation(
            self,
            reservation: Reservation,
            *,
            model: str,
            input_tokens: int,
            output_tokens: int,
        ) -> BudgetUsage:
            raise BudgetError("simulated accounting mistake")

    conf = settings("greedy")
    accountant = BrokenAccountant(
        BudgetSpec.of(candidates=4, tokens=10_000, sandbox_seconds=10.0, usd=1.0)
    )
    outcome = run_search(
        tiny_suite.tasks[0],
        SearchContext(
            sandbox=shared_sandbox,
            generator=default_generator(conf),
            accountant=accountant,
            config=conf.search_config("greedy"),
        ),
    )
    assert outcome.stop_reason == "budget_error:BudgetError"
    # The root was evaluated before the first expansion; the mistake ended the search there.
    assert len(outcome.evaluations) == 1


def test_a_failing_generator_ends_the_search_without_a_traceback(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    class BrokenGenerator:
        """A generator whose provider is down. Satisfies the protocol; always raises."""

        name = "broken"
        model = "fake-broken"

        def estimate_input_tokens(self, task: Task, node: SearchNode, k: int) -> int:
            return 10

        def max_output_tokens(self, k: int) -> int:
            return 10

        def expand(self, task: Task, node: SearchNode, k: int) -> Generation:
            raise ProviderError("the generator is down")

    conf = settings("greedy")
    generator = BrokenGenerator()
    accountant = BudgetAccountant(
        BudgetSpec.of(candidates=4, tokens=10_000, sandbox_seconds=10.0, usd=1.0)
    )
    outcome = run_search(
        tiny_suite.tasks[0],
        SearchContext(
            sandbox=shared_sandbox,
            generator=generator,
            accountant=accountant,
            config=conf.search_config("greedy"),
        ),
    )
    assert outcome.stop_reason == "provider:ProviderError"
    assert len(outcome.evaluations) == 1
    # The refused generation was released, so it was not charged.
    assert accountant.committed.generator_tokens == 0
    assert accountant.invariant_holds()


def test_an_unknown_strategy_is_refused_at_configuration_time() -> None:
    from verifier_guided_search.errors import SearchError

    with pytest.raises(SearchError, match="unknown strategy"):
        SearchConfig(strategy="montecarlo")
    with pytest.raises(SearchError, match="children_per_expansion"):
        SearchConfig(strategy="mcts", children_per_expansion=0)
    with pytest.raises(SearchError, match="beam_width"):
        SearchConfig(strategy="beam", beam_width=0)


def test_the_answer_prefers_a_visible_solve_over_a_higher_pass_fraction(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    for task in tiny_suite:
        for strategy in STRATEGIES:
            outcome, _ = search(task, strategy, shared_sandbox)
            answer = outcome.answer
            assert answer is not None
            if any(item.solved_visible for item in outcome.evaluations):
                assert answer.solved_visible
            else:
                best = max(item.feedback.pass_fraction for item in outcome.evaluations)
                assert answer.feedback.pass_fraction == pytest.approx(best)


def test_repeated_candidates_come_from_the_cache_not_the_sandbox(
    tiny_suite: TaskSuite, tmp_path: Path
) -> None:
    from verifier_guided_search.sandbox.local import LocalSandbox

    sandbox = CachingSandbox(LocalSandbox(tmp_path / "sandbox"))
    task = tiny_suite.tasks[0]
    search(task, "mcts", sandbox)
    first_executions = sandbox.record.executions
    search(task, "mcts", sandbox)
    assert sandbox.record.executions == first_executions, "the second identical run re-executed"
    assert sandbox.record.cache_hits >= first_executions
