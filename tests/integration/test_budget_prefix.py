"""The claim the whole budget sweep rests on: a smaller budget is a prefix of a larger one.

``experiments/sweep.py`` runs each (task, strategy) pair once at the largest budget and reads the
smaller levels off as prefixes. That is five times less compute, and it is only legitimate if a run
with a candidate cap of ``B`` really does produce exactly the first ``B`` evaluations of a run with
a larger cap. Nothing enforces that structurally - a strategy that peeked at the remaining budget
would break it silently, and every curve in the README would become fiction.

So this file runs the search *again* with a real, smaller accountant and compares evaluation by
evaluation, for all four strategies, on real tasks, in the real sandbox.
"""

from __future__ import annotations

import pytest

from verifier_guided_search.experiments.sweep import (
    DEFAULT_STRATEGIES,
    SweepConfig,
    SweepDeps,
    default_generator,
    run_matrix,
)
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.search.budget import BudgetAccountant
from verifier_guided_search.search.strategies import (
    Evaluation,
    SearchContext,
    SearchOutcome,
    run_search,
)
from verifier_guided_search.tasks.suite import TaskSuite

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]

_SEED = 20260927
_LEVELS = (1, 2, 4)


def config(strategy: str) -> SweepConfig:
    return SweepConfig(
        seed=_SEED,
        instances_per_family=1,
        strategies=(strategy,),
        budget_levels=_LEVELS,
        bootstrap_resamples=0,
    )


def identity(evaluation: Evaluation) -> tuple[int, int, str, str]:
    """What must match between a prefix and a real smaller run."""
    return (
        evaluation.order,
        evaluation.node_index,
        evaluation.program_sha256,
        evaluation.feedback.digest,
    )


def run_at(task: object, strategy: str, candidates: int, sandbox: CachingSandbox) -> SearchOutcome:
    settings = config(strategy)
    generator = default_generator(settings)
    return run_search(
        task,  # type: ignore[arg-type]
        SearchContext(
            sandbox=sandbox,
            generator=generator,
            accountant=BudgetAccountant(settings.budget(generator, candidates=candidates)),
            config=settings.search_config(strategy),
            actor="prefix-test",
        ),
    )


@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_a_real_smaller_run_equals_the_prefix_of_a_larger_one(
    strategy: str, tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """The claim, checked directly: same evaluations, same order, same verdicts."""
    for task in tiny_suite:
        full = run_at(task, strategy, _LEVELS[-1], shared_sandbox)
        for level in _LEVELS:
            actual = run_at(task, strategy, level, shared_sandbox)
            expected = full.prefix(level)
            assert [identity(item) for item in actual.evaluations] == [
                identity(item) for item in expected
            ], f"{strategy} on {task.task_id} at budget {level}"


@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_the_prefix_never_exceeds_the_budget(
    strategy: str, tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    for task in tiny_suite:
        full = run_at(task, strategy, _LEVELS[-1], shared_sandbox)
        for level in _LEVELS:
            assert len(full.prefix(level)) <= level


@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_a_smaller_run_commits_to_the_same_answer_as_the_prefix(
    strategy: str, tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """The aggregate scores the *answer*, so the answers have to agree too, not just the rows."""
    from verifier_guided_search.search.strategies import best_evaluation

    for task in tiny_suite:
        full = run_at(task, strategy, _LEVELS[-1], shared_sandbox)
        for level in _LEVELS:
            actual = run_at(task, strategy, level, shared_sandbox)
            from_prefix = best_evaluation(full.prefix(level))
            assert (actual.answer is None) == (from_prefix is None)
            if actual.answer is not None and from_prefix is not None:
                assert actual.answer.program_sha256 == from_prefix.program_sha256
                assert actual.answer.solved_hidden == from_prefix.solved_hidden
                assert actual.answer.false_pass == from_prefix.false_pass


def test_the_prefix_stops_at_the_first_visible_solve(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """A run that solved at evaluation 2 did not do a third, whatever the cap allowed."""
    for task in tiny_suite:
        for strategy in DEFAULT_STRATEGIES:
            outcome = run_at(task, strategy, _LEVELS[-1], shared_sandbox)
            solved_at = next(
                (item.order for item in outcome.evaluations if item.solved_visible), None
            )
            if solved_at is None:
                continue
            larger = outcome.prefix(_LEVELS[-1])
            assert len(larger) == solved_at + 1
            assert all(not item.solved_visible for item in larger[:-1])


def test_a_prefix_of_zero_is_empty_and_a_negative_one_is_refused(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    from verifier_guided_search.errors import SearchError

    outcome = run_at(tiny_suite.tasks[0], "greedy", _LEVELS[-1], shared_sandbox)
    assert outcome.prefix(0) == ()
    with pytest.raises(SearchError, match="must not be negative"):
        outcome.prefix(-1)


def test_the_matrix_is_reproducible_run_to_run(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """Same seed, same suite, same rows - otherwise a results file cannot be reproduced."""
    settings = SweepConfig(
        seed=_SEED,
        instances_per_family=1,
        strategies=("mcts", "beam"),
        budget_levels=_LEVELS,
        bootstrap_resamples=0,
    )
    generator = default_generator(settings)
    deps = SweepDeps(sandbox=shared_sandbox, generator=generator)
    first = run_matrix(tiny_suite, settings, deps)
    again = run_matrix(tiny_suite, settings, SweepDeps(sandbox=shared_sandbox, generator=generator))
    assert [
        (run.task_id, run.strategy, [identity(item) for item in run.outcome.evaluations])
        for run in first
    ] == [
        (run.task_id, run.strategy, [identity(item) for item in run.outcome.evaluations])
        for run in again
    ]


def test_the_accountant_stays_within_the_budget_in_every_run(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """The spec's property, re-checked against real sandbox timings rather than synthetic usage."""
    for strategy in DEFAULT_STRATEGIES:
        settings = config(strategy)
        generator = default_generator(settings)
        for task in tiny_suite:
            for level in _LEVELS:
                spec = settings.budget(generator, candidates=level)
                accountant = BudgetAccountant(spec)
                outcome = run_search(
                    task,
                    SearchContext(
                        sandbox=shared_sandbox,
                        generator=generator,
                        accountant=accountant,
                        config=settings.search_config(strategy),
                        actor="budget-test",
                    ),
                )
                assert accountant.invariant_holds()
                assert accountant.committed.exceeds(spec) is None
                assert accountant.committed.candidates == len(outcome.evaluations)
                assert len(outcome.evaluations) <= level
