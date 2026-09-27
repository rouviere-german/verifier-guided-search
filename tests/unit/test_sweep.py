"""Aggregation: turning runs into the numbers the README prints, with fake runs so it is exact.

These tests build :class:`TaskStrategyRun` objects by hand, so the arithmetic of every reported
metric is checked against values worked out on paper rather than against whatever the search
happened to do. The search side of the same code is covered by
``tests/integration/test_budget_prefix.py``.
"""

from __future__ import annotations

import pytest

from verifier_guided_search.core.canonical import sha256_hex
from verifier_guided_search.errors import SearchError
from verifier_guided_search.experiments.sweep import (
    DEFAULT_BUDGET_LEVELS,
    DEFAULT_STRATEGIES,
    SweepConfig,
    TaskStrategyRun,
    aggregate,
    compute_to_target,
)
from verifier_guided_search.sandbox.protocol import (
    GroupResult,
    TestResult,
    Verdict,
    VisibleFeedback,
)
from verifier_guided_search.search.budget import BudgetUsage
from verifier_guided_search.search.strategies import Evaluation, SearchOutcome
from verifier_guided_search.value.features import FEATURE_NAMES

pytestmark = pytest.mark.unit


def evaluation(
    order: int,
    *,
    visible: bool,
    hidden: bool,
    program: str | None = None,
    seconds: float = 0.1,
) -> Evaluation:
    name = program or f"p{order}"
    digest = sha256_hex(name)
    visible_group = GroupResult(
        results=(TestResult(name="v00", status="pass" if visible else "fail"),)
    )
    hidden_group = GroupResult(
        results=(TestResult(name="h00", status="pass" if hidden else "fail"),)
    )
    feedback = VisibleFeedback(program_sha256=digest, status="ok", visible=visible_group)
    verdict = Verdict(
        program_sha256=digest,
        suite_sha256=sha256_hex("suite"),
        status="ok",
        visible=visible_group,
        hidden=hidden_group,
        duration_seconds=seconds,
    )
    return Evaluation(
        order=order,
        node_index=order,
        depth=order,
        program=name,
        program_sha256=digest,
        edit_distance_from_root=order,
        feedback=feedback,
        verdict=verdict,
        features=tuple(0.0 for _ in FEATURE_NAMES),
        value_estimate=0.5,
        cache_hit=False,
    )


def run(
    task: str,
    strategy: str,
    evaluations: list[Evaluation],
    *,
    tokens: int = 1_000,
) -> TaskStrategyRun:
    outcome = SearchOutcome(
        strategy=strategy,
        task_id=task,
        family=task.split("/", maxsplit=1)[0],
        evaluations=tuple(evaluations),
        stop_reason="solved" if any(e.solved_visible for e in evaluations) else "exhausted",
        usage=BudgetUsage(candidates=len(evaluations), generator_tokens=tokens),
        nodes=len(evaluations) + 1,
        expansions=len(evaluations),
        best_order=0,
        generator_calls=len(evaluations),
    )
    return TaskStrategyRun(
        task_id=task, family=outcome.family, strategy=strategy, outcome=outcome, wall_seconds=0.5
    )


def test_a_solve_on_the_fourth_evaluation_counts_from_budget_four_onwards() -> None:
    evaluations = [
        evaluation(0, visible=False, hidden=False),
        evaluation(1, visible=False, hidden=False),
        evaluation(2, visible=False, hidden=False),
        evaluation(3, visible=True, hidden=True),
    ]
    runs = [run("f/00", "mcts", evaluations)]
    assert aggregate(runs, strategy="mcts", candidates=2, resamples=0).hidden_solve_rate == 0.0
    assert aggregate(runs, strategy="mcts", candidates=4, resamples=0).hidden_solve_rate == 1.0
    assert aggregate(runs, strategy="mcts", candidates=8, resamples=0).hidden_solve_rate == 1.0


def test_a_visible_pass_with_a_hidden_failure_is_a_false_pass_not_a_solve() -> None:
    runs = [run("f/00", "beam", [evaluation(0, visible=True, hidden=False)])]
    stats = aggregate(runs, strategy="beam", candidates=1, resamples=0)
    assert stats.hidden_solve_rate == 0.0
    assert stats.visible_solve_rate == 1.0
    assert stats.false_pass_rate == 1.0
    assert stats.hidden_solved == 0
    assert stats.visible_solved == 1
    assert stats.false_passes == 1


def test_rates_are_computed_over_tasks() -> None:
    runs = [
        run("a/00", "mcts", [evaluation(0, visible=True, hidden=True)]),
        run("b/00", "mcts", [evaluation(0, visible=True, hidden=False)]),
        run("c/00", "mcts", [evaluation(0, visible=False, hidden=False)]),
        run("d/00", "mcts", [evaluation(0, visible=False, hidden=False)]),
    ]
    stats = aggregate(runs, strategy="mcts", candidates=1, resamples=0)
    assert stats.tasks == 4
    assert stats.hidden_solve_rate == pytest.approx(0.25)
    assert stats.visible_solve_rate == pytest.approx(0.5)
    assert stats.false_pass_rate == pytest.approx(0.25)


def test_the_bootstrap_interval_brackets_the_estimate() -> None:
    runs = [
        run(f"t{index}/00", "mcts", [evaluation(0, visible=index < 6, hidden=index < 6)])
        for index in range(12)
    ]
    stats = aggregate(runs, strategy="mcts", candidates=1, resamples=300, seed=1)
    assert stats.hidden_solve_rate == pytest.approx(0.5)
    assert stats.hidden_ci_low <= stats.hidden_solve_rate <= stats.hidden_ci_high
    assert stats.hidden_ci_low < stats.hidden_ci_high


def test_only_the_named_strategys_runs_are_aggregated() -> None:
    runs = [
        run("a/00", "mcts", [evaluation(0, visible=True, hidden=True)]),
        run("a/00", "greedy", [evaluation(0, visible=False, hidden=False)]),
    ]
    assert aggregate(runs, strategy="mcts", candidates=1, resamples=0).hidden_solve_rate == 1.0
    assert aggregate(runs, strategy="greedy", candidates=1, resamples=0).hidden_solve_rate == 0.0
    with pytest.raises(SearchError, match="no runs for strategy"):
        aggregate(runs, strategy="beam", candidates=1, resamples=0)


def test_usage_at_a_level_sums_only_the_prefixs_sandbox_time() -> None:
    evaluations = [
        evaluation(0, visible=False, hidden=False, seconds=0.2),
        evaluation(1, visible=False, hidden=False, seconds=0.3),
        evaluation(2, visible=True, hidden=True, seconds=0.5),
    ]
    item = run("f/00", "mcts", evaluations, tokens=900)
    count, seconds, tokens = item.usage_at(2)
    assert count == 2
    assert seconds == pytest.approx(0.5)
    assert tokens == 600, "tokens are pro-rated by the share of evaluations in the prefix"
    count, seconds, tokens = item.usage_at(3)
    assert (count, tokens) == (3, 900)
    assert seconds == pytest.approx(1.0)


def test_mean_usage_is_reported_per_task() -> None:
    runs = [
        run("a/00", "mcts", [evaluation(0, visible=True, hidden=True, seconds=0.4)]),
        run(
            "b/00",
            "mcts",
            [
                evaluation(0, visible=False, hidden=False, seconds=0.2),
                evaluation(1, visible=True, hidden=True, seconds=0.2),
            ],
        ),
    ]
    stats = aggregate(runs, strategy="mcts", candidates=4, resamples=0)
    assert stats.mean_candidates_used == pytest.approx(1.5)
    assert stats.mean_sandbox_seconds == pytest.approx(0.4)


def test_a_run_that_evaluated_nothing_counts_as_unsolved() -> None:
    runs = [run("a/00", "mcts", [])]
    stats = aggregate(runs, strategy="mcts", candidates=4, resamples=0)
    assert stats.hidden_solve_rate == 0.0
    assert stats.mean_candidates_used == 0.0
    assert runs[0].answer_at(4) is None
    assert runs[0].usage_at(4) == (0, 0.0, 0)


def test_compute_to_target_finds_the_first_level_that_reaches_it() -> None:
    levels = [
        aggregate(
            [
                run(
                    f"t{index}/00",
                    "mcts",
                    [
                        evaluation(
                            order, visible=order == 3 and index < 7, hidden=order == 3 and index < 7
                        )
                        for order in range(4)
                    ],
                )
                for index in range(10)
            ],
            strategy="mcts",
            candidates=candidates,
            resamples=0,
        )
        for candidates in (1, 2, 4, 8)
    ]
    assert [item.hidden_solve_rate for item in levels] == [0.0, 0.0, 0.7, 0.7]
    assert compute_to_target(levels, 0.6) == 4
    assert compute_to_target(levels, 0.7) == 4
    assert compute_to_target(levels, 0.9) is None


def test_compute_to_target_reads_levels_in_ascending_order() -> None:
    runs = [run("a/00", "mcts", [evaluation(0, visible=True, hidden=True)])]
    levels = [
        aggregate(runs, strategy="mcts", candidates=candidates, resamples=0)
        for candidates in (8, 1, 4)
    ]
    assert compute_to_target(levels, 0.5) == 1, "levels must be sorted before the first crossing"


def test_the_config_validates_its_strategies_and_levels() -> None:
    with pytest.raises(SearchError, match="unknown strategies"):
        SweepConfig(seed=1, instances_per_family=1, strategies=("nope",))
    with pytest.raises(SearchError, match="ascending"):
        SweepConfig(seed=1, instances_per_family=1, budget_levels=(4, 2))
    with pytest.raises(SearchError, match="ascending"):
        SweepConfig(seed=1, instances_per_family=1, budget_levels=())
    with pytest.raises(SearchError, match="distinct"):
        SweepConfig(seed=1, instances_per_family=1, budget_levels=(1, 1, 2))


def test_the_default_config_matches_the_documented_shape() -> None:
    config = SweepConfig(seed=1, instances_per_family=2)
    assert config.strategies == DEFAULT_STRATEGIES
    assert config.budget_levels == DEFAULT_BUDGET_LEVELS
    assert len(config.budget_levels) == 5, "the spec asks for five budget levels"
    assert config.max_candidates == config.budget_levels[-1]
    assert config.target_solve_rate == pytest.approx(0.60)


def test_only_the_candidate_cap_changes_between_levels() -> None:
    from verifier_guided_search.search.generator import MutationGenerator

    config = SweepConfig(seed=1, instances_per_family=1)
    generator = MutationGenerator()
    small = config.budget(generator, candidates=1)
    large = config.budget(generator, candidates=16)
    assert small.max_candidates == 1
    assert large.max_candidates == 16
    assert small.max_generator_tokens == large.max_generator_tokens
    assert small.max_sandbox_micros == large.max_sandbox_micros
    assert small.max_usd_micros == large.max_usd_micros


def test_the_search_config_carries_the_sweeps_knobs() -> None:
    config = SweepConfig(
        seed=1,
        instances_per_family=1,
        children_per_expansion=6,
        beam_width=5,
        c_puct=0.9,
        use_value_model=False,
        timeout_seconds=2.5,
    )
    search = config.search_config("beam")
    assert search.strategy == "beam"
    assert search.children_per_expansion == 6
    assert search.beam_width == 5
    assert search.c_puct == pytest.approx(0.9)
    assert search.use_value_model is False
    assert search.limits.timeout_seconds == pytest.approx(2.5)
