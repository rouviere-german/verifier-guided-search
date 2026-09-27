"""The ablations, run for real on a tiny suite: one knob changes per row, and nothing else.

The value of an ablation is entirely in its controls, so that is what is tested: two rows differ
only
in the knob named, the value-model row really does turn the model off (rather than leaving it wired
and only flipping a flag), and every row is measured on the same tasks at the same budget.

Whether a knob *helps* is a result, not a property, and is reported in docs/EXPERIMENTS.md from the
bench run. Nothing here asserts a direction.
"""

from __future__ import annotations

import pytest

from verifier_guided_search.experiments.ablations import (
    C_PUCT_OFF_KNOB,
    DEFAULT_C_PUCT_GRID,
    DEFAULT_K_GRID,
    aggregate_at,
    best_value,
    run_ablations,
)
from verifier_guided_search.experiments.sweep import SweepConfig, SweepDeps, default_generator
from verifier_guided_search.sandbox.client import CachingSandbox
from verifier_guided_search.search.trace import TraceStore
from verifier_guided_search.tasks.suite import TaskSuite
from verifier_guided_search.value.model import constant_value_model

pytestmark = [pytest.mark.integration, pytest.mark.sandbox]

_SEED = 20260927
_LEVELS = (1, 2, 4)


@pytest.fixture(scope="module")
def ablations(tiny_suite: TaskSuite, shared_sandbox: CachingSandbox) -> object:
    config = SweepConfig(
        seed=_SEED, instances_per_family=1, budget_levels=_LEVELS, bootstrap_resamples=0
    )
    deps = SweepDeps(
        sandbox=shared_sandbox,
        generator=default_generator(config),
        value_model=constant_value_model(0.5),
    )
    return run_ablations(
        tiny_suite,
        config,
        deps,
        budget=_LEVELS[-1],
        c_puct_grid=(0.0, 1.4),
        k_grid=(2, 4),
    )


def test_every_knob_in_the_spec_is_swept(ablations: object) -> None:
    assert ablations.knobs == (  # type: ignore[attr-defined]
        "c_puct",
        C_PUCT_OFF_KNOB,
        "children_per_expansion",
        "use_value_model",
    )
    assert len(ablations.by_knob("c_puct")) == 2  # type: ignore[attr-defined]
    assert len(ablations.by_knob(C_PUCT_OFF_KNOB)) == 2  # type: ignore[attr-defined]
    assert len(ablations.by_knob("children_per_expansion")) == 2  # type: ignore[attr-defined]
    assert {row.value for row in ablations.by_knob("use_value_model")} == {"on", "off"}  # type: ignore[attr-defined]


def test_the_identifiable_c_puct_sweep_runs_without_the_value_model(ablations: object) -> None:
    """With the value model on, every positive c_puct gives the same ordering, so the grid is swept
    again with it off, where Q is the graded visible pass fraction and the constant matters.

    The mechanism: score = Q + c * P * sqrt(N) / (1 + n). With Q equal across siblings the argmax is
    argmax of P / (1 + n) for *every* positive c - the constant cancels. The value model here scores
    almost every candidate the same, so the positive rows are mechanically identical rather than
    empirically tied.
    """
    on = {row.value: row for row in ablations.by_knob("c_puct")}  # type: ignore[attr-defined]
    off = {row.value: row for row in ablations.by_knob(C_PUCT_OFF_KNOB)}  # type: ignore[attr-defined]
    assert set(on) == set(off)
    # Same values of the knob, and the off rows really ran without the model: their value estimates
    # are pass fractions, so the recorded candidate usage is free to differ from the on rows.
    assert all(row.stats.strategy == "mcts" for row in off.values())


def test_every_row_is_measured_on_the_same_tasks_at_the_same_budget(
    ablations: object, tiny_suite: TaskSuite
) -> None:
    for row in ablations.rows:  # type: ignore[attr-defined]
        assert row.stats.tasks == len(tiny_suite)
        assert row.stats.candidates_budget == _LEVELS[-1]
        assert row.stats.strategy == "mcts", "the ablations sweep MCTS only"
    assert ablations.task_count == len(tiny_suite)  # type: ignore[attr-defined]
    assert ablations.suite_sha256 == tiny_suite.sha256  # type: ignore[attr-defined]


def test_the_value_model_row_really_runs_without_the_model(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """The "off" row must not merely flip a flag while the model is still wired in."""
    from verifier_guided_search.search.budget import BudgetAccountant
    from verifier_guided_search.search.strategies import SearchContext, run_search

    config = SweepConfig(
        seed=_SEED,
        instances_per_family=1,
        strategies=("mcts",),
        budget_levels=_LEVELS,
        bootstrap_resamples=0,
        use_value_model=False,
    )
    generator = default_generator(config)
    outcome = run_search(
        tiny_suite.tasks[0],
        SearchContext(
            sandbox=shared_sandbox,
            generator=generator,
            accountant=BudgetAccountant(config.budget(generator)),
            config=config.search_config("mcts"),
            value_model=constant_value_model(0.5),
        ),
    )
    # With the model off, a node's value is its visible pass fraction, never the constant 0.5.
    assert [item.value_estimate for item in outcome.evaluations] == [
        item.feedback.pass_fraction for item in outcome.evaluations
    ]


def test_ablation_runs_are_not_collected_as_training_data(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    """A row is a measurement, not a trace: feeding it back would make the next model
    self-confirming."""
    store = TraceStore()
    config = SweepConfig(
        seed=_SEED, instances_per_family=1, budget_levels=_LEVELS, bootstrap_resamples=0
    )
    deps = SweepDeps(sandbox=shared_sandbox, generator=default_generator(config), traces=store)
    run_ablations(
        tiny_suite,
        config,
        deps,
        budget=_LEVELS[-1],
        c_puct_grid=(1.4,),
        k_grid=(4,),
        include_value_model=False,
    )
    assert len(store) == 0, "ablation runs wrote trace rows"


def test_a_budget_that_is_not_a_configured_level_is_refused(
    tiny_suite: TaskSuite, shared_sandbox: CachingSandbox
) -> None:
    config = SweepConfig(
        seed=_SEED, instances_per_family=1, budget_levels=_LEVELS, bootstrap_resamples=0
    )
    deps = SweepDeps(sandbox=shared_sandbox, generator=default_generator(config))
    with pytest.raises(ValueError, match="is not one of the configured levels"):
        run_ablations(tiny_suite, config, deps, budget=99)


def test_the_spread_of_a_knob_is_reportable(ablations: object) -> None:
    low, high = aggregate_at(ablations.rows, "c_puct")  # type: ignore[attr-defined]
    assert 0.0 <= low <= high <= 1.0
    with pytest.raises(ValueError, match="no rows for knob"):
        aggregate_at(ablations.rows, "no_such_knob")  # type: ignore[attr-defined]


def test_the_best_row_is_the_highest_solve_rate(ablations: object) -> None:
    rows = ablations.by_knob("c_puct")  # type: ignore[attr-defined]
    winner = best_value(rows)
    assert winner.stats.hidden_solve_rate == max(row.stats.hidden_solve_rate for row in rows)
    with pytest.raises(ValueError, match="no ablation rows"):
        best_value([])


def test_the_default_grids_span_both_sides_of_the_default() -> None:
    """An ablation grid that never crosses the default cannot tell you the default is right."""
    from verifier_guided_search.search.puct import DEFAULT_C_PUCT

    assert min(DEFAULT_C_PUCT_GRID) < DEFAULT_C_PUCT < max(DEFAULT_C_PUCT_GRID)
    assert 0.0 in DEFAULT_C_PUCT_GRID, "pure exploitation must be in the grid"
    assert min(DEFAULT_K_GRID) < 4 < max(DEFAULT_K_GRID)


def test_the_results_serialise_for_a_results_file(ablations: object) -> None:
    payload = ablations.model_dump(mode="json")  # type: ignore[attr-defined]
    assert payload["strategy"] == "mcts"
    assert payload["candidates_budget"] == _LEVELS[-1]
    assert len(payload["rows"]) == len(ablations.rows)  # type: ignore[attr-defined]
    assert payload["value_model_version"]
